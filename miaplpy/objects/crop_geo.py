#!/usr/bin/env python3
############################################################
# Program is part of MiaplPy                                #
# Copyright (c) 2022, Sara Mirzaee                          #
# Author: Sara Mirzaee                                      #
############################################################
import numpy as np
import h5py
import rasterio
from rasterio.windows import Window
from typing import Optional, List, Tuple
from pathlib import Path
import xarray as xr
import netCDF4
from osgeo import gdal, osr
from os import fspath
from compass.utils.helpers import bbox_to_utm
from pyproj import CRS
from mintpy.utils import ptime, attribute as attr
from miaplpy.objects.utils import read_attribute
import time
from pydantic import BaseModel
import rioxarray
import isce3
from compass.utils.h5_helpers import Meta, add_dataset_and_attrs

from mintpy.objects import (DATA_TYPE_DICT,
                            GEOMETRY_DSET_NAMES,
                            DSET_UNIT_DICT)

BOOL_ZERO = np.bool_(0)
INT_ZERO = np.int16(0)
FLOAT_ZERO = np.float32(0.0)
CPX_ZERO = np.complex64(0.0)

dataType = np.complex64

slcDatasetNames = ['slc']
DSET_UNIT_DICT['slc'] = 'i'
gdal.SetCacheMax(2**30)

class GeoGrid(BaseModel):
    class Config:
        extra = "allow"

def init_geocoded_dataset(grid_group, dataset_name, geo_grid, dtype, description, shape):
    cslc_ds = grid_group.require_dataset(dataset_name, dtype=dtype,
                                         shape=shape)
    cslc_ds.attrs['description'] = description
    dx = geo_grid.spacing_x
    x0 = geo_grid.start_x + 0.5 * dx
    xf = x0 + (geo_grid.width - 1) * dx
    x_vect = np.linspace(x0, xf, geo_grid.width, dtype=np.float64)

    # Compute y scale
    dy = geo_grid.spacing_y
    y0 = geo_grid.start_y + 0.5 * dy
    yf = y0 + (geo_grid.length - 1) * dy
    y_vect = np.linspace(y0, yf, geo_grid.length, dtype=np.float64)

    # following copied and pasted (and slightly modified) from:
    # https://github-fn.jpl.nasa.gov/isce-3/isce/wiki/CF-Conventions-and-Map-Projections
    x_ds = grid_group.require_dataset('x_coordinates', dtype='float64',
                                      data=x_vect, shape=x_vect.shape)
    y_ds = grid_group.require_dataset('y_coordinates', dtype='float64',
                                      data=y_vect, shape=y_vect.shape)

    # Mapping of dimension scales to datasets is not done automatically in HDF5
    # We should label appropriate arrays as scales and attach them to datasets
    # explicitly as show below.
    x_ds.make_scale()
    cslc_ds.dims[1].attach_scale(x_ds)
    y_ds.make_scale()
    cslc_ds.dims[0].attach_scale(y_ds)

    # Associate grid mapping with data - projection created later
    cslc_ds.attrs['grid_mapping'] = np.string_("projection")

    grid_meta_items = [
        Meta('x_spacing', geo_grid.spacing_x,
             'Spacing of the geographical grid along X-direction',
             {'units': 'meters'}),
        Meta('y_spacing', geo_grid.spacing_y,
             'Spacing of the geographical grid along Y-direction',
             {'units': 'meters'})
    ]
    for meta_item in grid_meta_items:
        add_dataset_and_attrs(grid_group, meta_item)

    # Set up osr for wkt
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(geo_grid.epsg)

    # Create a new single int dataset for projections
    projection_ds = grid_group.require_dataset('projection', (), dtype='i')
    projection_ds[()] = geo_grid.epsg

    # WGS84 ellipsoid
    projection_ds.attrs['semi_major_axis'] = 6378137.0
    projection_ds.attrs['inverse_flattening'] = 298.257223563
    projection_ds.attrs['ellipsoid'] = np.string_("WGS84")

    # Additional fields
    projection_ds.attrs['epsg_code'] = geo_grid.epsg

    # CF 1.7+ requires this attribute to be named "crs_wkt"
    # spatial_ref is old GDAL way. Using that for testing only.
    # For NISAR replace with "crs_wkt"
    projection_ds.attrs['spatial_ref'] = np.string_(srs.ExportToWkt())

    # UTM zones
    if (geo_grid.epsg > 32600 and geo_grid.epsg < 32661) or \
         (geo_grid.epsg > 32700 and geo_grid.epsg < 32761):
        # Set up grid mapping
        projection_ds.attrs['grid_mapping_name'] = np.string_('universal_transverse_mercator')
        projection_ds.attrs['utm_zone_number'] = geo_grid.epsg % 100

        # Setup units for x and y
        x_ds.attrs['standard_name'] = np.string_("projection_x_coordinate")
        x_ds.attrs['long_name'] = np.string_("x coordinate of projection")
        x_ds.attrs['units'] = np.string_("m")

        y_ds.attrs['standard_name'] = np.string_("projection_y_coordinate")
        y_ds.attrs['long_name'] = np.string_("y coordinate of projection")
        y_ds.attrs['units'] = np.string_("m")

    return cslc_ds


class cropSLC:
    def __init__(self, pairs_dict: Optional[List[Path]] = None,
                 geo_bbox: Optional[Tuple[float, float, float, float]] = None):
        self.pairsDict = pairs_dict
        self.name = 'slc'
        self.dates = sorted([date for date in self.pairsDict.keys()])
        self.dsNames = list(self.pairsDict[self.dates[0]].datasetDict.keys())
        self.dsNames = [i for i in slcDatasetNames if i in self.dsNames]
        self.numSlc = len(self.pairsDict)
        self.bperp = np.zeros(self.numSlc)
        dsname0 = self.pairsDict[self.dates[0]].datasetDict['slc']
        self.geo_bbox = geo_bbox
        self.bb_utm = None
        self.crs, self.geotransform, self.extent, self.pixel_width, self.pixel_height, self.geogrid = self.get_transform(dsname0)
        #if geo_bbox is None:
        #    self.bb_utm = self.extent
        #else:
        #    self.bb_utm = bbox_to_utm(self.geo_bbox, epsg_src=4326, epsg_dst=self.crs.to_epsg())
        self.rdr_bbox = self.get_rdr_bbox()
        self.length, self.width = self.get_size()



    def get_transform(self, src_file):
        geogrid = GeoGrid
        dask_chunks = (1, 128 * 10, 128 * 10)
        inp_file = 'NETCDF:{}:/science/SENTINEL1/CSLC/grids/VV'.format(src_file)
        da_ref = rioxarray.open_rasterio(inp_file, chunks=dask_chunks).sel(band=1)
        if self.geo_bbox is None:
            self.bb_utm = da_ref.rio.bounds()
        else:
            self.bb_utm = bbox_to_utm(self.geo_bbox, epsg_src=4326, epsg_dst=da_ref.rio.crs.to_epsg())
        da_ref = da_ref.sel(x=slice(self.bb_utm[0], self.bb_utm[2]), y=slice(self.bb_utm[3], self.bb_utm[1]))
        crs = da_ref.rio.crs
        shape = da_ref.shape
        x_origin = da_ref.x[0]
        y_origin = da_ref.y[-1]
        pixel_width = da_ref.x[1] - da_ref.x[0]
        pixel_height = da_ref.y[0] - da_ref.y[1]
        geotransform = (x_origin, pixel_width, 0, y_origin, 0, pixel_height)
        extent = (0, 0, shape[1], shape[0])
        geogrid.spacing_x = pixel_width
        geogrid.spacing_y = pixel_height
        geogrid.start_x = x_origin
        geogrid.start_y = y_origin
        geogrid.length = len(da_ref.y)
        geogrid.width = len(da_ref.x)
        geogrid.epsg = da_ref.rio.crs.to_epsg()

        return crs, geotransform, extent, pixel_width, pixel_height, geogrid

    def get_subset_transform(self):
        # Define the cropping extent
        width = self.rdr_bbox[2] - self.rdr_bbox[0]
        length = self.rdr_bbox[3] - self.rdr_bbox[1]
        crop_extent = (self.bb_utm[0], self.bb_utm[1], self.bb_utm[2], self.bb_utm[3])
        crop_transform = rasterio.transform.from_bounds(self.bb_utm[0],
                                                        self.bb_utm[1],
                                                        self.bb_utm[2],
                                                        self.bb_utm[3], width, length)
        return crop_transform, crop_extent

    def get_rdr_bbox(self):
        # calculate the image coordinates
        col1 = abs(int((self.bb_utm[0] - self.geotransform[0]) / self.geotransform[1]))
        col2 = abs(int((self.bb_utm[2] - self.geotransform[0]) / self.geotransform[1]))
        row1 = abs(int((self.bb_utm[1] - self.geotransform[3]) / self.geotransform[5]))
        row2 = abs(int((self.bb_utm[3] - self.geotransform[3]) / self.geotransform[5]))

        if col2 > self.extent[2]:
            col2 = self.extent[2]
            bb_utm2 = col2 * self.geotransform[1] + self.geotransform[0]
            self.bb_utm = (self.bb_utm[0], self.bb_utm[1], bb_utm2, self.bb_utm[3])
        if row2 > self.extent[3]:
            row2 = self.extent[3]
            bb_utm3 = row2 * self.geotransform[5] + self.geotransform[3]
            self.bb_utm = (self.bb_utm[0], self.bb_utm[1], self.bb_utm[2], bb_utm3)

        if col1 < 0:
            col1 = 0
            bb_utm0 = self.geotransform[0]
            self.bb_utm = (bb_utm0, self.bb_utm[1], self.bb_utm[2], self.bb_utm[3])
        if row1 < 0:
            row1 = 0
            bb_utm1 = self.geotransform[3]
            self.bb_utm = (self.bb_utm[0], bb_utm1, self.bb_utm[2], self.bb_utm[3])

        return col1, row1, col2, row2

    def get_size(self):
        length = self.rdr_bbox[3] - self.rdr_bbox[1]
        width = self.rdr_bbox[2] - self.rdr_bbox[0]
        return length, width


    def get_date_list(self):
        self.dateList = sorted([date for date in self.pairsDict.keys()])
        return self.dateList


    def read_subset(self, slc_file):

        with h5py.File(slc_file, 'r') as f:
            subset_slc = f['science']['SENTINEL1']['CSLC']['grids']['VV'][self.rdr_bbox[1]:self.rdr_bbox[3],
                  self.rdr_bbox[0]:self.rdr_bbox[2]]

        return subset_slc

    def get_metadata(self):
        slcObj = [v for v in self.pairsDict.values()][0]
        self.metadata = slcObj.get_metadata()
        if 'UNIT' in self.metadata.keys():
            self.metadata.pop('UNIT')
        return self.metadata

    def write2hdf5(self, outputFile='slcStack.h5', access_mode='a', compression=None, extra_metadata=None):

        dsNames = [i for i in slcDatasetNames if i in self.dsNames]
        maxDigit = max([len(i) for i in dsNames])
        self.outputFile = outputFile

        dask_chunks = (1, 128 * 10, 128 * 10)

        f = h5py.File(self.outputFile, access_mode)
        dynamic_layer_group = f.require_group(f'dynamic_layers')

        print('create HDF5 file {} with {} mode'.format(self.outputFile, access_mode))
        dsName = 'slc'
        dsShape = (self.numSlc, self.geogrid.length, self.geogrid.width)
        dsDataType = dataType
        dsCompression = compression

        if dsName in dynamic_layer_group:
            slc_ds = dynamic_layer_group[dsName]
        else:
            #ds = dynamic_layer_group.create_dataset(dsName,
            #                      shape=dsShape,
            #                      maxshape=(None, dsShape[1], dsShape[2]),
            #                      dtype=dsDataType,
            #                      chunks=True,
            #                      compression=dsCompression)
            slc_ds = init_geocoded_dataset(dynamic_layer_group, dsName, self.geogrid, np.complex64,
                                           np.string_(dsName), dsShape)

            # Setup units for x and y
            slc_ds.attrs['standard_name'] = np.string_("slc_images")
            slc_ds.attrs['long_name'] = np.string_("Coregistered_slc_images")
            slc_ds.attrs['units'] = np.string_("i")


        output_raster = isce3.io.Raster(f"IH5:::ID={slc_ds.id.id}".encode("utf-8"),
                                        update=True)

        prog_bar = ptime.progressBar(maxValue=self.numSlc)

        # 3D datasets containing slc.
        for i in range(self.numSlc):
            slcObj = self.pairsDict[self.dates[i]]
            slc_file = slcObj.datasetDict['slc']
            #subset = self.read_subset(slc_file)
            inp_file = 'NETCDF:{}:/science/SENTINEL1/CSLC/grids/VV'.format(slc_file)
            subset = rioxarray.open_rasterio(inp_file, chunks=dask_chunks).sel(band=1, x=slice(self.bb_utm[0],
                                                                                               self.bb_utm[2]),
                                                                               y=slice(self.bb_utm[3], self.bb_utm[1]))
            slc_ds[i, :, :] = subset[:, :]
            self.bperp[i] = slcObj.get_perp_baseline()
            prog_bar.update(i + 1, suffix='{}'.format(self.dates[i][0]))
        prog_bar.close()
        slc_ds.attrs['MODIFICATION_TIME'] = str(time.time())

        output_raster.set_geotransform(self.geotransform)
        output_raster.set_epsg(self.geogrid.epsg)
        del output_raster

        ###############################
        # 1D dataset containing dates of all images
        dsName = 'date'
        dsDataType = np.string_
        dsShape = (self.numSlc, 1)
        print('create dataset /{d:<{w}} of {t:<25} in size of {s}'.format(d=dsName,
                                                                          w=maxDigit,
                                                                          t=str(dsDataType),
                                                                          s=dsShape))

        data = np.array(self.dates, dtype=dsDataType)
        if not dsName in f.keys():
            f.create_dataset(dsName, data=data)

        ###############################
        # 1D dataset containing perpendicular baseline of all pairs
        dsName = 'bperp'
        dsDataType = np.float32
        dsShape = (self.numSlc,)
        print('create dataset /{d:<{w}} of {t:<25} in size of {s}'.format(d=dsName,
                                                                          w=maxDigit,
                                                                          t=str(dsDataType),
                                                                          s=dsShape))
        data = np.array(self.bperp, dtype=dsDataType)
        if not dsName in f.keys():
            f.create_dataset(dsName, data=data)

        ###############################
        # Attributes
        self.get_metadata()
        if extra_metadata:
            self.metadata.update(extra_metadata)
            # print('add extra metadata: {}'.format(extra_metadata))
        self.metadata = attr.update_attribute4subset(self.metadata, self.rdr_bbox)

        self.metadata['FILE_TYPE'] = 'timeseries'  # 'slc'
        for key, value in self.metadata.items():
            f.attrs[key] = value
        '''
        crop_transform, crop_extent = self.get_subset_transform()
        crs = self.crs.to_string()

        # Write georeference information
        if "georeference" in f:
            grp = f["georeference"]
        else:
            grp = f.create_group("georeference")

        if not "transform" in grp.keys():
            grp.create_dataset("transform", data=crop_transform, dtype=np.float32)
        grp.attrs["crs"] = crs.encode('utf-8')
        grp.attrs["extent"] = crop_extent
        grp.attrs["pixel_size_x"] = self.pixel_width
        grp.attrs["pixel_size_y"] = self.pixel_height
        '''
        f.close()
        print('Finished writing to {}'.format(self.outputFile))
        return self.outputFile














'''

#############################
# Open the geocoded data file
src_file = rasterio.open('geocoded_data.tif')

# Read the relevant metadata
srs = src_file.crs.to_string()
extent = src_file.bounds
pixel_size = src_file.res[0]

# Define the cropping extent
crop_extent = (xmin, ymin, xmax, ymax)  # replace with your desired extent
crop_transform = rasterio.transform.from_bounds(*crop_extent, src_file.transform)

# Read the data within the cropping extent
window = src_file.window(*crop_extent)
data = src_file.read(1, window=window)

# Create the HDF5 file
h5_file = h5py.File('cropped_data.h5', 'w')

# Define the data structure
dtype = data.dtype
chunks = (256, 256)  # adjust the chunk size as needed
shape = data.shape

dataset = h5_file.create_dataset('data', shape, dtype=dtype, chunks=chunks)

# Write the data
dataset.write_direct(data)

# Write georeference information
grp = h5_file.create_group("georeference")
grp.create_dataset("transform", data=crop_transform, dtype=crop_transform.dtype)
grp.attrs["crs"] = srs
grp.attrs["extent"] = extent
grp.attrs["pixel_size"] = pixel_size

h5_file.close()
##############


import netCDF4

# Open the HDF5 NetCDF file
nc_file = netCDF4.Dataset('geocoded_data.h5', 'r')

# Read the georeference information
transform = nc_file['georeference/transform'][:]
srs = nc_file['georeference'].getncattr('crs')
extent = nc_file['georeference'].getncattr('extent')
pixel_size = nc_file['georeference'].getncattr('pixel_size')

# Close the HDF5 NetCDF file
nc_file.close()
'''