#!/usr/bin/env python3
############################################################
# Program is part of MiaplPy                                #
# Copyright (c) 2022, Sara Mirzaee                          #
# Author: Sara Mirzaee                                      #
############################################################
import numpy as np
import rasterio
from rasterio.windows import Window
from typing import Optional, List, Tuple, Union
from pathlib import Path
import xarray as xr
import h5py
from osgeo import gdal, osr
from os import fspath
import datetime
from compass.utils.helpers import bbox_to_utm
from pyproj import CRS
from mintpy.utils import ptime, attribute as attr
from miaplpy.objects.utils import read_attribute
import time
import rioxarray
import isce3
from typing import Optional, Any
# from compass.utils.h5_helpers import Meta, add_dataset_and_attrs

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


HDF5_OPTS = dict(
    # https://docs.h5py.org/en/stable/high/dataset.html#filter-pipeline
    chunks=(1280, 1280),
    compression="gzip",
    compression_opts=4,
    shuffle=True,
    dtype=np.complex64
)


def create_grid_mapping(group, crs: CRS, gt: list):
    """Set up the grid mapping variable."""
    # https://github.com/corteva/rioxarray/blob/21284f67db536d9c104aa872ab0bbc261259e59e/rioxarray/rioxarray.py#L34
    dset = group.create_dataset('spatial_ref', (), dtype=int)
    dset.attrs.update(crs.to_cf())
    # Also add the GeoTransform
    gt_string = " ".join([str(x) for x in gt])
    dset.attrs.update(
        dict(
            GeoTransform=gt_string,
            units="unitless",
            long_name=(
                "Dummy variable containing geo-referencing metadata in attributes"
            ),
        )
    )

    return dset


def create_tyx_dsets(
    group,
    gt: list,
    times: list,
    shape: tuple):
    """Create the time, y, and x coordinate datasets."""
    y, x = create_yx_arrays(gt, shape)
    times, calendar, units = create_time_array(times)

    #if not group.dimensions:
    #    group.dimensions = dict(time=times.size, y=y.size, x=x.size)
    # Create the datasets
    t_ds = group.create_dataset("time", (len(times),), data=times, dtype=float)
    y_ds = group.create_dataset("y", (len(y),), data=y, dtype=float)
    x_ds = group.create_dataset("x", (len(x),), data=x, dtype=float)

    t_ds.attrs["standard_name"] = "time"
    t_ds.attrs["long_name"] = "time"
    t_ds.attrs["calendar"] = calendar
    t_ds.attrs["units"] = units

    for name, ds in zip(["y", "x"], [y_ds, x_ds]):
        ds.attrs["standard_name"] = f"projection_{name}_coordinate"
        ds.attrs["long_name"] = f"{name.replace('_', ' ')} coordinate of projection"
        ds.attrs["units"] = "m"

    return t_ds, y_ds, x_ds


def create_yx_arrays(
    gt: list, shape: tuple
) -> tuple:
    """Create the x and y coordinate datasets."""
    ysize, xsize = shape
    # Parse the geotransform
    x_origin, x_res, _, y_origin, _, y_res = gt
    # Make the x/y arrays
    # Note that these are the center of the pixels, whereas the GeoTransform
    # is the upper left corner of the top left pixel.
    y = np.arange(y_origin + y_res / 2, y_origin + y_res * ysize, y_res)
    x = np.arange(x_origin + x_res / 2, x_origin + x_res * xsize, x_res)
    return y, x


def create_time_array(dates: datetime.datetime):
    # 'calendar': 'standard',
    # 'units': 'seconds since 2017-02-03 00:00:00.000000'
    # Create the time array
    times = [datetime.datetime.strptime(dd, '%Y%m%d') for dd in dates]
    since_time = times[0]
    time = np.array([(t - since_time).total_seconds() for t in times])
    calendar = "standard"
    units = f"seconds since {since_time.strftime('%Y-%m-%d %H:%M:%S.%f')}"
    return time, calendar, units


def add_complex_ctype(h5file: h5py.File):
    """Add the complex64 type to the root of the HDF5 file.
    This is required for GDAL to recognize the complex data type.
    """
    with h5py.File(h5file, "a") as hf:
        if "complex64" in hf["/"]:
            return
        ctype = h5py.h5t.py_create(np.complex64)
        ctype.commit(hf["/"].id, np.string_("complex64"))


def create_geo_dataset_3d(
    *,
    group,
    name: str,
    description: str,
    fillvalue: float,
    attrs: dict,
    timelength: int,
    dtype,):

    dimensions = ["time", "y", "x"]
    if attrs is None:
        attrs = {}
    attrs.update(long_name=description)

    options = HDF5_OPTS
    options["chunks"] = (timelength, *options["chunks"])
    options['dtype'] = dtype

    dset = group.create_dataset(
        name,
        ndim=3,
        fillvalue=fillvalue,
        **options,
    )
    dset.attrs.update(attrs)
    dset.attrs["grid_mapping"] = 'spatial_ref'
    return dset


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
        self.crs, self.geotransform, self.extent, self.shape = self.get_transform(dsname0)
        #if geo_bbox is None:
        #    self.bb_utm = self.extent
        #else:
        #    self.bb_utm = bbox_to_utm(self.geo_bbox, epsg_src=4326, epsg_dst=self.crs.to_epsg())
        #
        self.length, self.width = self.shape
        self.rdr_bbox = self.get_rdr_bbox()
        self.lengthc, self.widthc = self.get_size()



    def get_transform(self, src_file):
        #geogrid = GeoGrid
        dask_chunks = (1, 128 * 10, 128 * 10)
        inp_file = 'NETCDF:{}:/science/SENTINEL1/CSLC/grids/VV'.format(src_file)
        da_ref = rioxarray.open_rasterio(inp_file, chunks=dask_chunks).sel(band=1)
        shape = da_ref.shape
        if self.geo_bbox is None:
            self.bb_utm = da_ref.rio.bounds()
        else:
            self.bb_utm = bbox_to_utm(self.geo_bbox, epsg_src=4326, epsg_dst=da_ref.rio.crs.to_epsg())
        da_ref = da_ref.sel(x=slice(self.bb_utm[0], self.bb_utm[2]), y=slice(self.bb_utm[3], self.bb_utm[1]))
        # crs = da_ref.rio.crs
        gt = da_ref.rio.transform()
        geotransform = (gt[2], gt[0], 0, gt[5], 0, gt[4])
        extent = da_ref.rio.bounds()
        crs = CRS.from_epsg(da_ref.rio.crs.to_epsg())
        return crs, geotransform, extent, shape

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
        col2 = int(abs(np.ceil((self.bb_utm[2] - self.geotransform[0]) / self.geotransform[1])))
        row2 = int(abs(np.ceil((self.bb_utm[1] - self.geotransform[3]) / self.geotransform[5])))
        row1 = abs(int((self.bb_utm[3] - self.geotransform[3]) / self.geotransform[5]))

        if col2 > self.width:
            col2 = self.width
            bb_utm2 = col2 * self.geotransform[1] + self.geotransform[0]
            self.bb_utm = (self.bb_utm[0], self.bb_utm[1], bb_utm2, self.bb_utm[3])
        if row2 > self.length:
            row2 = self.length
            bb_utm1 = row2 * self.geotransform[5] + self.geotransform[3]
            self.bb_utm = (self.bb_utm[0], bb_utm1, self.bb_utm[2], self.bb_utm[3])

        if col1 < 0:
            col1 = 0
            bb_utm0 = self.geotransform[0]
            self.bb_utm = (bb_utm0, self.bb_utm[1], self.bb_utm[2], self.bb_utm[3])
        if row1 < 0:
            row1 = 0
            bb_utm3 = self.geotransform[3]
            self.bb_utm = (self.bb_utm[0], self.bb_utm[1], self.bb_utm[2], bb_utm3)
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

    def write2hdf5(self, outputFile='slcStack.nc', access_mode='a', compression=None, extra_metadata=None):

        dsNames = [i for i in slcDatasetNames if i in self.dsNames]
        maxDigit = max([len(i) for i in dsNames])
        self.outputFile = outputFile
        print('create HDF5 file {} with {} mode'.format(self.outputFile, access_mode))
        dsName = 'slc'

        f = h5py.File(self.outputFile, access_mode)
        print('create HDF5 file {} with {} mode'.format(self.outputFile, access_mode))
        create_grid_mapping(group=f, crs=self.crs, gt=list(self.geotransform))
        create_tyx_dsets(group=f, gt=list(self.geotransform), times=self.dates, shape=(self.lengthc, self.widthc))

        dsShape = (self.numSlc, self.lengthc, self.widthc)
        dsDataType = dataType
        dsCompression = compression

        self.bperp = np.zeros(self.numSlc)

        print(('create dataset /{d:<{w}} of {t:<25} in size of {s}'
               ' with compression = {c}').format(d=dsName,
                                                 w=maxDigit,
                                                 t=str(dsDataType),
                                                 s=dsShape,
                                                 c=dsCompression))

        if dsName in f.keys():
            ds = f[dsName]
        else:
            ds = f.create_dataset(dsName,
                                  shape=dsShape,
                                  maxshape=(None, dsShape[1], dsShape[2]),
                                  dtype=dsDataType,
                                  chunks=True,
                                  compression=dsCompression)

            ds.attrs.update(long_name="SLC complex data")
            ds.attrs["grid_mapping"] = 'spatial_ref'

        prog_bar = ptime.progressBar(maxValue=self.numSlc)
        for i in range(self.numSlc):
            box = self.rdr_bbox
            slcObj = self.pairsDict[self.dates[i]]
            dsSlc, metadata = slcObj.read(dsName, box=box)
            ds[i, :, :] = dsSlc[:, :]

            self.bperp[i] = slcObj.get_perp_baseline()
            prog_bar.update(i + 1, suffix='{}'.format(self.dates[i][0]))

        prog_bar.close()
        ds.attrs['MODIFICATION_TIME'] = str(time.time())

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