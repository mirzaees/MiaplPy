#! /usr/bin/env python3
############################################################
# Copyright(c) 2017, Sara Mirzaee                          #
############################################################

import os
import sys

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

blockPrint()
import datetime
import numpy as np
from miaplpy.objects.arg_parser import MiaplPyParser
import h5py
from math import sqrt, exp
from osgeo import gdal, osr
import dask.array as da
from pyproj import CRS
import mintpy.multilook as multilook

gdal.UseExceptions()

enablePrint()

DEFAULT_TILE_SIZE = [128, 128]
DEFAULT_ENVI_OPTIONS = (
        "INTERLEAVE=BIL",
        "SUFFIX=ADD"
    )

DEFAULT_TIFF_OPTIONS = (
    "COMPRESS=DEFLATE",
    "ZLEVEL=4",
    "TILED=YES",
    f"BLOCKXSIZE={DEFAULT_TILE_SIZE[1]}",
    f"BLOCKYSIZE={DEFAULT_TILE_SIZE[0]}",
)


def main(iargs=None):
    """
        Overwrite filtered SLC images in Isce merged/SLC directory.
    """

    Parser = MiaplPyParser(iargs, script='generate_interferograms')
    inps = Parser.parse()

    dateStr = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d:%H%M%S')

    if not iargs is None:
        msg = os.path.basename(__file__) + ' ' + ' '.join(iargs[:])
        string = dateStr + " * " + msg
        print(string)
    else:
        msg = os.path.basename(__file__) + ' ' + ' '.join(sys.argv[1::])
        string = dateStr + " * " + msg
        print(string)

    print(inps.out_dir)
    os.makedirs(inps.out_dir, exist_ok=True)

    ifg_file = inps.out_dir + f"/{inps.reference}_{inps.secondary}.int.tif"
    cor_file = inps.out_dir + f"/{inps.reference}_{inps.secondary}.cor.tif"

    if inps.azlooks * inps.rglooks > 1:
        ifg_file_full = inps.out_dir + f"/{inps.reference}_{inps.secondary}_full.int.tif"
        run_inreferogram(inps, ifg_file_full)
        multilook.multilook_gdal(ifg_file_full, inps.azlooks, inps.rglooks, out_file=ifg_file)
        
    else:
        run_inreferogram(inps, ifg_file)

    window_size = (6, 12)
    estimate_correlation(ifg_file, cor_file, window_size)

    return


def run_inreferogram(inps, ifg_file):

    if os.path.exists(ifg_file):
        sys.exit()
        # return

    with h5py.File(inps.stack_file, 'r') as ds:
        attrs = dict(ds.attrs)
        if 'spatial_ref' in attrs.keys():
            projection = attrs['spatial_ref']  # CRS.from_wkt(attrs['spatial_ref'])
            geotransform = [attrs['X_FIRST'], attrs['X_STEP'], 0, attrs['Y_FIRST'], 0, attrs['Y_STEP']]
            geotransform = [float(x) for x in geotransform]
            nodata = np.nan
            drivername = 'GTiff'
            default_options = DEFAULT_TIFF_OPTIONS
        else:
            geotransform = 'None'
            projection = 'None'
            nodata = np.nan
            drivername = 'ENVI'
            default_options = DEFAULT_ENVI_OPTIONS

        date_list = np.array([x.decode('UTF-8') for x in ds['date'][:]])
        ref_ind = np.where(date_list == inps.reference)[0]
        sec_ind = np.where(date_list == inps.secondary)[0]
        phase_series = ds['phase']

        length = phase_series.shape[1]
        width = phase_series.shape[2]

        box_size = 3000

        dtype = gdal.GDT_CFloat32
        driver = gdal.GetDriverByName(drivername)
        out_raster = driver.Create(ifg_file, width, length, 1, dtype, default_options)
        band = out_raster.GetRasterBand(1)

        for i in range(0, length, box_size):
            for j in range(0, width, box_size):
                ref_phase = phase_series[ref_ind, i:i+box_size, j:j+box_size].squeeze()
                sec_phase = phase_series[sec_ind, i:i+box_size, j:j+box_size].squeeze()

                ifg = np.exp(1j * np.angle(np.exp(1j * ref_phase) * np.exp(-1j * sec_phase)))
                band.WriteArray(ifg, j, i)

        band.SetNoDataValue(np.nan)
        out_raster.FlushCache()
        out_raster = None

    write_projection(ifg_file, geotransform, projection, nodata)

    return 


def write_projection(dst_file, geotransform, projection, nodata) -> None:

    if not geotransform == 'None':
        ds_dst = gdal.Open(dst_file, gdal.GA_Update)
        ds_dst.SetGeoTransform(geotransform)
        ds_dst.SetProjection(projection)
        ds_dst.GetRasterBand(1).SetNoDataValue(nodata)
        ds_src = ds_dst = None
    return


def estimate_correlation(ifg_file, cor_file, window_size):
    #if os.path.exists(cor_file):
    #   return
    
    ds_src = gdal.Open(ifg_file, gdal.GA_Update)
    projection = ds_src.GetProjection()
    geotransform = ds_src.GetGeoTransform()
    nodata = ds_src.GetRasterBand(1).GetNoDataValue()

    if geotransform is None:
        driver_name = 'ENVI'
        default_options = DEFAULT_ENVI_OPTIONS
    else:
        driver_name = 'GTiff'
        default_options = DEFAULT_TIFF_OPTIONS

    ds = gdal.Open(ifg_file)
    phase = ds.GetRasterBand(1).ReadAsArray()
    length, width = ds.RasterYSize, ds.RasterXSize
    nan_mask = np.isnan(phase)
    zero_mask = np.angle(phase) == 0
    image = np.exp(1j * np.nan_to_num(np.angle(phase)))

    col_size, row_size = window_size
    row_pad = row_size // 2
    col_pad = col_size // 2

    image_padded = np.pad(
        image, ((row_pad, row_pad), (col_pad, col_pad)), mode="constant"
    )

    integral_img = np.cumsum(np.cumsum(image_padded, axis=0), axis=1)

    window_mean = (
            integral_img[row_size:, col_size:]
            - integral_img[:-row_size, col_size:]
            - integral_img[row_size:, :-col_size]
            + integral_img[:-row_size, :-col_size]
    )
    window_mean /= row_size * col_size

    cor = np.clip(np.abs(window_mean), 0, 1)
    cor[nan_mask] = np.nan
    cor[zero_mask] = 0

    dtype = gdal.GDT_Float32
    driver = gdal.GetDriverByName(driver_name)
    out_raster = driver.Create(cor_file, width, length, 1, dtype, default_options)
    band = out_raster.GetRasterBand(1)
    band.WriteArray(cor, 0, 0)
    band.SetNoDataValue(np.nan)
    out_raster.FlushCache()
    out_raster = None

    write_projection(cor_file, geotransform, projection, nodata)

    return


if __name__ == '__main__':
    main()


