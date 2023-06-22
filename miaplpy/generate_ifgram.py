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
#import isceobj
import numpy as np
from miaplpy.objects.arg_parser import MiaplPyParser
import h5py
from math import sqrt, exp
from osgeo import gdal
import dask.array as da
from pyproj import CRS

enablePrint()

DEFAULT_ENVI_OPTIONS = (
        "INTERLEAVE=BIL",
        "SUFFIX=ADD"
    )


def main(iargs=None):
    """
        Overwrite filtered SLC images in ISCE merged/SLC directory.
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

    ifg_file = inps.out_dir + f"/{inps.reference}_{inps.secondary}.int"
    cor_file = inps.out_dir + f"/{inps.reference}_{inps.secondary}.cor"

    run_inreferogram(inps, ifg_file)
    window_size = (6, 12)
    estimate_correlation(ifg_file, cor_file, window_size)

    return


def estimate_correlation(ifg_file, cor_file, window_size):
    if os.path.exists(cor_file):
        return
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
    driver = gdal.GetDriverByName('ENVI')
    out_raster = driver.Create(cor_file, width, length, 1, dtype, DEFAULT_ENVI_OPTIONS)
    band = out_raster.GetRasterBand(1)
    band.WriteArray(cor, 0, 0)
    band.SetNoDataValue(np.nan)
    out_raster.FlushCache()
    out_raster = None

    write_projection(ifg_file, cor_file)

    return

def run_inreferogram(inps, ifg_file):

    if os.path.exists(ifg_file):
        return

    with h5py.File(inps.stack_file, 'r') as ds:
        date_list = np.array([x.decode('UTF-8') for x in ds['date'][:]])
        ref_ind = np.where(date_list == inps.reference)[0]
        sec_ind = np.where(date_list == inps.secondary)[0]
        phase_series = ds['phase']

        length = phase_series.shape[1]
        width = phase_series.shape[2]

        box_size = 3000

        dtype = gdal.GDT_CFloat32
        driver = gdal.GetDriverByName('ENVI')
        out_raster = driver.Create(ifg_file, width, length, 1, dtype, DEFAULT_ENVI_OPTIONS)
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

    write_projection(inps.stack_file, ifg_file)

    return


def write_projection(src_file, dst_file) -> None:
    if src_file.endswith('.h5'):
        with h5py.File(src_file, 'r') as ds:
            if 'spatial_ref' in ds:
                geotransform = tuple([int(float(x)) for x in ds['spatial_ref'].attrs['GeoTransform'].split()])
                projection = CRS.from_wkt(ds['spatial_ref'].attrs['crs_wkt']).to_wkt()
            else:
                geotransform = (0, 1, 0, 0, 0, 1)
                projection = CRS.from_epsg(4326).to_wkt()

            nodata = np.nan
    else:
        ds_src = gdal.Open(src_file, gdal.GA_Update)
        projection = ds_src.GetProjection()
        geotransform = ds_src.GetGeoTransform()
        nodata = ds_src.GetRasterBand(1).GetNoDataValue()

    ds_dst = gdal.Open(dst_file, gdal.GA_Update)
    ds_dst.SetGeoTransform(geotransform)
    ds_dst.SetProjection(projection)
    ds_dst.GetRasterBand(1).SetNoDataValue(nodata)
    ds_src = ds_dst = None
    return

def run_interferogram_2(inps):
    # sequential = True
    sequential = False
    out_file = inps.out_dir + '.tif'
    dask_chunks = (128 * 10, 128 * 10)

    # sequential interferograms
    with h5py.File(inps.stack_file, 'r') as ds:
        if 'spatial_ref' in ds.attrs:
            projection = CRS.from_wkt(ds['spatial_ref'].attrs['crs_wkt'])
            geotransform = tuple([int(float(x)) for x in ds['spatial_ref'].attrs['GeoTransform'].split()])

        date_list = np.array([x.decode('UTF-8') for x in ds['date'][:]])
        ref_ind = np.where(date_list == inps.reference)[0][0]
        sec_ind = np.where(date_list == inps.secondary)[0][0]
        phase_series = ds['phase_seq']
        amplitudes = ds['amplitude_seq']
        ref_phase = da.from_array(phase_series[ref_ind, :, :], chunks=dask_chunks)
        sec_phase = da.from_array(phase_series[sec_ind, :, :], chunks=dask_chunks)
        ref_amplitude = da.from_array(amplitudes[ref_ind, :, :], chunks=dask_chunks)
        sec_amplitude = da.from_array(amplitudes[sec_ind, :, :], chunks=dask_chunks)
        denom = np.sqrt(ref_amplitude**2 * sec_amplitude**2)
        if sequential:
            if ref_ind - sec_ind == 1:
                numer = ref_amplitude * np.exp(1j * ref_phase)
            if ref_ind - sec_ind == -1:
                numer = sec_amplitude * np.exp(-1j * sec_phase)
            if (ref_ind - sec_ind) > 1:
                phase = ref_phase + sec_phase
                for ti in range(sec_ind + 1, ref_ind):
                    sec_phase = da.from_array(phase_series[ti, :, :], chunks=dask_chunks)
                    phase += sec_phase
                numer = (ref_amplitude * sec_amplitude) * np.exp(1j * phase)
            if (ref_ind - sec_ind) < 1:
                phase = ref_phase + sec_phase
                for ti in range(ref_ind + 1, sec_ind):
                    sec_phase = da.from_array(phase_series[ti, :, :], chunks=dask_chunks)
                    phase += sec_phase
                numer = (ref_amplitude * sec_amplitude) * np.exp(-1j * phase)
        else:
            numer = (ref_amplitude * sec_amplitude) * np.exp(1j * (ref_phase - sec_phase))

    ifg = (numer / denom).astype("complex64")
    # Make sure we didn't lose the geo information
    ifg.rio.write_crs(projection, inplace=True)
    ifg.rio.write_nodata(float("NaN"), inplace=True)
    ifg.rio.to_raster(out_file, tiled=True)
    return out_file

def run_coherence(ifg_filename):
    outfile = ifg_filename.split('.')[0] + '_cor.tif'
    da_ifg = rioxarray.open_rasterio(ifg_filename, chunks=True)
    np.abs(da_ifg).rio.to_raster(outfile, driver="GTiff", suffix="add")
    return

def run_interferogram_old(inps, resampName):
    if inps.azlooks * inps.rglooks > 1:
        extention = '.ml.slc'
    else:
        extention = '.slc'

    #sequential = True
    sequential = False

    if sequential:
        # sequential interferograms
        with h5py.File(inps.stack_file, 'r') as ds:
            date_list = np.array([x.decode('UTF-8') for x in ds['date'][:]])
            ref_ind = np.where(date_list == inps.reference)[0][0]
            sec_ind = np.where(date_list == inps.secondary)[0][0]
            phase_series = ds['phase_seq']
            amplitudes = ds['amplitude_seq']
            length = phase_series.shape[1]
            width = phase_series.shape[2]

            resampInt = resampName + '.int'

            intImage = isceobj.createIntImage()
            intImage.setFilename(resampInt)
            intImage.setAccessMode('write')
            intImage.setWidth(width)
            intImage.setLength(length)
            intImage.createImage()

            out_ifg = intImage.asMemMap(resampInt)
            box_size = 3000
            num_row = int(np.ceil(length / box_size))
            num_col = int(np.ceil(width / box_size))

            for i in range(num_row):
                for k in range(num_col):
                    row_1 = i * box_size
                    row_2 = i * box_size + box_size
                    col_1 = k * box_size
                    col_2 = k * box_size + box_size
                    if row_2 > length:
                        row_2 = length
                    if col_2 > width:
                        col_2 = width

                    ref_phase = phase_series[ref_ind, row_1:row_2, col_1:col_2].reshape(row_2 - row_1, col_2 - col_1)
                    sec_phase = phase_series[sec_ind, row_1:row_2, col_1:col_2].reshape(row_2 - row_1, col_2 - col_1)
                    ref_amplitude = amplitudes[ref_ind, row_1:row_2, col_1:col_2].reshape(row_2 - row_1, col_2 - col_1)
                    sec_amplitude = amplitudes[sec_ind, row_1:row_2, col_1:col_2].reshape(row_2 - row_1, col_2 - col_1)
                    denom = 1 #np.sqrt(ref_amplitude ** 2 * sec_amplitude ** 2)

                    if ref_ind - sec_ind == 1:
                        ifg = (ref_amplitude * sec_amplitude) * np.exp(1j * ref_phase)
                    if ref_ind - sec_ind == -1:
                        ifg = (ref_amplitude * sec_amplitude) * np.exp(-1j * sec_phase)
                    if (ref_ind - sec_ind) > 1:
                        phase = ref_phase + sec_phase
                        for ti in range(sec_ind+1, ref_ind):
                            sec_phase = phase_series[ti, row_1:row_2, col_1:col_2].reshape(row_2 - row_1, col_2 - col_1)
                            phase += sec_phase
                        ifg = (ref_amplitude * sec_amplitude) * np.exp(1j * phase)
                    if (ref_ind - sec_ind) < 1:
                        phase = ref_phase + sec_phase
                        for ti in range(ref_ind + 1, sec_ind):
                            sec_phase = phase_series[ti, row_1:row_2, col_1:col_2].reshape(row_2 - row_1, col_2 - col_1)
                            phase += sec_phase
                        ifg = (ref_amplitude * sec_amplitude) * np.exp(-1j * phase)

                    #ifg[np.isnan(ifg)] = 0 + 1j*0
                    out_ifg[row_1:row_2, col_1:col_2, 0] = np.nan_to_num(ifg[:, :]/denom)

            intImage.renderHdr()
            intImage.finalizeImage()

    else:

        with h5py.File(inps.stack_file, 'r') as ds:
            date_list = np.array([x.decode('UTF-8') for x in ds['date'][:]])
            ref_ind = np.where(date_list == inps.reference)[0]
            sec_ind = np.where(date_list == inps.secondary)[0]
            phase_series = ds['phase']
            amplitudes = ds['amplitude']

            length = phase_series.shape[1]
            width = phase_series.shape[2]

            resampInt = resampName + '.int'

            intImage = isceobj.createIntImage()
            intImage.setFilename(resampInt)
            intImage.setAccessMode('write')
            intImage.setWidth(width)
            intImage.setLength(length)
            intImage.createImage()

            out_ifg = intImage.asMemMap(resampInt)
            box_size = 3000
            num_row = int(np.ceil(length / box_size))
            num_col = int(np.ceil(width / box_size))
            for i in range(num_row):
                for k in range(num_col):
                    row_1 = i * box_size
                    row_2 = i * box_size + box_size
                    col_1 = k * box_size
                    col_2 = k * box_size + box_size
                    if row_2 > length:
                        row_2 = length
                    if col_2 > width:
                        col_2 = width

                    ref_phase = phase_series[ref_ind, row_1:row_2, col_1:col_2].reshape(row_2 - row_1, col_2 - col_1)
                    sec_phase = phase_series[sec_ind, row_1:row_2, col_1:col_2].reshape(row_2 - row_1, col_2 - col_1)
                    ref_amplitude = amplitudes[ref_ind, row_1:row_2, col_1:col_2].reshape(row_2 - row_1, col_2 - col_1)
                    sec_amplitude = amplitudes[sec_ind, row_1:row_2, col_1:col_2].reshape(row_2 - row_1, col_2 - col_1)
                    denom = 1 #np.sqrt(ref_amplitude ** 2 * sec_amplitude ** 2)

                    ifg = (ref_amplitude * sec_amplitude) * np.exp(1j * (ref_phase - sec_phase)) / denom

                    # ifg[np.isnan(ifg)] = 0 + 1j * 0
                    out_ifg[row_1:row_2, col_1:col_2, 0] = np.nan_to_num(ifg[:, :])

            intImage.renderHdr()
            intImage.finalizeImage()

    return length, width



def runFilter(infile, outfile, filterStrength):
    from mroipac.filter.Filter import Filter

    # Initialize the flattened interferogram
    intImage = isceobj.createIntImage()
    intImage.load( infile + '.xml')
    intImage.setAccessMode('read')
    intImage.createImage()

    # Create the filtered interferogram
    filtImage = isceobj.createIntImage()
    filtImage.setFilename(outfile)
    filtImage.setWidth(intImage.getWidth())
    filtImage.setAccessMode('write')
    filtImage.createImage()

    objFilter = Filter()
    objFilter.wireInputPort(name='interferogram',object=intImage)
    objFilter.wireOutputPort(name='filtered interferogram', object=filtImage)
    objFilter.goldsteinWerner(alpha=filterStrength)

    intImage.finalizeImage()
    filtImage.finalizeImage()

def runFilterG(infile, outfile, filterStrength):

    # Initialize the flattened interferogram
    intImage = isceobj.createIntImage()
    intImage.load(infile + '.xml')
    intImage.setAccessMode('read')
    intImage.createImage()

    # Create the filtered interferogram
    filtImage = isceobj.createIntImage()
    filtImage.setFilename(outfile)
    filtImage.setWidth(intImage.getWidth())
    filtImage.setLength(intImage.getLength())
    filtImage.setAccessMode('write')
    filtImage.createImage()

    img = intImage.memMap(mode='r', band=0)
    original = np.fft.fft2(img[:, :])
    center = np.fft.fftshift(original)
    LowPassCenter = center * gaussianLP(100, img.shape)
    LowPass = np.fft.ifftshift(LowPassCenter)
    inverse_LowPass = np.fft.ifft2(LowPass)

    out_filtered = filtImage.asMemMap(outfile)
    out_filtered[:, :, 0] = inverse_LowPass[:, :]

    filtImage.renderHdr()

    intImage.finalizeImage()
    filtImage.finalizeImage()


def estCoherence(outfile, corfile):
    from mroipac.icu.Icu import Icu

    # Create phase sigma correlation file here
    filtImage = isceobj.createIntImage()
    filtImage.load(outfile + '.xml')
    filtImage.setAccessMode('read')
    filtImage.createImage()

    phsigImage = isceobj.createImage()
    phsigImage.dataType = 'FLOAT'
    phsigImage.bands = 1
    phsigImage.setWidth(filtImage.getWidth())
    phsigImage.setFilename(corfile)
    phsigImage.setAccessMode('write')
    phsigImage.createImage()

    icuObj = Icu(name='sentinel_filter_icu')
    icuObj.configure()
    icuObj.unwrappingFlag = False
    icuObj.useAmplitudeFlag = False
    # icuObj.correlationType = 'NOSLOPE'

    icuObj.icu(intImage=filtImage, phsigImage=phsigImage)
    phsigImage.renderHdr()

    filtImage.finalizeImage()
    phsigImage.finalizeImage()

def run_interpolation(filtifg, tcoh_file, length, width):
    from scipy.spatial import Delaunay
    from scipy.interpolate import LinearNDInterpolator

    ifg = np.memmap(filtifg, dtype=np.complex64, mode='r+', shape=(length, width))
    with h5py.File(tcoh_file, 'r') as ds:
        tcoh_ds = ds['temporalCoherence']

        box_size = 3000
        num_row = int(np.ceil(length / box_size))
        num_col = int(np.ceil(width / box_size))
        for i in range(num_row):
            for k in range(num_col):
                row_1 = i * box_size
                row_2 = i * box_size + box_size
                col_1 = k * box_size
                col_2 = k * box_size + box_size
                if row_2 > length:
                    row_2 = length
                if col_2 > width:
                    col_2 = width

                ifg_sub = ifg[row_1:row_2, col_1:col_2]
                tcoh = tcoh_ds[0, row_1:row_2, col_1:col_2]
                mask = np.zeros((tcoh.shape[0], tcoh.shape[1]))
                mask[tcoh >= 0.5] = 1

                y, x = np.where(mask == 1)
                yi, xi = np.where(mask == 0)
                zifg = np.angle(ifg_sub[y, x])
                points = np.array([[r, c] for r, c in zip(y, x)])
                tri = Delaunay(points)
                func = LinearNDInterpolator(tri, zifg, fill_value=0)
                interp_points = np.array([[r, c] for r, c in zip(yi.flatten(), xi.flatten())])
                res = np.exp(1j * func(interp_points)) * (np.abs(ifg_sub[yi, xi]).flatten())
                ifg[y + row_1, x + col_1] = ifg_sub[y, x]
                ifg[yi + row_1, xi + col_1] = res

    del ifg
    return


def distance(point1,point2):
    return sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

def gaussianLP(D0,imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2, cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x] = exp(((-distance((y,x),center)**2)/(2*(D0**2))))
    return base

def gaussianHP(D0,imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x] = 1 - exp(((-distance((y,x),center)**2)/(2*(D0**2))))
    return base

if __name__ == '__main__':
    main()


