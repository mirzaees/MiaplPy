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
import rioxarray
import dask.array as da

enablePrint()


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
    os.makedirs(inps.out_dir, exist_ok=True)

    ifg_file_name = run_interferogram(inps)
    run_coherence(ifg_file_name)

    '''
    resampName = inps.out_dir + '/fine'
    resampInt = resampName + '.int'
    filtInt = os.path.dirname(resampInt) + '/filt_fine.int'
    cor_file = os.path.dirname(resampInt) + '/filt_fine.cor'

    if os.path.exists(cor_file + '.xml'):
        return

    length, width = run_interferogram(inps, resampName)

    filter_strength = inps.filter_strength
    runFilter(resampInt, filtInt, filter_strength)

    estCoherence(filtInt, cor_file)
    #run_interpolation(filtInt, inps.stack_file, length, width)
    '''
    return

def write_projection(src_file: Filename, dst_file: Filename) -> None:
    with h5py.File(src_file, 'r') as ds:
        projection = ds.attrs["spatial_ref"].decode('utf-8')
        geotransform = d['georeference']['transform'][()]
        #extent = ds['georeference'].attrs['extent']
        nodata = np.nan

    ds_dst = gdal.Open(os.fspath(dst_file), gdal.GA_Update)
    ds_dst.SetGeoTransform(geotransform)
    ds_dst.SetProjection(projection)
    ds_dst.GetRasterBand(1).SetNoDataValue(nodata)
    ds_src = ds_dst = None
    return

def run_interferogram(inps):
    # sequential = True
    sequential = False
    out_file = inps.out_dir + '.tif'
    dask_chunks = (128 * 10, 128 * 10)

    # sequential interferograms
    with h5py.File(inps.stack_file, 'r') as ds:
        if 'spatial_ref' in ds.attrs:
            projection = ds.attrs["spatial_ref"].decode('utf-8')
        else:
            prpjection = 'None'
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

                    if ref_ind - sec_ind == 1:
                        ifg = ref_amplitude * np.exp(1j * ref_phase)
                    if ref_ind - sec_ind == -1:
                        ifg = sec_amplitude * np.exp(-1j * sec_phase)
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

                    ifg[np.isnan(ifg)] = 0 + 1j*0
                    out_ifg[row_1:row_2, col_1:col_2, 0] = ifg[:, :]

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

                    ifg = (ref_amplitude * sec_amplitude) * np.exp(1j * (ref_phase - sec_phase))

                    ifg[np.isnan(ifg)] = 0 + 1j * 0
                    out_ifg[row_1:row_2, col_1:col_2, 0] = ifg[:, :]

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


