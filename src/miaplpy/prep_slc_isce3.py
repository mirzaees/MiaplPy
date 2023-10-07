#!/usr/bin/env python3
############################################################
# Program is part of MiaplPy                                #
# Copyright (c) 2022, Sara Mirzaee                          #
# Author: Sara Mirzaee                                      #
############################################################

import os
import sys
import time

import h5py
import numpy as np
from datetime import datetime
import argparse
import glob
from pyproj import CRS
from pyproj.transformer import Transformer
# from compass.utils.helpers import bbox_to_utm

try:
    from osgeo import gdal
except ImportError:
    raise ImportError('Can not import gdal [version>=3.0]!')

from mintpy.constants import EARTH_RADIUS
from mintpy.utils import readfile, writefile, ptime, attribute as attr, utils as ut
from miaplpy.objects import utils as mut
from miaplpy.objects.slcStack import (slcDatasetNames,
                                         slcStackDict,
                                         slcDict)
# from mintpy.multilook import multilook_data
# from mintpy.objects import geometry, ifgramStack, sensor
# from mintpy.subset import read_subset_template2box
# from mintpy.utils import attribute as attr, ptime, utils as ut, writefile

datasetName2templateKey = mut.datasetName2templateKey
####################################################################################

EXAMPLE = """example:
  prep_slc_isce.py -s ./stitched -g ./stitched    # for stitched level
  prep_slc_isce.py -s ./stack/t064_135527_iw2 -g ./stack/t064_135527_iw2/20220101  # for burst level
  """

GEOMETRY_PREFIXS = ['z', 'y', 'x', 'heading', 'layoverShadowMask', 'localIncidence']

def create_parser():
    """Command line parser."""
    parser = argparse.ArgumentParser(description='Prepare ISCE metadata files.',
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     epilog=EXAMPLE)
    parser.add_argument('-t', '--template', type=str, nargs='+',
                        dest='template_file', help='Template file with path info.')
    parser.add_argument('-s', '--slc-dir', dest='slcDir', type=str, default=None,
                        help='The directory which contains all SLCs\n'+
                             'e.g.: $PROJECT_DIR/merged/SLC')
    parser.add_argument('-f', '--file-pattern', nargs = '+', dest='slcFiles', type=str,
                        default=['*.h5'],
                        help='A list of files that will be used in miaplpy\n'
                             'e.g.: t064_135527_iw2.h5py')
    #parser.add_argument('-m', '--meta-file', dest='metaFile', type=str, default=None,
    #                    help='Metadata file to extract common metada for the stack:\n'
    #                         'e.g.: t064_135527_iw2_20220101_VV.json')
    parser.add_argument('-g', '--geometry-dir', dest='geometryDir', type=str, default=None,
                        help=' directory with geometry files ')
    parser.add_argument('--geom-files', dest='geometryFiles', type=str, nargs='*',
                        default=['{}.geo'.format(i) for i in GEOMETRY_PREFIXS],
                        help='List of geometry file basenames. Default: %(default)s.\n'
                             'All geometry files need to be in the same directory.')
    parser.add_argument('--force', dest='update_mode', action='store_false',
                        help='Force to overwrite all .rsc metadata files.')
    return parser


def cmd_line_parse(iargs = None):
    parser = create_parser()
    inps = parser.parse_args(args=iargs)
    if all(not i for i in [inps.slcDir, inps.geometryDir]):
        parser.print_usage()
        raise SystemExit('error: at least one of the following arguments are required: -s, -g, -m')
    return inps


def run_or_skip(inps, ds_name_dict, out_file):
    flag = 'run'

    # check 1 - update mode status
    if not inps.updateMode:
        return flag

    # check 2 - output file existence
    if ut.run_or_skip(out_file, readable=True) == 'run':
        return flag

    # check 3 - output dataset info
    key = [i for i in ['unwrapPhase', 'height'] if i in ds_name_dict.keys()][0]
    ds_shape = ds_name_dict[key][1]
    in_shape = ds_shape[-2:]

    if 'unwrapPhase' in ds_name_dict.keys():
        # compare date12 and size
        ds = gdal.Open(inps.unwFile, gdal.GA_ReadOnly)
        in_date12_list = [ds.GetRasterBand(i+1).GetMetadata("unwrappedPhase")['Dates']
                          for i in range(ds_shape[0])]
        in_date12_list = ['_'.join(d.split('_')[::-1]) for d in in_date12_list]

        try:
            out_obj = ifgramStack(out_file)
            out_obj.open(print_msg=False)
            out_shape = (out_obj.length, out_obj.width)
            out_date12_list = out_obj.get_date12_list(dropIfgram=False)

            if out_shape == in_shape and set(in_date12_list).issubset(set(out_date12_list)):
                print('All date12   exists in file {} with same size as required,'
                      ' no need to re-load.'.format(os.path.basename(out_file)))
                flag = 'skip'
        except:
            pass

    elif 'height' in ds_name_dict.keys():
        # compare dataset names and size
        in_dsNames = list(ds_name_dict.keys())
        in_size = in_shape[0] * in_shape[1] * 4 * len(in_dsNames)

        out_obj = geometry(out_file)
        out_obj.open(print_msg=False)
        out_dsNames = out_obj.datasetNames
        out_shape = (out_obj.length, out_obj.width)
        out_size = os.path.getsize(out_file)

        if (set(in_dsNames).issubset(set(out_dsNames))
                and out_shape == in_shape
                and out_size > in_size * 0.3):
            print('All datasets exists in file {} with same size as required,'
                  ' no need to re-load.'.format(os.path.basename(out_file)))
            flag = 'skip'

    return flag


#def bbox_to_utm(bounds, src_epsg, dst_epsg):
#    t = Transformer.from_crs(src_epsg, dst_epsg, always_xy=True)
#    left, bottom, right, top = bounds
#    bbox = (*t.transform(left, bottom), *t.transform(right, top))  # type: ignore
#    return bbox

def bbox_to_utm(bbox, epsg_dst, epsg_src=4326):
    """Convert a list of points to a specified UTM coordinate system.
        If epsg_src is 4326 (lat/lon), assumes points_xy are in degrees.
    """
    xmin, ymin, xmax, ymax = bbox
    t = Transformer.from_crs(epsg_src, epsg_dst, always_xy=True)
    xs = [xmin, xmax]
    ys = [ymin, ymax]
    xt, yt = t.transform(xs, ys)
    xys = list(zip(xt, yt))
    return *xys[0], *xys[1]


def read_subset_box(template_file, meta):
    """Read subset info from template file

    Parameters: template_file - str, path of template file
                meta          - dict, metadata
    Returns:    pix_box       - tuple of 4 int in (x0, y0, x1, y1)
                meta          - dict, metadata
    """

    if template_file and os.path.isfile(template_file):
        # read subset info from template file
        pix_box, geo_box = mut.read_subset_template2box(template_file)
        crs = CRS.from_wkt(meta['spatial_ref'])
        x_origin = float(meta["X_FIRST"])
        y_origin = float(meta["Y_FIRST"])
        pixel_width = float(meta["X_STEP"])
        pixel_height = float(meta["Y_STEP"])
        gt = (x_origin, pixel_width, 0, y_origin, 0, pixel_height)
        # geo_box --> pix_box
        if geo_box is not None:
            bb_utm = bbox_to_utm(geo_box, epsg_src=4326, epsg_dst=crs.to_epsg())
            xmin = int((bb_utm[0] - gt[0]) / gt[1])
            xmax = int((bb_utm[2] - gt[0]) / gt[1])
            ymin = int((bb_utm[3] - gt[3]) / gt[5])
            ymax = int((bb_utm[1] - gt[3]) / gt[5])

            if (ymin < 0 and ymax < 0) or (ymin > meta['LENGTH'] and ymax > meta['LENGTH']):
                raise ValueError('Subset latitude out of range!!!')
                sys.exit()
            if (xmin < 0 and xmax < 0) or (xmin > meta['WIDTH'] and xmax > meta['WIDTH']):
                raise ValueError('Subset longitude out of range!!!')
                sys.exit()

            if ymax > meta['LENGTH']:
                ymax = meta['LENGTH']
                bb_utm1 = ymax * gt[5] + gt[3]
                bb_utm = (bb_utm[0], bb_utm1, bb_utm[2], bb_utm[3])

            if xmax > meta['WIDTH']:
                xmax = meta['WIDTH']
                bb_utm2 = xmax * gt[1] + gt[0]
                bb_utm = (bb_utm[0], bb_utm[1], bb_utm2, bb_utm[3])

            if xmin < 0:
                xmin = 0
                bb_utm0 = gt[0]
                bb_utm = (bb_utm0, bb_utm[1], bb_utm[2], bb_utm[3])

            if ymin < 0:
                ymin = 0
                bb_utm3 = gt[3]
                bb_utm = (bb_utm[0], bb_utm[1], bb_utm[2], bb_utm3)

            pix_box = (xmin, ymin, xmax, ymax)
            print('prep_slc_isce3: ', pix_box)
            #coord = ut.coordinate(meta)
            #pix_box = coord.bbox_geo2radar(geo_box)
            #pix_box = coord.check_box_within_data_coverage(pix_box)
            print(f'input bounding box in lalo: {geo_box}')

    else:
        pix_box = None

    if pix_box is not None:
        # update metadata against the new bounding box
        print(f'input bounding box in yx: {pix_box}')
        meta = attr.update_attribute4subset(meta, pix_box)
    else:
        # translate box of None to tuple of 4 int
        length, width = int(meta['LENGTH']), int(meta['WIDTH'])
        pix_box = (0, 0, width, length)

    # ensure all index are in int16
    pix_box = tuple(int(i) for i in pix_box)

    return pix_box, meta


def get_utm_zone(wkt):
    # Open the GeoTIFF file
    # ds = gdal.Open(geotiff_path)
    # if not ds:
    #     print("Error: Couldn't open the file")
    #     return None

    # Get the projection reference
    # wkt = ds.GetProjectionRef()

    # Convert the WKT string to an OSR SpatialReference object
    sr = osr.SpatialReference()
    sr.ImportFromWkt(wkt)

    # Check if the spatial reference is UTM
    if sr.IsProjected() and "UTM" in sr.GetAttrValue("PROJCS"):
        # zone_number = sr.GetUTMZone()
        zone_number = sr.GetAttrValue("PROJCS")[-3::]
        return zone_number
    else:
        print("The provided GeoTIFF is not in a UTM projection.")
        return None

def extract_metadata(h5_file, template_file):
    """Extract ISCE3 metadata for MiaplPy."""
    meta = {}

    with h5py.File(h5_file, 'r') as ds:
        metadata = ds['metadata']
        pixel_height = float(ds['data']['y_spacing'][()])
        pixel_width = float(ds['data']['x_spacing'][()])
        x_origin = min(ds['data']['x_coordinates'][()])
        y_origin = max(ds['data']['y_coordinates'][()])
        xcoord = ds['data']['x_coordinates'][()]
        ycoord = ds['data']['y_coordinates'][()]
        dsg = ds['data']['projection'].attrs
        meta['spatial_ref'] = dsg['spatial_ref'].decode('utf-8')
        meta['WAVELENGTH'] = float(metadata['processing_information']['input_burst_metadata']['wavelength'][()])
        meta["ORBIT_DIRECTION"] = metadata['orbit']['orbit_direction'][()].decode('utf-8')
        meta['POLARIZATION'] = metadata['processing_information']['input_burst_metadata']['polarization'][()].decode('utf-8')

        meta['STARTING_RANGE'] = float(metadata['processing_information']['input_burst_metadata']['starting_range'][()])
        datestr = metadata['orbit']['reference_epoch'][()].decode("utf-8")

    if meta["ORBIT_DIRECTION"].startswith("D"):
        meta["HEADING"] = -168
    else:
        meta["HEADING"] = -12

    meta["ALOOKS"] = 1
    meta["RLOOKS"] = 1
    meta['PLATFORM'] = "Sen"
    #crs = CRS.from_wkt(meta['spatial_ref'].decode("utf-8"))
    crs = meta['spatial_ref'] #.decode("utf-8")
    utc = datetime.strptime(datestr[0:-3], '%Y-%m-%d %H:%M:%S.%f')
    meta["CENTER_LINE_UTC"] = utc.hour * 3600.0 + utc.minute * 60.0 + utc.second + utc.microsecond * (
        1e-6)  # Starting line in fact
    meta["X_FIRST"] = x_origin - pixel_width // 2
    meta["Y_FIRST"] = y_origin - pixel_height // 2
    meta["X_STEP"] = pixel_width
    meta["Y_STEP"] = pixel_height
    meta["X_UNIT"] = meta["Y_UNIT"] = "meters"
    meta["EARTH_RADIUS"] = EARTH_RADIUS
    meta["HEIGHT"] = 693000.0
    # Range and Azimuth pixel size need revision, values are just to fill in
    meta["RANGE_PIXEL_SIZE"] = abs(pixel_width)
    meta["AZIMUTH_PIXEL_SIZE"] = abs(pixel_height)
    meta["PROCESSOR"] = "isce3"
    meta["ANTENNA_SIDE"] = -1

    pix_box, geo_box = mut.read_subset_template2box(template_file)
    # get the common raster bound among input files
    if geo_box:
        # assuming bbox is in lat/lon coordinates
        epsg_src = 4326
        utm_bbox = bbox_to_utm(geo_box, CRS.from_wkt(crs).to_epsg(), epsg_src)
    else:
        utm_bbox = None
    bounds = get_raster_bounds(xcoord, ycoord, utm_bbox)
    meta['bbox'] = ",".join([str(b) for b in bounds])

    col1, row1, col2, row2 = get_rows_cols(xcoord, ycoord, bounds)
    length = row2 - row1
    width = col2 - col1
    meta['LENGTH'] = length
    meta['WIDTH'] = width

    meta['STARTING_RANGE'] += col1 * abs(pixel_width)

    return meta

def update_meta(ref_file, meta):
    ds = gdal.Open(ref_file, gdal.GA_ReadOnly)

    meta['LENGTH'] = ds.RasterYSize
    meta['WIDTH'] = ds.RasterXSize
    geotransform = ds.GetGeoTransform()
    pixel_width = geotransform[1]
    pixel_height = geotransform[5]
    meta["X_FIRST"] = geotransform[0] - pixel_width // 2
    meta["Y_FIRST"] = geotransform[3] - pixel_height // 2
    meta["X_STEP"] = pixel_width
    meta["Y_STEP"] = pixel_height
    meta["RANGE_PIXEL_SIZE"] = abs(pixel_width)
    meta["AZIMUTH_PIXEL_SIZE"] = abs(pixel_height)

    return

def get_rows_cols(xcoord, ycoord, bounds):
    """Get row and cols of the bounding box to subset"""
    xindex = np.where(np.logical_and(xcoord >= bounds[0], xcoord <= bounds[2]))[0]
    yindex = np.where(np.logical_and(ycoord >= bounds[1], ycoord <= bounds[3]))[0]
    row1, row2 = min(yindex), max(yindex)
    col1, col2 = min(xindex), max(xindex)
    return col1, row1, col2, row2


def get_raster_bounds(xcoord, ycoord, utm_bbox=None):
    """Get common bounds among all data"""
    x_bounds = []
    y_bounds = []

    west = min(xcoord)
    east = max(xcoord)
    north = max(ycoord)
    south = min(ycoord)

    x_bounds.append([west, east])
    y_bounds.append([south, north])
    if not utm_bbox is None:
        x_bounds.append([utm_bbox[0], utm_bbox[2]])
        y_bounds.append([utm_bbox[1], utm_bbox[3]])

    bounds = max(x_bounds)[0], max(y_bounds)[0], min(x_bounds)[1], min(y_bounds)[1]
    return bounds

####################################################################################


def write_geometry(outfile, demFile, incAngleFile, azAngleFile=None, waterMaskFile=None,
                   box=None, xstep=1, ystep=1):
    """Write geometry HDF5 file from list of VRT files."""

    print('-'*50)
    # box to gdal arguments
    # link: https://gdal.org/python/osgeo.gdal.Band-class.html#ReadAsArray
    if box is not None:
        kwargs = dict(
            xoff=box[0],
            yoff=box[1],
            win_xsize=box[2]-box[0],
            win_ysize=box[3]-box[1])
    else:
        kwargs = dict()

    print(f'writing data to HDF5 file {outfile} with a mode ...')
    with h5py.File(outfile, 'a') as f:

        # height
        ds = gdal.Open(demFile, gdal.GA_ReadOnly)
        bnd = ds.GetRasterBand(1)
        data = np.array(bnd.ReadAsArray(**kwargs), dtype=np.float32)
        data = multilook_data(data, ystep, xstep, method='nearest')
        data[data == bnd.GetNoDataValue()] = np.nan
        f['height'][:, :] = data

        # slantRangeDistance
        f['slantRangeDistance'][:, :] = float(f.attrs['STARTING_RANGE'])

        # incidenceAngle
        ds = gdal.Open(incAngleFile, gdal.GA_ReadOnly)
        bnd = ds.GetRasterBand(1)
        data = bnd.ReadAsArray(**kwargs)
        data = multilook_data(data, ystep, xstep, method='nearest')
        data[data == bnd.GetNoDataValue()] = np.nan
        f['incidenceAngle'][:, :] = data

        # azimuthAngle
        if azAngleFile is not None:
            ds = gdal.Open(azAngleFile, gdal.GA_ReadOnly)
            bnd = ds.GetRasterBand(1)
            data = bnd.ReadAsArray(**kwargs)
            data = multilook_data(data, ystep, xstep, method='nearest')
            data[data == bnd.GetNoDataValue()] = np.nan
            # azimuth angle of the line-of-sight vector:
            # ARIA: vector from target to sensor measured from the east  in counterclockwise direction
            # ISCE: vector from sensor to target measured from the north in counterclockwise direction
            # convert ARIA format to ISCE format, which is used in mintpy
            data -= 90
            f['azimuthAngle'][:,:] = data

        # waterMask
        if waterMaskFile is not None:
            # read
            ds = gdal.Open(waterMaskFile, gdal.GA_ReadOnly)
            bnd = ds.GetRasterBand(1)
            water_mask = bnd.ReadAsArray(**kwargs)
            water_mask = multilook_data(water_mask, ystep, xstep, method='nearest')
            water_mask[water_mask == bnd.GetNoDataValue()] = False

            # assign False to invalid pixels based on incAngle data
            ds = gdal.Open(incAngleFile, gdal.GA_ReadOnly)
            bnd = ds.GetRasterBand(1)
            data = bnd.ReadAsArray(**kwargs)
            data = multilook_data(data, ystep, xstep, method='nearest')
            water_mask[data == bnd.GetNoDataValue()] = False

            # write
            f['waterMask'][:,:] = water_mask

    print(f'finished writing to HD5 file: {outfile}')
    return outfile


def write_ifgram_stack(outfile, unwStack, cohStack, connCompStack, ampStack=None,
                       box=None, xstep=1, ystep=1, mli_method='nearest'):
    """Write ifgramStack HDF5 file from stack VRT files
    """

    print('-'*50)
    stackFiles = [unwStack, cohStack, connCompStack, ampStack]
    max_digit = max(len(os.path.basename(str(i))) for i in stackFiles)
    for stackFile in stackFiles:
        if stackFile is not None:
            print('open {f:<{w}} with gdal ...'.format(f=os.path.basename(stackFile), w=max_digit))

    dsUnw = gdal.Open(unwStack, gdal.GA_ReadOnly)
    dsCoh = gdal.Open(cohStack, gdal.GA_ReadOnly)
    dsComp = gdal.Open(connCompStack, gdal.GA_ReadOnly)
    if ampStack is not None:
        dsAmp = gdal.Open(ampStack, gdal.GA_ReadOnly)
    else:
        dsAmp = None

    # extract NoDataValue (from the last */date2_date1.vrt file for example)
    ds = gdal.Open(dsUnw.GetFileList()[-1], gdal.GA_ReadOnly)
    noDataValueUnw = ds.GetRasterBand(1).GetNoDataValue()
    print(f'grab NoDataValue for unwrapPhase     : {noDataValueUnw:<5} and convert to 0.')

    ds = gdal.Open(dsCoh.GetFileList()[-1], gdal.GA_ReadOnly)
    noDataValueCoh = ds.GetRasterBand(1).GetNoDataValue()
    print(f'grab NoDataValue for coherence       : {noDataValueCoh:<5} and convert to 0.')

    ds = gdal.Open(dsComp.GetFileList()[-1], gdal.GA_ReadOnly)
    noDataValueComp = ds.GetRasterBand(1).GetNoDataValue()
    print(f'grab NoDataValue for connectComponent: {noDataValueComp:<5} and convert to 0.')
    ds = None

    if dsAmp is not None:
        ds = gdal.Open(dsAmp.GetFileList()[-1], gdal.GA_ReadOnly)
        noDataValueAmp = ds.GetRasterBand(1).GetNoDataValue()
        print(f'grab NoDataValue for magnitude       : {noDataValueAmp:<5} and convert to 0.')
        ds = None

    # sort the order of interferograms based on date1_date2 with date1 < date2
    nPairs = dsUnw.RasterCount
    d12BandDict = {}
    for ii in range(nPairs):
        bnd = dsUnw.GetRasterBand(ii+1)
        d12 = bnd.GetMetadata("unwrappedPhase")["Dates"]
        d12 = sorted(d12.split("_"))
        d12 = f'{d12[0]}_{d12[1]}'
        d12BandDict[d12] = ii+1
    d12List = sorted(d12BandDict.keys())
    print(f'number of interferograms: {len(d12List)}')

    # box to gdal arguments
    # link: https://gdal.org/python/osgeo.gdal.Band-class.html#ReadAsArray
    if box is not None:
        kwargs = dict(
            xoff=box[0],
            yoff=box[1],
            win_xsize=box[2]-box[0],
            win_ysize=box[3]-box[1])
    else:
        kwargs = dict()

    if xstep * ystep > 1:
        msg = f'apply {xstep} x {ystep} multilooking/downsampling via {mli_method} to: unwrapPhase, coherence'
        msg += ', magnitude' if dsAmp is not None else ''
        msg += f'\napply {xstep} x {ystep} multilooking/downsampling via nearest to: connectComponent'
        print(msg)
    print(f'writing data to HDF5 file {outfile} with a mode ...')
    with h5py.File(outfile, "a") as f:

        prog_bar = ptime.progressBar(maxValue=nPairs)
        for ii in range(nPairs):
            d12 = d12List[ii]
            bndIdx = d12BandDict[d12]
            prog_bar.update(ii+1, suffix=f'{d12} {ii+1}/{nPairs}')

            f["date"][ii,0] = d12.split("_")[0].encode("utf-8")
            f["date"][ii,1] = d12.split("_")[1].encode("utf-8")
            f["dropIfgram"][ii] = True

            bnd = dsUnw.GetRasterBand(bndIdx)
            data = bnd.ReadAsArray(**kwargs)
            data = multilook_data(data, ystep, xstep, method=mli_method)
            data[data == noDataValueUnw] = 0      #assign pixel with no-data to 0
            data *= -1.0                          #date2_date1 -> date1_date2
            f["unwrapPhase"][ii,:,:] = data

            bperp = float(bnd.GetMetadata("unwrappedPhase")["perpendicularBaseline"])
            bperp *= -1.0                         #date2_date1 -> date1_date2
            f["bperp"][ii] = bperp

            bnd = dsCoh.GetRasterBand(bndIdx)
            data = bnd.ReadAsArray(**kwargs)
            data = multilook_data(data, ystep, xstep, method=mli_method)
            data[data == noDataValueCoh] = 0      #assign pixel with no-data to 0
            f["coherence"][ii,:,:] = data

            bnd = dsComp.GetRasterBand(bndIdx)
            data = bnd.ReadAsArray(**kwargs)
            data = multilook_data(data, ystep, xstep, method='nearest')
            data[data == noDataValueComp] = 0     #assign pixel with no-data to 0
            f["connectComponent"][ii,:,:] = data

            if dsAmp is not None:
                bnd = dsAmp.GetRasterBand(bndIdx)
                data = bnd.ReadAsArray(**kwargs)
                data = multilook_data(data, ystep, xstep, method=mli_method)
                data[data == noDataValueAmp] = 0  #assign pixel with no-data to 0
                f["magnitude"][ii,:,:] = data

        prog_bar.close()

        # add MODIFICATION_TIME metadata to each 3D dataset
        for dsName in ['unwrapPhase','coherence','connectComponent']:
            f[dsName].attrs['MODIFICATION_TIME'] = str(time.time())

    print(f'finished writing to HD5 file: {outfile}')
    dsUnw = None
    dsCoh = None
    dsComp = None
    dsAmp = None
    return outfile


def add_slc_metadata(metadata_in, dates=[]):
    """Add metadata unique for each slc
    Parameters: metadata_in   : dict, input common metadata for the entire dataset
                dates         : list of str in YYYYMMDD or YYMMDD format
    Returns:    metadata      : dict, updated metadata
    """
    # make a copy of input metadata
    metadata = {}
    for k in metadata_in.keys():
        metadata[k] = metadata_in[k]
    metadata['FILE_TYPE'] = '.slc'
    metadata['DATE'] = '{}'.format(dates[1])
    random_baseline = str(np.random.randint(1, 400))
    metadata['P_BASELINE_TOP_HDR'] = random_baseline
    metadata['P_BASELINE_BOTTOM_HDR'] = random_baseline

    return metadata


def prepare_stack(inputDir, filePattern, metadata={}, update_mode=True):

    print('preparing RSC file for ', filePattern)
    isce_files = sorted(glob.glob(os.path.join(os.path.abspath(inputDir), '*', filePattern)))

    if len(isce_files) == 0:
        raise FileNotFoundError('no file found in pattern: {}'.format(filePattern))

    # write .rsc file for each slc file
    num_file = len(isce_files)
    slc_dates = np.sort(os.listdir(inputDir))
    prog_bar = ptime.progressBar(maxValue=num_file)
    for i in range(num_file):
        # prepare metadata for current file
        isce_file = isce_files[i]
        dates = [slc_dates[0], os.path.basename(os.path.dirname(isce_file))]
        #slc_metadata = mut.read_attribute(isce_file.split('.slc')[0], metafile_ext='.json')
        slc_metadata = {}
        slc_metadata.update(metadata)
        slc_metadata = add_slc_metadata(slc_metadata, dates)

        # write .rsc file
        rsc_file = isce_file.split('.h5')[0] + '.rsc'
        writefile.write_roipac_rsc(slc_metadata, rsc_file,
                                   update_mode=update_mode,
                                   print_msg=False)
        prog_bar.update(i + 1, suffix='{}_{}'.format(dates[0], dates[1]))
    prog_bar.close()


    return

def prepare_geometry(geom_dir, metadata=dict(), update_mode=True):
    """Prepare and extract metadata from geometry files"""

    print('prepare .rsc file for geometry files')
    # grab all existed files

    # default file basenames
    geom_files = ['{}.geo'.format(i) for i in GEOMETRY_PREFIXS]

    # get absolute file paths
    geom_files = [os.path.join(geom_dir, i) for i in geom_files]

    if not os.path.exists(geom_files[0]):
        geom_files = [os.path.join(os.path.abspath(geom_dir), x + '.geo') for x in GEOMETRY_PREFIXS]

    geom_files = [i for i in geom_files if os.path.isfile(i)]

    # write rsc file for each file
    for geom_file in geom_files:
        # prepare metadata for current file
        # geom_metadata = mut.read_attribute(geom_file.split('.geo')[0], metafile_ext='.hdr')
        geom_metadata = {}
        geom_metadata.update(metadata)
        geom_metadata['OG_FILE_PATH'] = geom_files[0]
        # write .rsc file
        rsc_file = geom_file + '.rsc'
        writefile.write_roipac_rsc(geom_metadata, rsc_file,
                                   update_mode=update_mode,
                                   print_msg=True)
    return metadata

####################################################################################
def load_isce3(iargs=None):
    """Prepare and load ISCE3 data and metadata into HDF5/MiaplPy format."""

    #start_time = time.time()

    inps = cmd_line_parse(iargs)
    inps.processor = 'isce3'
    print(f'update mode: {inps.update_mode}')

    meta_dir = inps.geometryDir
    metaFile = sorted(glob.glob(meta_dir + '/*/static*_iw*.h5'))[0]

    # extract metadata
    metadata = extract_metadata(metaFile, inps.template_file[0])
   
    # convert all value to string format
    for key, value in metadata.items():
        metadata[key] = str(value)

    # write to .rsc file
    metadata = readfile.standardize_metadata(metadata)

    # rsc_file = os.path.join(os.path.dirname(inps.metaFile), 'data.rsc')
    rsc_file = metaFile.split('.')[-2] + '.rsc' #os.path.join(meta_dir, 'data.rsc')
    if rsc_file:
        print('writing ', rsc_file)
        writefile.write_roipac_rsc(metadata, rsc_file)

        # prepare metadata for geometry file
    # if inps.geometryDir:
    #    metadata = prepare_geometry(inps.geometryDir,
    #                                metadata=metadata,
    #                                update_mode=inps.update_mode)

    # prepare metadata for slc file
    if inps.slcDir and inps.slcFiles:
        for namePattern in inps.slcFiles:
            prepare_stack(inps.slcDir, namePattern,
                          metadata=metadata,
                          update_mode=inps.update_mode)

    return


#########################################################################
if __name__ == '__main__':
    """Main driver."""
    load_isce3()
