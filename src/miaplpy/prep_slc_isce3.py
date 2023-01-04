#!/usr/bin/env python3
############################################################
# Program is part of MiaplPy                                #
# Copyright (c) 2022, Sara Mirzaee                          #
# Author: Sara Mirzaee                                      #
############################################################

import os
import time

import h5py
import numpy as np
from datetime import datetime
import argparse
import glob

try:
    from osgeo import gdal
except ImportError:
    raise ImportError('Can not import gdal [version>=3.0]!')


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


def read_subset_box(template_file, meta):
    """Read subset info from template file

    Parameters: template_file - str, path of template file
                meta          - dict, metadata
    Returns:    pix_box       - tuple of 4 int in (x0, y0, x1, y1)
                meta          - dict, metadata
    """

    if template_file and os.path.isfile(template_file):

        # read subset info from template file
        pix_box, geo_box = read_subset_template2box(template_file)

        # geo_box --> pix_box
        if geo_box is not None:
            coord = ut.coordinate(meta)
            pix_box = coord.bbox_geo2radar(geo_box)
            pix_box = coord.check_box_within_data_coverage(pix_box)
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

def extract_metadata(h5_file):
    """Extract ISCE3 metadata for MiaplPy."""
    meta = {}


    with h5py.File(h5_file, 'r') as ds:
        metadata = ds['metadata']
        meta['LENGTH'] = metadata['processing']['length'][()]
        meta['WAVELENGTH'] = float(metadata['s1ab_burst_metadata']['wavelength'][()])
        meta['WIDTH'] = metadata['processing']['width'][()]
        meta["ORBIT_DIRECTION"] = metadata['orbit']['orbit_direction'][()].decode('utf-8')
        meta['POLARIZATION'] = metadata['s1ab_burst_metadata']['polarization'][()].decode('utf-8')
        meta["STARTING_RANGE"] = float(metadata['s1ab_burst_metadata']['starting_range'][()])
        meta["RANGE_PIXEL_SIZE"] = metadata['s1ab_burst_metadata']['range_pixel_spacing'][()]
        x_step = float(metadata['processing']['x_posting'][()])
        y_step = float(metadata['processing']['y_posting'][()])
        x_first = float(metadata['processing']['start_x'][()])
        y_first = float(metadata['processing']['start_y'][()])
        datestr = metadata['orbit']['ref_epoch'][()].decode("utf-8")
        utc = datetime.strptime(datestr[0:-3], '%Y-%m-%d %H:%M:%S.%f')
        meta["CENTER_LINE_UTC"] = utc.hour * 3600.0 + utc.minute * 60.0 + utc.second + utc.microsecond * (1e-6)  # Starting line in fact
        S = min(ds['SLC']['y'][:])
        N = max(ds['SLC']['y'][:])
        E = min(ds['SLC']['x'][:])
        W = max(ds['SLC']['x'][:])

    if meta["ORBIT_DIRECTION"].startswith("D"):
        meta["HEADING"] = -168
    else:
        meta["HEADING"] = -12
    meta["ALOOKS"] = 1
    meta["RLOOKS"] = 1

    meta["X_FIRST"] = f'{x_first:.9f}'
    meta["Y_FIRST"] = f'{y_first:.9f}'
    meta["X_STEP"] = f'{x_step:.9f}'
    meta["Y_STEP"] = f'{y_step:.9f}'
    meta["X_UNIT"] = "m"
    meta["Y_UNIT"] = "m"

    meta["ANTENNA_SIDE"] = -1
    meta["PROCESSOR"] = "isce3"
    meta["PLATFORM"] = "Sen"
    meta["EARTH_RADIUS"] = 6337286.638938101
    meta["HEIGHT"] = 693000.0
    meta['INTERLEAVE'] = 'BSQ'
    meta['PLATFORM'] = 'sen'
    meta['FILE_TYPE'] = 'slc'

    meta["Y_REF1"] = str(S)
    meta["Y_REF2"] = str(S)
    meta["Y_REF3"] = str(N)
    meta["Y_REF4"] = str(N)
    meta["X_REF1"] = str(W)
    meta["X_REF2"] = str(E)
    meta["X_REF3"] = str(W)
    meta["X_REF4"] = str(E)

    return meta

####################################################################################
def extract_metadata_old(json_file):
    """Extract ISCE3 metadata for MiaplPy."""
    f = open(json_file)
    jess_dict = json.load(f)

    meta = {}
    # copy over all metadata from unwrapStack
    for key, value in jess_dict.items():
        if key not in ["Dates", "perpendicularBaseline"]:
            meta[key] = value

    meta["ANTENNA_SIDE"] = -1
    meta["PROCESSOR"] = "isce3"
    #meta["FILE_LENGTH"] = jess_dict['geogrid']['length']
    meta["LENGTH"] = jess_dict['geogrid']['length']
    meta["ORBIT_DIRECTION"] = jess_dict['orbit_direction'].upper()
    meta["PLATFORM"] = "Sen"
    meta["WAVELENGTH"] = float(jess_dict['wavelength'])
    meta["WIDTH"] = jess_dict['geogrid']['width']
    #meta["NUMBER_OF_PAIRS"] = ds.RasterCount
    meta["STARTING_RANGE"] = float(jess_dict['starting_range'])

    if meta["ORBIT_DIRECTION"].startswith("D"):
        meta["HEADING"] = -168
    else:
        meta["HEADING"] = -12

    meta["ALOOKS"] = 1
    meta["RLOOKS"] = 1
    meta["RANGE_PIXEL_SIZE"] = float(meta["range_pixel_spacing"]) * meta["RLOOKS"]

    # number of independent looks
    #sen_dict = sensor.SENSOR_DICT['sen']
    #rgfact = sen_dict['IW2']['range_resolution'] / sen_dict['range_pixel_size']
    #azfact = sen_dict['IW2']['azimuth_resolution'] / sen_dict['azimuth_pixel_size']
    #meta['NCORRLOOKS'] = meta['RLOOKS'] * meta['ALOOKS'] / (rgfact * azfact)

    # geo transformation
    #transform = ds.GetGeoTransform()

    x_step = float(meta['geogrid']['spacing_x'])
    y_step = float(meta['geogrid']['spacing_y'])
    x_first = float(meta['geogrid']['start_x'])
    y_first = float(meta['geogrid']['start_y'])


    meta["X_FIRST"] = f'{x_first:.9f}'
    meta["Y_FIRST"] = f'{y_first:.9f}'
    meta["X_STEP"] = f'{x_step:.9f}'
    meta["Y_STEP"] = f'{y_step:.9f}'
    meta["X_UNIT"] = "m"
    meta["Y_UNIT"] = "m"

    utc = meta["sensing_start"].split(' ')[1]
    utc = time.strptime(utc, "%H:%M:%S.%f")
    meta["CENTER_LINE_UTC"] = utc.tm_hour*3600.0 + utc.tm_min*60.0 + utc.tm_sec  # Starting line in fact

    # following values probably won't be used anywhere for the geocoded data
    # earth radius
    meta["EARTH_RADIUS"] = 6337286.638938101
    # nominal altitude of Sentinel1 orbit
    meta["HEIGHT"] = 693000.0

    S = meta['runconfig']['processing']['geocoding']['bottom_right']['y']
    N = meta['runconfig']['processing']['geocoding']['top_left']['y']
    E = meta['runconfig']['processing']['geocoding']['bottom_right']['x']
    W = meta['runconfig']['processing']['geocoding']['top_left']['x']

    meta["Y_REF1"] = str(S)
    meta["Y_REF2"] = str(S)
    meta["Y_REF3"] = str(N)
    meta["Y_REF4"] = str(N)
    meta["X_REF1"] = str(W)
    meta["X_REF2"] = str(E)
    meta["X_REF3"] = str(W)
    meta["X_REF4"] = str(E)

    meta['INTERLEAVE'] = 'BSQ'
    meta['PLATFORM'] = 'sen'

    return meta


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
        f['height'][:,:] = data

        # slantRangeDistance
        f['slantRangeDistance'][:,:] = float(f.attrs['STARTING_RANGE'])

        # incidenceAngle
        ds = gdal.Open(incAngleFile, gdal.GA_ReadOnly)
        bnd = ds.GetRasterBand(1)
        data = bnd.ReadAsArray(**kwargs)
        data = multilook_data(data, ystep, xstep, method='nearest')
        data[data == bnd.GetNoDataValue()] = np.nan
        f['incidenceAngle'][:,:] = data

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
        geom_files = [os.path.join(os.path.abspath(geom_dir), x + 'geo') for x in GEOMETRY_PREFIXS]

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
    metaFile = sorted(glob.glob(meta_dir + '/*_iw*.h5'))[0]

    # extract metadata
    metadata = extract_metadata(metaFile)
    inps.template_file = None
    box, metadata = read_subset_box(inps.template_file, metadata)

    # convert all value to string format
    for key, value in metadata.items():
        metadata[key] = str(value)

    # write to .rsc file
    metadata = readfile.standardize_metadata(metadata)

    # rsc_file = os.path.join(os.path.dirname(inps.metaFile), 'data.rsc')
    rsc_file = os.path.join(meta_dir, 'data.rsc')
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
