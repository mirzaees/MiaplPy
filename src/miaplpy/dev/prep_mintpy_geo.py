#!/usr/bin/env python
############################################################
# Program is part of MintPy                                #
# Copyright (c) 2013, Zhang Yunjun, Heresh Fattahi         #
# Author: Talib Oliver Cabrerra, Scott Staniewicz          #
############################################################


import argparse
import datetime
import glob
import itertools
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pyproj
from osgeo import gdal, osr
from dolphin import io, stitching
from dolphin.utils import full_suffix, get_dates
from mintpy.utils import arg_utils, ptime, readfile, writefile
from mintpy.utils.utils0 import calc_azimuth_from_east_north_obs
#from dolphin.workflows import group_by_burst
####################################################################################
EXAMPLE = """example:

  prep_mintpy_geo.py -u './unwrap/*.unw'  -c './correlations/*.tif' -g './geometry'  -m gslcs/

"""  # noqa: E501

# """
# Scott TODO:
# - UTM_ZONE, EPSG from the stitched IFG (it won't work to get a single GSLC burst)
# - pixel size is wrong since we're taking range/azimuth size, instead of geocoded size
# - HEIGHT: do we wanna try to get that from the saved orbit info?


def _create_parser():
    parser = argparse.ArgumentParser(
        description="Prepare Sweets products for MintPy",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=EXAMPLE,
    )

    parser.add_argument(
        "-u",
        "--unw-file-glob",
        type=str,
        default="./interferograms/unwrapped/*.unw.tif",
        help="path pattern of unwrapped interferograms (default: %(default)s).",
    )
    parser.add_argument(
        "-c",
        "--cor-file-glob",
        type=str,
        default="./interferograms/stitched/*.cor",
        help="path pattern of unwrapped interferograms (default: %(default)s).",
    )
    parser.add_argument(
        "-g",
        "--geom-dir",
        default="./geometry",
        help="Geometry directory (default: %(default)s).\n",
    )
    parser.add_argument(
        "-m",
        "--meta-file",
        type=str,
        help="GSLC metadata file or directory",
    )
    parser.add_argument(
        "-b",
        "--baseline-dir",
        dest="baselineDir",
        type=str,
        default=None,
        help="baseline directory (default: %(default)s).",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        type=str,
        default="./mintpy",
        help="output directory (default: %(default)s).",
    )
    parser.add_argument(
        "-r",
        "--range",
        dest="lks_x",
        type=int,
        default=1,
        help=(
            "number of looks in range direction, (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "-a",
        "--azimuth",
        dest="lks_y",
        type=int,
        default=1,
        help=(
            "number of looks in azimuth direction, (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--single-reference",
        action="store_true",
        help=(
            "Indicate that all the unwrapped ifgs are single reference, which allows"
            " use to create the timeseries.h5 file directly without inversion."
        ),
    )
    
    parser.add_argument(
        "-p",
        "--package",
        dest="package",
        type=str,
        default='dolphin',
        help=("PAckage used for phase linking, options [dolphin, miaplpy]"
        ),
    )


    parser = arg_utils.add_subset_argument(parser, geo=True)

    return parser


def cmd_line_parse(iargs=None):
    """Create the command line parser."""
    parser = _create_parser()
    inps = parser.parse_args(args=iargs)

    # in case meta_file is input as wildcard
    inps.meta_file = sorted(glob.glob(inps.meta_file))[0]

    return inps


def prepare_metadata(meta_file, int_file, nlks_x=1, nlks_y=1):
    """Get the metadata from the GSLC metadata file and the unwrapped interferogram."""
    print("-" * 50)

    cols, rows = io.get_raster_xysize(int_file)
    meta_compass = h5py.File(meta_file, "r")
    meta = {}

    geotransform = io.get_raster_gt(int_file)
    meta["LENGTH"] = rows
    meta["WIDTH"] = cols

    meta["X_FIRST"] = geotransform[0]
    meta["Y_FIRST"] = geotransform[3]
    meta["X_STEP"] = geotransform[1]
    meta["Y_STEP"] = geotransform[5]
    meta["X_UNIT"] = meta["Y_UNIT"] = "meters"

    ds_src = gdal.Open(int_file)
    wkt = ds_src.GetProjectionRef()
    sr = osr.SpatialReference()
    sr.ImportFromWkt(wkt)
    if sr.IsProjected() and "UTM" in sr.GetAttrValue("PROJCS"):
        meta['UTM_ZONE'] = sr.GetAttrValue("PROJCS")[-3::]
    else:
        meta['UTM_ZONE'] = None

    crs = io.get_raster_crs(int_file)
    meta["EPSG"] = crs.to_epsg()

    if "/science" in meta_compass:
        root = "/science/SENTINEL1/CSLC"
        processing_ds = f"{root}/metadata/processing_information"
        burst_ds = f"{processing_ds}/s1_burst_metadata"
        if burst_ds not in meta_compass:
            burst_ds = f"{processing_ds}/input_burst_metadata"
    else:
        root = ''
        processing_ds = f"{root}/metadata/processing_information"
        burst_ds = f"{processing_ds}/input_burst_metadata"

    xcoord = meta_compass[f"{root}/data/x_coordinates"][()]
    xindex = np.where(xcoord >= geotransform[0])[0]
    col1 = min(xindex)

    meta["WAVELENGTH"] = meta_compass[f"{burst_ds}/wavelength"][()]
    meta["RANGE_PIXEL_SIZE"] = meta_compass[f"{burst_ds}/range_pixel_spacing"][()]
    meta["AZIMUTH_PIXEL_SIZE"] = 14.1
    meta["EARTH_RADIUS"] = 6371000.0

    t0 = datetime.datetime.strptime(
        meta_compass[f"{burst_ds}/sensing_start"][()].decode("utf-8"),
        "%Y-%m-%d %H:%M:%S.%f",
    )
    t1 = datetime.datetime.strptime(
        meta_compass[f"{burst_ds}/sensing_stop"][()].decode("utf-8"),
        "%Y-%m-%d %H:%M:%S.%f",
    )
    t_mid = t0 + (t1 - t0) / 2.0
    meta["CENTER_LINE_UTC"] = (
        t_mid - datetime.datetime(t_mid.year, t_mid.month, t_mid.day)
    ).total_seconds()
    meta["HEIGHT"] = 750000.0
    meta["STARTING_RANGE"] = float(meta_compass[f"{burst_ds}/starting_range"][()]) + col1 * abs(meta["RANGE_PIXEL_SIZE"])
    meta["PLATFORM"] = meta_compass[f"{burst_ds}/platform_id"][()].decode("utf-8")
    meta["ORBIT_DIRECTION"] = meta_compass[f"{root}/metadata/orbit/orbit_direction"][
        ()
    ].decode("utf-8")
    meta["ALOOKS"] = 1
    meta["RLOOKS"] = 1

    # apply optional user multilooking
    if nlks_x > 1:
        meta["RANGE_PIXEL_SIZE"] = str(float(meta["RANGE_PIXEL_SIZE"]) * nlks_x)
        meta["RLOOKS"] = str(float(meta["RLOOKS"]) * nlks_x)

    if nlks_y > 1:
        meta["AZIMUTH_PIXEL_SIZE"] = str(float(meta["AZIMUTH_PIXEL_SIZE"]) * nlks_y)
        meta["ALOOKS"] = str(float(meta["ALOOKS"]) * nlks_y)

    return meta


def _get_xy_arrays(atr):
    x0 = float(atr["X_FIRST"])
    y0 = float(atr["Y_FIRST"])
    x_step = float(atr["X_STEP"])
    y_step = float(atr["Y_STEP"])
    rows = int(atr["LENGTH"])
    cols = int(atr["WIDTH"])
    x_arr = x0 + x_step * np.arange(cols)
    y_arr = y0 + y_step * np.arange(rows)
    # Shift by half pixel to get the centers
    x_arr += x_step / 2
    y_arr += y_step / 2
    return x_arr, y_arr


def write_coordinate_system(
    filename, dset_name, xy_dim_names=("x", "y"), grid_mapping_dset="spatial_ref"
):
    """Write the coordinate system CF metadata to an existing HDF5 file."""
    x_dim_name, y_dim_name = xy_dim_names
    atr = readfile.read_attribute(filename)
    epsg = int(atr.get("EPSG", 4326))

    with h5py.File(filename, "a") as hf:
        crs = pyproj.CRS.from_user_input(epsg)
        dset = hf[dset_name]

        # Setup the dataset holding the SRS information
        srs_dset = hf.require_dataset(grid_mapping_dset, shape=(), dtype=int)
        srs_dset.attrs.update(crs.to_cf())
        dset.attrs["grid_mapping"] = grid_mapping_dset

        if "date" in hf:
            date_arr = [
                datetime.datetime.strptime(ds, "%Y%m%d")
                for ds in hf["date"][()].astype(str)
            ]
            days_since = [(d - date_arr[0]).days for d in date_arr]
            dt_dim = hf.create_dataset("time", data=days_since)
            dt_dim.make_scale()
            cf_attrs = dict(
                units=f"days since {str(date_arr[0])}", calendar="proleptic_gregorian"
            )
            dt_dim.attrs.update(cf_attrs)
            dset.dims[0].attach_scale(dt_dim)
            dset.dims[0].label = "time"
        else:
            dt_dim = date_arr = None
            # If we want to do something other than time as a 3rd dimension...
            #  We'll need to figure out what other valid dims there are
            # otherwise, we can just do `phony_dims="sort"` in xarray

        # add metadata to x,y coordinates
        is_projected = crs.is_projected
        is_geographic = crs.is_geographic
        x_arr, y_arr = _get_xy_arrays(atr)
        x_dim_dset = hf.create_dataset(x_dim_name, data=x_arr)
        x_dim_dset.make_scale(x_dim_name)
        y_dim_dset = hf.create_dataset(y_dim_name, data=y_arr)
        y_dim_dset.make_scale(y_dim_name)

        x_coord_attrs = {}
        x_coord_attrs["axis"] = "X"
        y_coord_attrs = {}
        y_coord_attrs["axis"] = "Y"
        if is_projected:
            units = "meter"
            # X metadata
            x_coord_attrs["long_name"] = "x coordinate of projection"
            x_coord_attrs["standard_name"] = "projection_x_coordinate"
            x_coord_attrs["units"] = units
            # Y metadata
            y_coord_attrs["long_name"] = "y coordinate of projection"
            y_coord_attrs["standard_name"] = "projection_y_coordinate"
            y_coord_attrs["units"] = units
        elif is_geographic:
            # X metadata
            x_coord_attrs["long_name"] = "longitude"
            x_coord_attrs["standard_name"] = "longitude"
            x_coord_attrs["units"] = "degrees_east"
            # Y metadata
            y_coord_attrs["long_name"] = "latitude"
            y_coord_attrs["standard_name"] = "latitude"
            y_coord_attrs["units"] = "degrees_north"
        y_dim_dset.attrs.update(y_coord_attrs)
        x_dim_dset.attrs.update(x_coord_attrs)

        ndim = dset.ndim
        dset.dims[ndim - 1].attach_scale(x_dim_dset)
        dset.dims[ndim - 2].attach_scale(y_dim_dset)
        dset.dims[ndim - 1].label = x_dim_name
        dset.dims[ndim - 2].label = y_dim_name


def _get_date_pairs(filenames):
    str_list = [Path(f).stem for f in filenames]
    return [str(f).replace(full_suffix(f), "") for f in str_list]


def prepare_timeseries(
    outfile,
    unw_files,
    metadata,
    baseline_dir=None,
):
    """Prepare the timeseries file."""
    print("-" * 50)
    print("preparing timeseries file: {}".format(outfile))

    # copy metadata to meta
    meta = {key: value for key, value in metadata.items()}
    phase2range = float(meta["WAVELENGTH"]) / (4.0 * np.pi)

    # grab date list from the filename
    date12_list = _get_date_pairs(unw_files)
    num_file = len(unw_files)
    print("number of unwrapped interferograms: {}".format(num_file))

    date_pairs = [dl.split("_") for dl in date12_list]
    date_list = sorted(set(itertools.chain.from_iterable(date_pairs)))
    # ref_date = date12_list[0].split("_")[0]
    # date_list = [ref_date] + [date12.split("_")[1] for date12 in date12_list]
    num_date = len(date_list)
    print("number of acquisitions: {}\n{}".format(num_date, date_list))

    # baseline info
    pbase = np.zeros(num_date, dtype=np.float32)
    if baseline_dir is not None:
        raise NotImplementedError

    # size info
    cols, rows = io.get_raster_xysize(unw_files[0])

    # define dataset structure
    dates = np.array(date_list, dtype=np.string_)
    ds_name_dict = {
        "date": [dates.dtype, (num_date,), dates],
        "bperp": [np.float32, (num_date,), pbase],
        "timeseries": [np.float32, (num_date, rows, cols), None],
    }

    # initiate HDF5 file
    meta["FILE_TYPE"] = "timeseries"
    meta["UNIT"] = "m"
    # meta["REF_DATE"] = ref_date # might not be the first date!
    writefile.layout_hdf5(outfile, ds_name_dict, metadata=meta)

    # writing data to HDF5 file
    print("writing data to HDF5 file {} with a mode ...".format(outfile))
    with h5py.File(outfile, "a") as f:
        prog_bar = ptime.progressBar(maxValue=num_file)
        for i, unw_file in enumerate(unw_files):
            # read data using gdal
            data = io.load_gdal(unw_file)

            f["timeseries"][i + 1] = data * phase2range
            prog_bar.update(i + 1, suffix=date12_list[i])
        prog_bar.close()

        print("set value at the first acquisition to ZERO.")
        f["timeseries"][0] = 0.0

    print("finished writing to HDF5 file: {}".format(outfile))
    return outfile


def prepare_geometry(outfile, geom_dir, metadata, water_mask_file=None):
    """Prepare the geometry file."""
    print("-" * 50)
    print(f"preparing geometry file: {outfile}")

    geom_path = Path(geom_dir)
    # copy metadata to meta
    meta = {key: value for key, value in metadata.items()}
    meta["FILE_TYPE"] = "geometry"

    file_to_path = {
        "los_east": geom_path / "los_east_rsm.tif",
        "los_north": geom_path / "los_north_rsm.tif",
        "height": geom_path / "height_rsm.tif",
        "shadowMask": geom_path / "layover_shadow_mask_rsm.tif",
    }

    if water_mask_file:
        file_to_path["waterMask"] = water_mask_file

    dsDict = {}
    for dsName, fname in file_to_path.items():
        try:
            data = readfile.read(str(fname), datasetName=dsName)[0].astype(float)
            # TODO: add general functionality to handle nodata into Mintpy
            data[data == 0] = np.nan
            dsDict[dsName] = data

            # write data to HDF5 file
        except KeyError as e:  # https://github.com/insarlab/MintPy/issues/1081
            print(f"Skipping {fname}: {e}")

    # Compute the azimuth and incidence angles from east/north coefficients
    east = dsDict["los_east"]
    north = dsDict["los_north"]
    azimuth_angle = calc_azimuth_from_east_north_obs(east, north)
    dsDict["azimuthAngle"] = azimuth_angle

    up = np.sqrt(1 - east**2 - north**2)
    incidence_angle = np.rad2deg(np.arccos(up))
    dsDict["incidenceAngle"] = incidence_angle

    dsDict['slantRangeDistance'] = np.ones(incidence_angle.shape, dtype=np.float32) * range_distance(metadata)

    writefile.write(dsDict, outfile, metadata=meta)
    return outfile


def range_distance(atr):

    print('input file is geocoded, return center range distance for the whole area')

    range_n, dR = float(atr['STARTING_RANGE']), float(atr['RANGE_PIXEL_SIZE'])
    length, width = int(atr['LENGTH']), int(atr['WIDTH'])

    range_f = range_n + dR * (width - 1)
    range_c = (range_f + range_n) / 2.0

    #print('near   range : %.2f m' % (range_n))
    #print('center range : %.2f m' % (range_c))
    #print('far    range : %.2f m' % (range_f))

    return np.array(range_c, np.float32)


def prepare_temporal_coherence(outfile, infile, metadata):
    """Prepare the temporal coherence file."""
    print("-" * 50)
    print("preparing temporal coherence file: {}".format(outfile))

    # copy metadata to meta
    meta = {key: value for key, value in metadata.items()}
    meta["FILE_TYPE"] = "temporalCoherence"
    meta["UNIT"] = "1"

    data = io.load_gdal(infile)

    print(data.shape)
    # write to HDF5 file
    writefile.write(data, outfile, metadata=meta)
    return outfile


def prepare_ps_mask(outfile, infile, metadata):
    """Prepare the PS mask file."""
    print("-" * 50)
    print("preparing PS mask file: {}".format(outfile))

    # copy metadata to meta
    meta = {key: value for key, value in metadata.items()}
    meta["FILE_TYPE"] = "mask"
    meta["UNIT"] = "1"

    # read data using gdal
    data = io.load_gdal(infile)

    # write to HDF5 file
    writefile.write(data, outfile, metadata=meta)
    return outfile


def prepare_stack(
    outfile,
    unw_files,
    cor_files,
    metadata,
    # baseline_dir=None,
):
    """Prepare the input unw stack."""
    print("-" * 50)
    print("preparing ifgramStack file: {}".format(outfile))
    # copy metadata to meta
    meta = {key: value for key, value in metadata.items()}

    # get list of *.unw file
    num_pair = len(unw_files)
    unw_ext = full_suffix(unw_files[0])

    print(f"number of unwrapped interferograms: {num_pair}")
    print(f"number of correlation files: {len(cor_files)}")

    # get list of *.unw.conncomp file
    if metadata['package'] == 'dolphin':
        cc_files = [x.split('.tif')[0] + '.conncomp' for x in unw_files]
        cc_files = [x for x in cc_files if Path(x).exists()]
    else:
        cc_files = [x + '.conncomp' for x in unw_files]
        cc_files = [x for x in cc_files if Path(x).exists()]
    print(f"number of connected components files: {len(cc_files)}")

    if len(cc_files) != len(unw_files) or len(cor_files) != len(unw_files):
        print(
            "the number of *.unw and *.unw.conncomp or *.cor files are NOT consistent"
        )
        if len(unw_files) > len(cor_files):
            print("skip creating ifgramStack.h5 file.")
            return
        print("Keeping only cor files which match a unw file")
        unw_dates_set = set([tuple(get_dates(f)) for f in unw_files])
        cor_files = [f for f in cor_files if tuple(get_dates(f)) in unw_dates_set]

    # get date info: date12_list
    date12_list = _get_date_pairs(unw_files)

    # TODO: compute the spatial baseline using COMPASS metadata
    pbase = np.zeros(num_pair, dtype=np.float32)

    # size info
    cols, rows = io.get_raster_xysize(unw_files[0])

    # define (and fill out some) dataset structure
    date12_arr = np.array([x.split("_") for x in date12_list], dtype=np.string_)
    drop_ifgram = np.ones(num_pair, dtype=np.bool_)
    ds_name_dict = {
        "date": [date12_arr.dtype, (num_pair, 2), date12_arr],
        "bperp": [np.float32, (num_pair,), pbase],
        "dropIfgram": [np.bool_, (num_pair,), drop_ifgram],
        "unwrapPhase": [np.float32, (num_pair, rows, cols), None],
        "coherence": [np.float32, (num_pair, rows, cols), None],
        "connectComponent": [
            np.float32,
            (num_pair, rows, cols),
            None,
        ],
    }

    # initiate HDF5 file
    meta["FILE_TYPE"] = "ifgramStack"
    writefile.layout_hdf5(outfile, ds_name_dict, metadata=meta)
    
    # writing data to HDF5 file
    print("writing data to HDF5 file {} with a mode ...".format(outfile))
    with h5py.File(outfile, "a") as f:
        prog_bar = ptime.progressBar(maxValue=num_pair)
        for i, (unw_file, cor_file, cc_file) in enumerate(
            zip(unw_files, cor_files, cc_files)
        ):
            # read/write *.unw file
            if metadata['package'] == 'dolphin':
                f["unwrapPhase"][i] = io.load_gdal(unw_file, band=1)
            else:
                f["unwrapPhase"][i] = io.load_gdal(unw_file, band=2)

            # read/write *.cor file
            f["coherence"][i] = io.load_gdal(cor_file)

            # read/write *.unw.conncomp file
            f["connectComponent"][i] = io.load_gdal(cc_file)

            prog_bar.update(i + 1, suffix=date12_list[i])
        prog_bar.close()

    print("finished writing to HDF5 file: {}".format(outfile))
    return outfile


def stitch_geometry(geom_path_list, geom_dir, meta, dem_file, matching_file):
    os.makedirs(geom_dir, exist_ok=True)
    #file_list = []
    #for burst, files in group_by_burst(geom_path_list, minimum_images=1).items():
    #    if len(files) > 1:
    #        print(f"Found {len(files)} static_layers files for {burst}")
    #    file_list.append(files[0])
    #print(f"Stitching {len(file_list)} images.")

    strides = {"x": int(meta['RLOOKS']), "y": int(meta['ALOOKS'])}
    stitched_geom_files = []
    # local_incidence_angle needed by anyone?
    datasets = ["los_east", "los_north", "layover_shadow_mask", "local_incidence_angle"]
  
    for ds_name in datasets:
        outfile = geom_dir + f"/{ds_name}_full.tif"
        print(f"Creating {outfile}")
        stitched_geom_files.append(outfile)
        # Used to be:
        # /science/SENTINEL1/CSLC/grids/static_layers
        # we might also move this to dolphin if we do use the layers
        ds_path = f"/data/{ds_name}"
        cur_files = [io.format_nc_filename(f, ds_name=ds_path) for f in geom_path_list]

        stitching.merge_images(
            cur_files,
            outfile=outfile,
            driver="GTiff",
            out_bounds=meta['bbox'],
            out_bounds_epsg=4326,
            target_aligned_pixels=True,
            in_nodata=0,
            strides=strides,
            resample_alg="nearest",
            overwrite=False,
        )

    #matching_file = '/net/kraken/nobak/smirzaee/Folsom/sequential/crop/scratch/unwrapped/20170114_20170120.unw.tif'

    height_file = geom_dir + "/height_rsm.tif"
    print(f"Creating {height_file}")
    stitched_geom_files.append(height_file)
    stitching.warp_to_match(
        input_file=dem_file,
        match_file=matching_file,
        output_file=height_file,
        resample_alg="cubic",
    )

    for ds_name in datasets:
        inpfile = geom_dir + f"/{ds_name}_full.tif"
        outfile = geom_dir + f"/{ds_name}_rsm.tif"
        print(f"Creating {outfile}")

        stitching.warp_to_match(
            input_file=inpfile,
            match_file=matching_file,
            output_file=outfile,
            resample_alg="cubic",
        )

    return

def get_data_bounds(file_path):
    dataset = gdal.Open(file_path, gdal.GA_ReadOnly)
    # Get the geospatial information
    geotransform = dataset.GetGeoTransform()

    # Define the source and target coordinate reference systems
    source_srs = osr.SpatialReference()
    source_srs.ImportFromWkt(dataset.GetProjection())

    # Create a target coordinate reference system for WGS 84 (lat/lon)
    target_srs = osr.SpatialReference()
    target_srs.ImportFromEPSG(4326)  # EPSG 4326 is WGS 84

    # Create a transformation from the source CRS to the target CRS
    transform = osr.CoordinateTransformation(source_srs, target_srs)

    # Calculate the corner coordinates in latitude and longitude
    # Bottom-left corner
    max_y, min_x, _ = transform.TransformPoint(geotransform[0], geotransform[3], 0)

    # Top-right corner
    min_y, max_x, _ = transform.TransformPoint(geotransform[0] + dataset.RasterXSize * geotransform[1], geotransform[3] + dataset.RasterYSize * geotransform[5], 0)
    bbox = (min_x, min_y, max_x, max_y)
    
    return bbox

def main(iargs=None):
    """Run the preparation functions."""
    inps = cmd_line_parse(iargs)

    unw_files = sorted(glob.glob(inps.unw_file_glob))
    print(f"Found {len(unw_files)} unwrapped files")
    cor_files = sorted(glob.glob(inps.cor_file_glob))
    print(f"Found {len(cor_files)} correlation files")

    dem_file = os.path.abspath(os.path.dirname(inps.geom_dir) + '/dem.tif')

    # translate input options
    processor = "sweets"  # isce_utils.get_processor(inps.meta_file)
    # metadata
    meta_file = Path(inps.meta_file)
    if meta_file.is_dir():
        # Search for the line of sight static_layers file
        try:
            # Grab the first one in the directory
            meta_file = next(meta_file.rglob("static_*.h5"))
        except StopIteration:
            raise ValueError(f"No static layers file found in {meta_file}")

    meta = prepare_metadata(
        meta_file, unw_files[0], nlks_x=inps.lks_x, nlks_y=inps.lks_y
    )

    if not inps.subset_lon is None:
        meta['bbox'] = (inps.subset_lon[0], inps.subset_lat[0], inps.subset_lon[1], inps.subset_lat[1])
    else:
        meta[ 'bbox'] = get_data_bounds(unw_files[0])

    meta['package'] = inps.package
    # output directory
    for dname in [inps.out_dir, os.path.join(inps.out_dir, "inputs")]:
        os.makedirs(dname, exist_ok=True)

    stack_file = os.path.join(inps.out_dir, "inputs/ifgramStack.h5")
    ts_file = os.path.join(inps.out_dir, "timeseries.h5")
    geom_file = os.path.join(inps.out_dir, "inputs/geometryGeo.h5")
    
    if inps.package == 'dolphin':
        geom_path_list = glob.glob(inps.meta_file + '/*/*/static*.h5')
        stitch_geometry(geom_path_list=geom_path_list, geom_dir=os.path.abspath(inps.geom_dir),
                        meta=meta, dem_file=dem_file, matching_file=unw_files[0])

    prepare_geometry(geom_file, geom_dir=inps.geom_dir, metadata=meta)

    if inps.single_reference:
        # time-series (if inputs are all single-reference)
        prepare_timeseries(
            outfile=ts_file,
            unw_files=unw_files,
            metadata=meta,
            processor=processor,
            # baseline_dir=inps.baseline_dir,
        )

    # prepare ifgstack with connected components
    prepare_stack(
        outfile=stack_file,
        unw_files=unw_files,
        cor_files=cor_files,
        metadata=meta,
        # baseline_dir=inps.baseline_dir,
    )

    print("Done.")
    return


if __name__ == "__main__":
    main(sys.argv[1:])
