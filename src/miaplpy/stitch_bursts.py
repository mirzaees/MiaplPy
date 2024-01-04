#!/usr/bin/env python3
########################
# Author: Sara Mirzaee
#######################
import os
import time
import argparse
import glob
from dolphin import stitching
import datetime
import h5py
from pyproj import CRS
import numpy as np
from osgeo import gdal
import shutil
from mintpy.utils import writefile, readfile, utils as ut
from pyproj import CRS
from mintpy.constants import EARTH_RADIUS

EXAMPLE = """example:
  stitch_bursts.py -i ./miaplpy*/inverted/interferograms_single_reference -g ./gslcs/t*/20170101/static*.h5  -o ./stitched_miaplpy
  """

DEFAULT_ENVI_OPTIONS = (
    "INTERLEAVE=BIL",
    "SUFFIX=ADD"
)

unwrap_options = {'two-stage': True,
                  'removeFilter': True,
                  'maxDiscontinuity': 1.2,
                  'initMethod': 'MCF',
                  'tileNumPixels': 10000000,
                  }

GEODATASETS = ['layover_shadow_mask', 'local_incidence_angle', 'los_east', 'los_north', 'x', 'y', 'z']


def create_parser():
    """Command line parser."""
    parser = argparse.ArgumentParser(description='Stitch interferograms and geometry.',
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     epilog=EXAMPLE)
    parser.add_argument('-t', '--template', type=str,
                        dest='template_file', help='Template file with path info.')
    parser.add_argument('-i', '--ifg-dir', dest='ifg_dir', type=str,
                        default='./miaplpy*/inverted/interferograms_single_reference',
                        help='A directory pattern to find interferograms\n'
                             'e.g.: ./miaplpy*/inverted/interferograms_single_reference')
    parser.add_argument('-g', '--geometry-dir', dest='geometry_dir', type=str, default='./gslcs/t*',
                        help=' directory with geometry files in bursts e.g.: ./gslcs/t*')
    parser.add_argument('-p', '--geom-pattern', dest='geo_pattern', type=str, default='static*.h5',
                        help='pattern of geometry files e.g.: static_layers*.h5')
    parser.add_argument('-o', '--output-dir', dest='output_dir', type=str, default=None,
                        help=' Output directory for stitched files e.g.: ./stitched_miaplpy')
    parser.add_argument('-b', '--bbox', dest='bbox', nargs=4, type=float, default=None,
                        help=("Bounding box of area of interest in decimal degrees longitude/latitude: \n"
                              "  (e.g. --bbox -106.1 30.1 -103.1 33.1 ). \n"))

    return parser


def cmd_line_parse(iargs=None):
    parser = create_parser()
    inps = parser.parse_args(args=iargs)
    inps.output_dir = os.path.abspath(inps.output_dir)
    return inps


def stitch_bursts(iargs=None):
    inps = cmd_line_parse(iargs)
    os.makedirs(inps.output_dir, exist_ok=True)
    template_file = os.path.join(inps.output_dir, 'miaplpyApp.cfg')
    shutil.copy(inps.template_file, template_file)
    # template = readfile.read_template(template_file)

    out_ifg_dir = os.path.join(inps.output_dir, 'interferograms')
    out_geometry_dir = os.path.join(inps.output_dir, 'geometry')
    corr_dir = os.path.join(inps.output_dir, 'correlations')

    os.makedirs(out_ifg_dir, exist_ok=True)
    os.makedirs(out_geometry_dir, exist_ok=True)
    os.makedirs(corr_dir, exist_ok=True)

    ifg_list = glob.glob(os.path.abspath(inps.ifg_dir))
    ifg_dict = {}
    cor_dict = {}
    for i, item in enumerate(ifg_list):
        ifg_dict[f'i{i}'] = sorted(glob.glob(item + '/*.int.tif'))
        cor_dict[f'i{i}'] = sorted(glob.glob(item + '/*.cor.tif'))

    date_lists = [os.path.basename(x).split('.')[0] for x in ifg_dict['i0']]
    grouped_ifgs = {date: [] for date in date_lists}
    grouped_cors = {date: [] for date in date_lists}
    for i, date in enumerate(date_lists):
        for key in ifg_dict.keys():
            if date in ifg_dict[key][i]:
                grouped_ifgs[date].append(ifg_dict[key][i])
                grouped_cors[date].append(cor_dict[key][i])

    for dates, cur_images in grouped_ifgs.items():
        outfile = out_ifg_dir + f"/{dates}.tif"
        if not os.path.exists(outfile):
            stitching.merge_images(
                cur_images,
                outfile=outfile,
                driver="ENVI",
                out_bounds=inps.bbox,
                out_bounds_epsg=4326,
                target_aligned_pixels=True,
                overwrite=True,
                strides={"x": 6, "y": 3},
                in_nodata=np.nan
            )

    for dates, cur_images in grouped_cors.items():
        outfile = corr_dir + f"/{dates}.tif"
        if not os.path.exists(outfile):
            stitching.merge_images(
                cur_images,
                outfile=outfile,
                driver="ENVI",
                out_bounds=inps.bbox,
                out_bounds_epsg=4326,
                target_aligned_pixels=True,
                overwrite=True,
                strides={"x": 6, "y": 3},
                in_nodata=np.nan
            )

    geom_files = glob.glob(os.path.abspath(inps.geometry_dir))
    box_size = 3000
    with h5py.File(geom_files[0], 'r') as ds:
        keys = [key for key in ds['data'].keys() if key in GEODATASETS]
    for file in geom_files:
        geotransform, projection, nodata = get_projection(file)
        with h5py.File(file, 'r') as ds:
            local_inc_ang = ds['data']['local_incidence_angle'][()]
            # mask = np.isnan(local_inc_ang)
            for key in keys:
                length, width = ds['data'][key].shape
                out_file = os.path.dirname(file) + f"/{key}.geo"
                if not os.path.exists(out_file):
                    dtype = gdal.GDT_Float32
                    driver = gdal.GetDriverByName('ENVI')
                    out_raster = driver.Create(out_file, width, length, 1, dtype, DEFAULT_ENVI_OPTIONS)
                    band = out_raster.GetRasterBand(1)
                    for i in range(0, length, box_size):
                        for j in range(0, width, box_size):
                            data = ds['data'][key][i:i + box_size, j:j + box_size].astype(float)
                            band.WriteArray(data, j, i)
                    band.SetNoDataValue(np.nan)
                    out_raster.FlushCache()
                    out_raster = None
                    write_projection(geotransform, projection, nodata, out_file)

    geo_stitch = {key: [] for key in keys}
    for file in geom_files:
        for key in keys:
            geo_stitch[key].append(os.path.dirname(file) + f"/{key}.geo")

    for key in keys:
        outfile = out_geometry_dir + f"/{key}.geo"
        if not os.path.exists(outfile):
            stitching.merge_images(
                geo_stitch[key][:],
                outfile=outfile,
                driver="ENVI",
                out_bounds=inps.bbox,
                out_bounds_epsg=4326,
                target_aligned_pixels=True,
                overwrite=True,
                strides={"x": 6, "y": 3}
            )
    
    datasets = ['tempCoh_average', 'mask_ps']
    for dset in datasets:
        tcoh_file = [x + f"/{dset}.tif" for x in glob.glob(os.path.dirname(os.path.abspath(inps.ifg_dir)))]
        outfile = inps.output_dir + f"/geometry/{dset}.tif"
   
        if not os.path.exists(outfile):
            stitching.merge_images(
                tcoh_file,
                outfile=outfile,
                driver="ENVI",
                out_bounds=inps.bbox,
                out_bounds_epsg=4326,
                target_aligned_pixels=True,
                overwrite=True,
                strides={"x": 6, "y": 3},
                in_nodata=0
            )

    matching_file = out_ifg_dir + f"/{date_lists[0]}.tif"

    for dset in datasets:

        inpfile = inps.output_dir + f"/geometry/{dset}.tif"
        outfile = inps.output_dir + f"/geometry/{dset}_rsm.tif"
        print(f"Creating {outfile}")

        stitching.warp_to_match(
            input_file=inpfile,
            match_file=matching_file,
            output_file=outfile,
            resample_alg="cubic",
        )

    for key in keys:
        inpfile = inps.output_dir + f"/geometry/{key}.geo"
        outfile = inps.output_dir + f"/geometry/{key}_rsm.tif"
        print(f"Creating {outfile}")

        stitching.warp_to_match(
            input_file=inpfile,
            match_file=matching_file,
            output_file=outfile,
            resample_alg="cubic",
        )

    inpfile = inps.output_dir + f"/dem.tif"
    outfile = inps.output_dir + f"/geometry/height_rsm.tif"
    print(f"Creating {outfile}")

    stitching.warp_to_match(
        input_file=inpfile,
        match_file=matching_file,
        output_file=outfile,
        resample_alg="cubic",
    )

    run_unwrap(inps.output_dir, out_ifg_dir, geom_files[0])

    return


def get_projection(file):
    with h5py.File(file, 'r') as ds:
        dsg = ds['data']['projection'].attrs      
        x_step = float(ds['data']['x_spacing'][()])
        y_step = float(ds['data']['y_spacing'][()])
        x_first = min(ds['data']['x_coordinates'][()])
        y_first = max(ds['data']['y_coordinates'][()])
        geotransform = (x_first, x_step, 0, y_first, 0, y_step)
        projection = CRS.from_wkt(dsg['spatial_ref'].decode('utf-8'))
        nodata = np.nan

    return geotransform, projection, nodata


def write_projection(geotransform, projection, nodata, dst_file) -> None:

    ds_dst = gdal.Open(dst_file, gdal.GA_Update)
    ds_dst.SetGeoTransform(geotransform)
    ds_dst.SetProjection(projection.to_wkt())
    ds_dst.GetRasterBand(1).SetNoDataValue(nodata)
    ds_src = ds_dst = None
    return


def run_unwrap(out_dir, ifg_dir, reference_file, write_job=False, job_obj=None):
    """ Unwraps interferograms
    """
    ifg_list = glob.glob(ifg_dir + '/*.tif')
    ifg_list = [os.path.abspath(x) for x in ifg_list]
    ds = gdal.Open(ifg_list[0], gdal.GA_ReadOnly)
    width = ds.RasterXSize
    length = ds.RasterYSize
    num_pixels = length * width

    with h5py.File(reference_file, 'r') as rf:
        attrs = rf['metadata']
        wavelength = float(attrs['processing_information']['input_burst_metadata']['wavelength'][()])
        earth_radius = EARTH_RADIUS
        height = 693000.0

    run_file_unwrap = os.path.join(out_dir, 'run_unwrap')
    print('Generate {}'.format(run_file_unwrap))

    run_commands = []
    num_cpu = 4
    ntiles = num_pixels // unwrap_options['tileNumPixels']
    if ntiles == 0:
        ntiles = 1
    num_cpu = num_cpu // ntiles
    num_lin = 0

    out_unwrap_dir = os.path.join(out_dir, 'unwrap')
    os.makedirs(out_unwrap_dir, exist_ok=True)
    config_dir = os.path.join(out_unwrap_dir, 'unwrap_configs')
    os.makedirs(config_dir, exist_ok=True)

    corr_file = os.path.join(out_dir, 'geometry/tempCoh_average_rsm.tif')

    for inp_ifg in ifg_list:
        unw_file = os.path.basename(inp_ifg).split('.int')[0] + '.unw'
        out_ifg = os.path.abspath(os.path.join(out_unwrap_dir, unw_file))

        scp_args = '--ifg {a1} --coherence {a2} --unwrapped_ifg {a3} ' \
                   '--max_discontinuity {a4} --init_method {a5} --length {a6} ' \
                   '--width {a7} --height {a8} --num_tiles {a9} --earth_radius {a10} ' \
                   ' --wavelength {a11} --tmp'.format(a1=inp_ifg, a2=corr_file, a3=out_ifg,
                                                      a4=unwrap_options['maxDiscontinuity'],
                                                      a5=unwrap_options['initMethod'],
                                                      a6=length, a7=width, a8=height, a9=ntiles,
                                                      a10=earth_radius, a11=wavelength)

        if unwrap_options['two-stage'] == 'yes':
            scp_args += ' --two-stage'
        cmd = 'unwrap_ifgram.py {}'.format(scp_args)
        cmd = cmd.lstrip()

        if not write_job:
            cmd = cmd + ' &\n'
            run_commands.append(cmd)
            num_lin += 1
            if num_lin == num_cpu:
                run_commands.append('wait\n\n')
                num_lin = 0
        else:
            cmd = cmd + '\n'
            run_commands.append(cmd)

    run_commands.append('wait\n\n')
    run_commands = [cmd.lstrip() for cmd in run_commands]

    with open(run_file_unwrap, 'w+') as frun:
        frun.writelines(run_commands)

    if write_job or not job_obj is None:
        job_obj.num_bursts = num_pixels // 3000000
        job_obj.write_batch_jobs(batch_file=run_file_unwrap, num_cores_per_task=ntiles)

    return


#########################################################################


if __name__ == '__main__':
    """Main driver."""
    stitch_bursts()
