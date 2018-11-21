#! /usr/bin/env python3
############################################################
# Copyright(c) 2017, Sara Mirzaee                          #
############################################################

import os
import sys
import time

import argparse
from numpy import linalg as LA
import numpy as np
from scipy.stats import anderson_ksamp
from skimage.measure import label
import _pysqsar_utilities as pysq
import pandas as pd
from dask import compute, delayed
sys.path.insert(0, os.getenv('RSMAS_ISCE'))
from dataset_template import Template



#################################
EXAMPLE = """example:
  PSQ_sentinel.py LombokSenAT156VV.template -p PATCH5_11
"""


def create_parser():
    """ Creates command line argument parser object. """

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, epilog=EXAMPLE)
    parser.add_argument('-v', '--version', action='version', version='%(prog)s 0.1')
    parser.add_argument('custom_template_file', nargs='?',
                        help='custom template with option settings.\n')
    parser.add_argument('-p','--patchdir', dest='patch_dir', type=str, required=True, help='patch file directory')

    return parser


def command_line_parse(args):
    """ Parses command line agurments into inps variable. """

    parser = create_parser()
    inps = parser.parse_args(args)
    return inps


def sequential_process(shp_df_chunk, seq_n, inps, pixels_dict={}, pixels_dict_ref={}):

    n_lines = np.shape(pixels_dict_ref['RSLC_ref'])[0]
    values = [delayed(pysq.phase_link)(x,pixels_dict=pixels_dict) for x in shp_df_chunk]
    
    results = compute(*values, scheduler='processes')
    print(results)
    results = pd.DataFrame(list(results))
    squeezed = np.zeros([inps.lin, inps.sam]) + 0j

    results_df = [results.loc[y] for y in range(len(results))]

    for item in results_df:
        lin,sam = item.at['ref_pixel'][0],item.at['ref_pixel'][1]

        pixels_dict_ref['RSLC_ref'][:, lin:lin + 1, sam:sam + 1] = \
            (np.multiply(item.at['amp_ref'][seq_n::], np.exp(1j*item.at['phase_ref'][seq_n::]))).reshape(n_lines,1,1)

        org_pixel = pixels_dict['RSLC'][seq_n::, lin, sam]

        map_pixel = np.exp(1j * item.at['phase_ref'][seq_n::, 0, 0]).reshape(n_lines, 1)
        map_pixel = np.matrix(map_pixel / LA.norm(map_pixel))

        squeezed[lin, sam] = np.matmul(map_pixel.getH(), org_pixel)

    return squeezed


###################################
def main(iargs=None):
    inps = command_line_parse(iargs)

    inps.project_name = os.path.basename(inps.custom_template_file).partition('.')[0]
    inps.project_dir = os.getenv('SCRATCHDIR') + '/' + inps.project_name
    inps.scratch_dir = os.getenv('SCRATCHDIR')
    
    inps.slave_dir = inps.project_dir + '/merged/SLC'
    inps.sq_dir = inps.project_dir + '/SqueeSAR'
    inps.list_slv = os.listdir(inps.slave_dir)
    inps.n_image = len(inps.list_slv)
    inps.work_dir = inps.sq_dir +'/'+ inps.patch_dir

    inps.patch_rows = np.load(inps.sq_dir + '/rowpatch.npy')
    inps.patch_cols = np.load(inps.sq_dir + '/colpatch.npy')
    patch_row, patch_col = inps.patch_dir.split('PATCH')[1].split('_')
    patch_row, patch_col = (int(patch_row), int(patch_col))

    inps.lin = inps.patch_rows[1][0][patch_row] - inps.patch_rows[0][0][patch_row]
    inps.sam = inps.patch_cols[1][0][patch_col] - inps.patch_cols[0][0][patch_col]

    rslc = np.memmap(inps.work_dir + '/RSLC', dtype=np.complex64, mode='r', shape=(inps.n_image, inps.lin, inps.sam))


    inps.range_win = int(Template(inps.custom_template_file).get_options()['squeesar.wsizerange'])
    inps.azimuth_win = int(Template(inps.custom_template_file).get_options()['squeesar.wsizeazimuth'])

    
    ###################### Sequential Phase linking ###############################
    

    rslc_ref = np.memmap(inps.work_dir + '/RSLC_ref', dtype='complex', mode='w+', shape=(inps.n_image, inps.lin, inps.sam))
    rslc_ref[:,:,:] = rslc[:,:,:]

    num_seq = np.int(np.floor(inps.n_image / 10))
    if os.path.isfile(inps.work_dir + '/sequential_df.pkl'):
        sequential_df = pd.read_pickle(inps.work_dir + '/sequential_df.pkl')
    else:
        sequential_df = pd.DataFrame(columns=['step_n', 'squeezed','datum_shift'])
        sequential_df = sequential_df.append({'step_n':0, 'squeezed':{}, 'datum_shift':{}}, ignore_index=True)

    step_0 = np.int(sequential_df.at[0,'step_n'])

    shp_df = pd.read_pickle(inps.work_dir + '/SHP.pkl')
    shp_df_chunk = [shp_df.loc[y] for y in range(len(shp_df))]

    time0 = time.time()
    for stepp in range(step_0, num_seq):

        first_line = stepp  * 10
        if stepp == num_seq-1:
            last_line = inps.n_image
        else:
            last_line = first_line + 10
        num_lines = last_line - first_line
        if stepp == 0:
            pixels_dict = {'RSLC': rslc[first_line:last_line, :, :]}
            pixels_dict_ref = {'RSLC_ref': rslc_ref[first_line:last_line, :, :]}

            squeezed_image = sequential_process(shp_df_chunk=shp_df_chunk, seq_n=stepp,
                                                inps=inps, pixels_dict=pixels_dict,
                                                pixels_dict_ref=pixels_dict_ref)
            sequential_df.at[0,'step_n'] = stepp
            sequential_df.at[0,'squeezed'] = squeezed_image
        else:
            rslc_seq = np.zeros([stepp + num_lines, inps.lin, inps.sam])+1j
            rslc_seq[0:stepp, :, :] = sequential_df.at[0,'squeezed']
            rslc_seq[stepp::, :, :] = rslc[first_line:last_line, :, :]
            pixels_dict = {'RSLC': rslc_seq}
            pixels_dict_ref = {'RSLC_ref': rslc_ref[first_line:last_line, :, :]}
            squeezed_im = sequential_process(shp_df_chunk=shp_df_chunk, seq_n=stepp,
                                             inps=inps, pixels_dict=pixels_dict,
                                             pixels_dict_ref=pixels_dict_ref)
            squeezed_image = np.dstack((sequential_df.at[0,'squeezed'].T,squeezed_im.T)).T

            sequential_df.at[0, 'step_n'] = stepp
            sequential_df.at[0, 'squeezed'] = squeezed_image


        sequential_df.to_pickle(inps.work_dir + '/sequential_df.pkl')
    
        

    ############## Datum Connection ##############################
    pixels_dict = {'RSLC': sequential_df.at[0,'squeezed']}

    values = [delayed(pysq.phase_link)(x,pixels_dict=pixels_dict) for x in shp_df_chunk]
    results = pd.DataFrame(list(compute(*values, scheduler='processes')))
    datum_connect = np.zeros([num_seq, inps.lin, inps.sam])
    datum_df = [results.loc[y] for y in range(len(results))]

    for item in datum_df:
        lin,sam = (item.at['ref_pixel'][0],item.at['ref_pixel'][1])
        datum_connect[:, lin:lin+1, sam:sam+1] = item.at['phase_ref'][:, 0, 0].reshape(num_seq, 1, 1)

    for stepp in range(num_seq):
        first_line = stepp * 10
        if stepp == num_seq-1:
            last_line = inps.n_image
        else:
            last_line = first_line + 10
        if step_0 == 0 or sequential_df.at[0, 'datum_shift']=={}:
            rslc_ref[first_line:last_line, :, :] = np.multiply(rslc_ref[first_line:last_line, :, :],
                                                               np.exp(1j*datum_connect[stepp, :, :]))
        else:
            rslc_ref[first_line:last_line, :, :] = \
                np.multiply(rslc_ref[first_line:last_line, :, :],
                            np.exp(1j * (datum_connect[stepp, :, :] - sequential_df.at[0, 'datum_shift'][stepp, :, :])))

    sequential_df.at[0, 'datum_shift'] = datum_connect
    np.save(inps.work_dir + '/endflag.npy', 'True')
    sequential_df.to_pickle(inps.work_dir + '/sequential_df.pkl')

    del rslc_ref, rslc
        
    timep = time.time() - time0
    print('time spent to do sequential phase linking {}: min'.format(timep/60))

if __name__ == '__main__':
    '''
    Phase linking process.
    '''
    main()

#################################################
