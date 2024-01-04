#! /usr/bin/env python3
############################################################
# Copyright(c) 2017, Sara Mirzaee                          #
############################################################

import os
import matplotlib
#matplotlib.use('TKagg')
matplotlib.use('Tkagg')
import matplotlib.pyplot as plt

import numpy as np
import csv
from numpy import linalg as LA
from scipy.linalg import lapack as lap
from math import sqrt, exp, isnan, log
from scipy.optimize import minimize
from skimage.measure._ccomp import label_cython as clabel
from scipy.stats import anderson_ksamp, ttest_ind
from mintpy.utils import ptime
from mintpy.utils import readfile
import time
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)

def incremental_iterate(patch_slc_images, out_folder, default_mini_stack_size, first_new_image, coords, phase_linking_method,
                        rslc_ref, tempCoh, mask_ps, SHP, NUMSHP):

    box_length = np.shape(rslc_ref)[1]
    box_width = np.shape(rslc_ref)[2]
    num_points = np.shape(coords)[0]
    num_new = np.shape(rslc_ref)[0] - first_new_image
    do_compress = False
    squeezed_images = np.zeros((1, box_length, box_width), dtype=np.complex64)
    new_compressed = np.zeros((1, box_length, box_width), dtype=np.complex64)
    
    if os.path.exists(out_folder.decode('UTF-8') + '/compressed.npy'):
        squeezed_images = np.load(out_folder.decode('UTF-8') + '/compressed.npy', allow_pickle=True)[:, :, :]
        num_compressed = np.shape(squeezed_images)[0]
        new_compressed = np.zeros((num_compressed, box_length, box_width), dtype=np.complex64)
    else:
        squeezed_images = np.zeros((1, box_length, box_width), dtype=np.complex64)
        num_compressed = 0
        new_compressed = np.zeros((1, box_length, box_width), dtype=np.complex64)
    
    first_old_image = first_new_image - default_mini_stack_size 
    num_old_images = default_mini_stack_size - 1
    if first_old_image < 0:
        first_old_image = 0
        num_old_images = 1
    
    prog_bar = ptime.progressBar(maxValue=np.shape(rslc_ref)[0])
    for i in range(first_new_image, np.shape(rslc_ref)[0]):
        if i % default_mini_stack_size == 0:
            new_compressed = np.zeros((num_compressed + 1, box_length, box_width), dtype=np.complex64)
            do_compress = True
        else:
            do_compress = False

        for t in range(num_compressed):
            new_compressed[t, :, :] = squeezed_images[t, :, :]

        num_images_in_stack = num_old_images + num_compressed + 1
        for t in range(num_points):
            data = (coords[i,0], coords[i,1])
            num_shp = NUMSHP[data[0], data[1]]
            if num_shp == 0:
                continue
            
            if do_compress == True:
                vec = np.zeros(default_mini_stack_size, dtype=np.complex64)
                for m in range(first_old_image, num_old_images + 1):
                    vec[m] = patch_slc_images[m, data[0], data[1]]

            if mask_ps[data[0], data[1]] == 1:
                if do_compress == True:
                    new_compressed[num_compressed + 1, data[0], data[1]] = compress_slcs(vec, vec)
            else:

                CCG = np.zeros((num_images_in_stack, num_shp))
                for m in range(num_shp):
                    row = int(np.real(SHP[data[0], data[1], m]))
                    col = int(np.imag(SHP[data[0], data[1], m]))
                    for p in range(num_compressed):
                        CCG[p, m] = squeezed_images[p, row, col]
                    for p in range(num_old_images + 1):
                        CCG[p + num_compressed, m] = patch_slc_images[first_old_image + p, row, col]
                
                vec_refined, noval, temp_quality, max_coh_index = phase_linking_process_cy(CCG, 0, phase_linking_method, 0)
                amp_refined = mean_along_axis_x(np.abs(CCG))
                if do_compress == True:
                    new_compressed[num_compressed + 1, data[0], data[1]] = compress_slcs(vec, vec_refined)

                if num_compressed == 0:
                    x0 = np.exp(-1j * np.angle(vec_refined[0]))
                else:
                    x0 = np.exp(-1j * np.angle(vec_refined[num_compressed]))

                rslc_ref[i, data[0], data[1]] = amp_refined[num_images_in_stack - 1] * np.exp(1j * np.angle(vec_refined[num_images_in_stack - 1])) * x0
                tempCoh[data[0], data[1]] = (tempCoh[data[0], data[1]] + temp_quality)/2
    

            if do_compress == True:
                squeezed_images = np.zeros((num_compressed + 1, box_length, box_width), dtype=np.complex64)
                squeezed_images[:, :, :] = new_compressed[:, :, :]
                num_compressed += 1

            first_old_image += 1
            num_old_images = i - first_old_image
        
        prog_bar.update(i, every=1, suffix='image {}/{} incremental {} pixels ministack {}'.format(i, np.shape(rslc_ref)[0], num_points, os.path.basename(out_folder.decode('UTF-8'))))

    np.save(out_folder.decode('UTF-8') + '/phase_ref.npy', rslc_ref)
    np.save(out_folder.decode('UTF-8') + '/tempCoh.npy', tempCoh)
    np.save(out_folder.decode('UTF-8') + '/compressed.npy', squeezed_images)
     
    return

def time_referencing(patch_slc_images, n_image, y, x):
    vec_refined = np.zeros(n_image, dtype=np.complex64)
    amp_refined =  np.zeros(n_image, dtype=np.float32)
    #cdef float complex x0

    x0 = np.exp(-1j * np.angle(patch_slc_images[0, y, x]))

    for m in range(n_image):
        vec_refined[m] = np.exp(1j * np.angle(patch_slc_images[m, y, x]))  * x0
        amp_refined[m] = np.angle(patch_slc_images[m, y, x])

    return vec_refined, amp_refined

def test_PS_cy(amplitude):
    """ checks if the pixel is PS """
        
    amp_dispersion = np.std(amplitude)/np.mean(amplitude)
    if amp_dispersion > 1:
        amp_dispersion = 1

    if amp_dispersion < 0.25:
        temp_quality = 1
    else:
        temp_quality = 0.01
    return temp_quality, 0

def process_patch_c(box, range_window, azimuth_window, width, length, n_image,
                    slcStackObj, distance_threshold, def_sample_rows, def_sample_cols, reference_row, reference_col,
                    phase_linking_method, total_num_mini_stacks, default_mini_stack_size,
                    ps_shp, shp_test, out_dir, lag, mask_file, num_archived):
    
    patch_slc_images = slcStackObj.read(datasetName='slc', box=box, print_msg=False)
    box_width = box[2] - box[0]
    box_length = box[3] - box[1]
    reference_index = np.zeros((box_length, box_width), dtype=np.int32)
    rslc_ref = np.zeros((n_image, box_length, box_width), dtype=np.complex64)
    tempCoh = np.zeros((box_length, box_width), dtype=np.float32)
    mask_ps = np.zeros((box_length, box_width), dtype=np.int32)
    SHP = np.zeros((box_length, box_width, range_window*azimuth_window), dtype=np.complex64)
    NUMSHP = np.zeros((box_length, box_width), dtype=np.int32)
    row1 = 0 
    row2 = box_length 
    col1 = 0 
    col2 = box_width 
    coords = np.zeros((box_length*box_width, 2), dtype=np.int32)
    #vec_refined = np.zeros(n_image, dtype=np.complex64)
    #amp_refined =  np.zeros(n_image, dtype=np.float32)
    index = box[4]
    time0 = time.time()
    mask = np.ones((box_length, box_width), dtype=np.int32)
  
    max_coh_index = 0

    if os.path.exists(mask_file.decode('UTF-8')):
        mask = (readfile.read(mask_file.decode('UTF-8'),
                              box=(box[0], box[1], box[2], box[3]))[0]*1).astype(np.int32)

    out_folder = out_dir + ('/PATCHES/PATCH_{:04.0f}'.format(index)).encode('UTF-8')
    
    os.makedirs(out_folder.decode('UTF-8'), exist_ok=True)
    if os.path.exists(out_folder.decode('UTF-8') + '/flag.npy'):
        return
    
    m = 0
    for i in range(box_length):
        for t in range(box_width):
            if mask[i, t] and not np.isnan(patch_slc_images[:, i, t]).any():
                coords[m, 0] = i #+ row1
                coords[m, 1] = t #+ col1
                m += 1

    #coords = np.array([[100, 100],[101, 100],[100, 101]])
    m = 4
    
    num_points = m - 1    
    
    print('\nFinding PS pixels PATCH {}'.format(index))
    prog_bar = ptime.progressBar(maxValue=num_points)
    
    if not os.path.exists(out_folder.decode('UTF-8') + '/shpp.npy'):
        for i in range(num_points): 
        
            data = (coords[i,0], coords[i,1])
            num_shp = 0
            shp = get_shp_row_col_c(data, patch_slc_images, def_sample_rows, def_sample_cols, azimuth_window,
                                    range_window, reference_row, reference_col, distance_threshold, shp_test)
          

            num_shp = np.shape(shp)[0]
            
            for t in range(num_shp):
                SHP[data[0], data[1], t] = shp[t]
                NUMSHP[data[0], data[1]] = num_shp

            if num_shp <= ps_shp:

                max_coh_index = 0

                vec_refined, amp_refined = time_referencing(patch_slc_images, n_image, data[0], data[1])

                temp_quality, reference_index[data[0], data[1]] = test_PS_cy(amp_refined)

                if temp_quality == 1:
                    mask_ps[data[0], data[1]] = 1

                
                x0 = np.exp(-1j * np.angle(vec_refined[0]))
                for m in range(n_image):
                    rslc_ref[m, data[0], data[1]] = amp_refined[m] * np.exp(1j * np.angle(vec_refined[m])) * x0

                if temp_quality < 0:
                    temp_quality = 0
                
                tempCoh[data[0], data[1]] = temp_quality        
            prog_bar.update(i, every=200, suffix='{}/{} PS pixels, patch {}'.format(i, num_points, index))
        

        np.save(out_folder.decode('UTF-8') + '/num_shp.npy', NUMSHP)
        np.save(out_folder.decode('UTF-8') + '/shp.npy', SHP)
        np.save(out_folder.decode('UTF-8') + '/tempCoh.npy', tempCoh)
        np.save(out_folder.decode('UTF-8') + '/mask_ps.npy', mask_ps)

    else:
        NUMSHP = np.load(out_folder.decode('UTF-8') + '/num_shp.npy', allow_pickle=True)[:, :]
        SHP = np.load(out_folder.decode('UTF-8') + '/shp.npy', allow_pickle=True)[:, :, :]
        tempCoh = np.load(out_folder.decode('UTF-8') + '/tempCoh.npy', allow_pickle=True)[:, :]
        mask_ps = np.load(out_folder.decode('UTF-8') + '/mask_ps.npy', allow_pickle=True)[:, :]

        for i in range(num_points): 
            data = (coords[i,0], coords[i,1])
            num_shp =NUMSHP[data[0], data[1]] 
            if num_shp <= ps_shp:
                vec_refined, amp_refined = time_referencing(patch_slc_images, n_image, data[0], data[1])
                x0 = np.exp(-1j * np.angle(vec_refined[0]))
                for m in range(n_image):
                    rslc_ref[m, data[0], data[1]] = amp_refined[m] * np.exp(1j * np.angle(vec_refined[m])) * x0


    print('\nDS phase linking PATCH {}'.format(index))


    if len(phase_linking_method) > 10 and phase_linking_method[0:10] == b'sequential':
        
        sequential_phase_linking(patch_slc_images, out_folder,
                                 default_mini_stack_size, total_num_mini_stacks, 
                                 coords, phase_linking_method, 
                                 rslc_ref, tempCoh, mask_ps, SHP, NUMSHP)


    elif phase_linking_method[0:9] == b'real_time':
        if num_archived > default_mini_stack_size:
            num_archived_stacks = num_archived // default_mini_stack_size
            sequential_phase_linking(patch_slc_images, out_folder, 
                                     default_mini_stack_size, num_archived_stacks, 
                                     coords, phase_linking_method,
                                     rslc_ref, tempCoh, mask_ps, SHP, NUMSHP)
            first_new_image = num_archived_stacks * default_mini_stack_size
        else:
            first_new_image = 1
            incremental_iterate(patch_slc_images, out_folder, 
                                default_mini_stack_size, first_new_image, 
                                coords, phase_linking_method,
                                rslc_ref, tempCoh, mask_ps, SHP, NUMSHP)

    else:

        prog_bar = ptime.progressBar(maxValue=num_points)

        reference_index = np.zeros((box_length, box_width), dtype=np.int32)
        for i in range(num_points):
            data = (coords[i,0], coords[i,1])
            num_shp = NUMSHP[data[0], data[1]]
            
            CCG = np.zeros((n_image, num_shp), dtype=np.complex64)
            for m in range(num_shp):
              
                row = int(np.real(SHP[data[0], data[1], m]))
                col = int(np.imag(SHP[data[0], data[1], m]))
                for p in range(n_image):
                    CCG[p, m] = patch_slc_images[p, row, col]
            
            vec_refined, noval, temp_quality, reference_index[data[0], data[1]] = phase_linking_process_cy(CCG, 0, phase_linking_method, lag)
            amp_refined = mean_along_axis_x(np.abs(CCG))

            for m in range(n_image):
                rslc_ref[m, data[0], data[1]] = amp_refined[m] * np.exp(1j * np.angle(vec_refined[m])) 
            
            tempCoh[data[0], data[1]] = temp_quality

            prog_bar.update(i, every=200, suffix='{}/{} DS pixels, patch {}'.format(i, num_points, index))
            
        np.save(out_folder.decode('UTF-8') + '/phase_ref.npy', rslc_ref)

        np.save(out_folder.decode('UTF-8') + '/tempCoh.npy', tempCoh)

        np.save(out_folder.decode('UTF-8') + '/reference_index.npy', reference_index)

    np.save(out_folder.decode('UTF-8') + '/flag.npy', [1])


    mi, se = divmod(time.time()-time0, 60)
    print('    Phase inversion of PATCH_{:04.0f} is Completed in {:02.0f} mins {:02.0f} secs\n'.format(index, mi, se))

    return

def compress_slcs(org_vec, ref_vec):
    out = 0
    n = np.shape(org_vec)[0]
    N = np.shape(ref_vec)[0]
    vm = np.zeros(n, dtype=np.complex64)
    f = N - n

    for i in range(n):
        vm[i] = np.exp(-1j * np.angle(ref_vec[i + f]))

    normm = norm_complex(vm)

    for i in range(n):
            out += org_vec[i] * (vm[i])/normm
    return out

def norm_complex( x):
    n = np.shape(x)[0]
    out = 0
    for i in range(n):
        out += np.abs(x[i])**2
    out = sqrt(out)
    return out


def sequential_phase_linking(patch_slc_images, out_folder, default_mini_stack_size,  total_num_mini_stacks, coords,  
                             phase_linking_method, rslc_ref, tempCoh, mask_ps, SHP, NUMSHP):

    box_length = np.shape(rslc_ref)[1]
    box_width = np.shape(rslc_ref)[2]
    reference_index = np.zeros((box_length, box_width), dtype=np.int32)
    num_points = np.shape(coords)[0]
    squeezed_images = np.zeros((total_num_mini_stacks, box_length, box_width), dtype=np.complex64)
    n_image = np.shape(rslc_ref)[0]
    vec_refined = np.zeros(n_image, dtype=np.complex64)
    
    prog_bar = ptime.progressBar(maxValue=total_num_mini_stacks * num_points)
    for i in range(total_num_mini_stacks):
        
        first_line = i * default_mini_stack_size
        last_line = first_line + default_mini_stack_size 
        
        if i == total_num_mini_stacks - 1 and n_image//default_mini_stack_size==total_num_mini_stacks:
            last_line = n_image

        num_lines = last_line - first_line
   
        for t in range(num_points):
           
            data = (coords[t,0], coords[t,1])
            num_shp = NUMSHP[data[0], data[1]]
            if num_shp == 0:
                continue
            
            vec = np.zeros(num_lines, np.complex64)
            for m in range(first_line, last_line):
                vec[m-first_line] = patch_slc_images[m, data[0], data[1]]

            if mask_ps[data[0], data[1]] == 1:
                squeezed_images[i, data[0], data[1]] = compress_slcs(vec, vec)
            else:
                CCG = np.zeros((i + num_lines, num_shp), dtype=np.complex64)
                for m in range(num_shp):
                    
                    row = int(np.real(SHP[data[0], data[1], m]))
                    col = int(np.imag(SHP[data[0], data[1], m]))
                    if i > 0:
                        for p in range(i):
                            CCG[p, m] = squeezed_images[p, row, col]
                    for p in range(num_lines):
                        CCG[p + i, m] = patch_slc_images[first_line + p, row, col]
                    
                vec_refined, noval, temp_quality, max_coh_index = phase_linking_process_cy(CCG, 0, phase_linking_method, 0)
                amp_refined = mean_along_axis_x(np.abs(CCG))
                
                squeezed_images[i, data[0], data[1]] = compress_slcs(vec, vec_refined)
                
                if i == 0:
                    reference_index[data[0], data[1]] = max_coh_index
                
                for m in range(first_line, last_line):
                    rslc_ref[m, data[0], data[1]] = amp_refined[m + i - first_line] * np.exp(1j * np.angle(vec_refined[m + i - first_line])) 
                
                tempCoh[data[0], data[1]] += temp_quality/total_num_mini_stacks

            
            patch = os.path.basename(out_folder.decode('UTF-8'))
        
            prog_bar.update(i, every=200, suffix='{}/{} DS pixels ministack {}, {}'.format(t, num_points, i, patch))
    

    np.save(out_folder.decode('UTF-8') + '/phase_ref.npy', rslc_ref)
    np.save(out_folder.decode('UTF-8') + '/tempCoh.npy', tempCoh)
    np.save(out_folder.decode('UTF-8') + '/compressed.npy', squeezed_images)
    np.save(out_folder.decode('UTF-8') + '/reference_index.npy', reference_index)

    return 

def mean_along_axis_x(x):
    n = np.shape(x)[0]
    out = np.zeros(n, dtype=np.float32)
    temp = 0
    for i in range(n):
        temp = 0
        for t in range(np.shape(x)[1]):
            temp += x[i, t]
        out[i] = temp/np.shape(x)[1]
    return out

def multiplymat22_float(x,  y):
    s1 = np.shape(x)[0]
    s2 = np.shape(y)[1]
    out = np.zeros((s1,s2), dtype=np.float32)

    for i in range(s1):
        for t in range(s2):
            for m in range(np.shape(x)[1]):
                out[i,t] += x[i,m] * y[m,t]

    return out

def multiplymat22(x, y):
    s1 = np.shape(x)[0]
    s2 = np.shape(y)[1]
    out = np.zeros((s1,s2), dtype=np.complex64)

    for i in range(s1):
        for t in range(s2):
            for m in range(np.shape(x)[1]):
                out[i,t] += x[i,m] * y[m,t]

    return out

def transposemat2(x):
    n1 = np.shape(x)[0]
    n2 = np.shape(x)[1]
    y = np.zeros((n2, n1), dtype=np.complex64)
    for i in range(n1):
        for j in range(n2):
            y[j, i] = x[i, j]
    return y

def outer_product(x, y):
    n = np.shape(x)[0]
    out = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for t in range(n):
            out[i, t] = x[i] * y[t]
    return out

def divide_elementwise(x, y):
    n1 = np.shape(x)[0]
    n2 = np.shape(x)[1]
    out = np.zeros((n1, n2), dtype=np.complex64)
    for i in range(n1):
        for t in range(n2):
            if x[i, t] == 0:
                out[i, t] = 0
            else:
                out[i, t] = x[i, t] / y[i, t]
    return out

def cov2corr_cy(cov_matrix):
    """ Converts covariance matrix to correlation/coherence matrix. """
    n = np.shape(cov_matrix)[0]
    v = np.zeros(n, dtype=np.float32)

    for i in range(n):
        v[i] = sqrt(np.abs(cov_matrix[i, i]))

    outer_v = outer_product(v, v)
    corr_matrix = divide_elementwise(cov_matrix, outer_v)

    return corr_matrix

def est_corr_cy(ccg):
    """ Estimate Correlation matrix from an ensemble."""
    
    cov_mat = multiplymat22(ccg,  conjmat2(transposemat2(ccg)))

    for i in range(np.shape(cov_mat)[0]):
        for t in range(np.shape(cov_mat)[1]):
            cov_mat[i, t] /= np.shape(ccg)[1]
    corr_matrix = cov2corr_cy(cov_mat)

    for i in range(np.shape(cov_mat)[0]):
        for t in range(np.shape(cov_mat)[1]):
            corr_matrix[i, t] = np.nan_to_num(corr_matrix[i, t])

    return corr_matrix

def conjmat2(x):
    n1 = np.shape(x)[0]
    n2 = np.shape(x)[1]
    y = np.zeros((n1, n2), dtype=np.complex64)
    for i in range(n1):
        for t in range(n2):
            y[i, t] = np.conjugate(x[i, t])
    return y

def  get_reference_index(coh):
    n = np.shape(coh)[0]
    abscoh = np.abs(coh)
    meanval = np.zeros(n, dtype=np.float32)
    maxval = 0
    index = 0

    for i in range(n):
        for t in range(n):
            meanval[i] += abscoh[i, t]
        meanval[i] /= n 
    
    for i in range(n):
        if maxval < meanval[i]:
            maxval = meanval[i]
            index = i 

    return index

def mask_diag(coh, lag):
    n = np.shape(coh)[0]
    N = np.ones((n, n), dtype=np.int32)
    mask = np.triu(N, lag) + np.tril(N, -lag)
    masked_coh = np.zeros((n, n), dtype=np.complex64)
    for i in range(n):
        for t in range(n):
            if mask[i,t] == 0:
                masked_coh[i, t] = coh[i, t]
    return masked_coh

def regularize_matrix_cy(M):
    """ Regularizes a matrix to make it positive semi definite. """
    status = 1
    t = 0
    N = np.zeros((np.shape(M)[0], np.shape(M)[1]), dtype=np.float32)
    en = 1e-6

    for i in range(np.shape(M)[0]):
        for t in range(np.shape(M)[1]):
            N[i, t] = M[i, t]

    t = 0
    while t < 100:
        status = is_semi_pos_def_chol_cy(N)
        if status == 0:
            break
        else:
            for i in range(np.shape(M)[0]):
                N[i, i] += en
            en *= 2
            t += 1
    return status, N

def is_semi_pos_def_chol_cy(x):
    """ Checks the positive semi definitness of a matrix. desired: res=0 """
    try:
        LA.cholesky(x)
        res = 0
    except:
        res = 1
    return res

def multiply_elementwise_dc(x, y):
    out = np.zeros((np.shape(y)[0], np.shape(y)[1]), dtype=np.complex64)
    for i in range(np.shape(x)[0]):
        for t in range(np.shape(x)[1]):
            out[i, t] = y[i,t] * x[i, t]

    return out

def PTA_L_BFGS_cy(coh, abscoh):
    """ Uses L-BFGS method to optimize PTA function and estimate phase values. """
    n_image = np.shape(coh)[0]
    vec = np.zeros(n_image, dtype=np.complex64)

    x = EMI_phase_estimation_cy(coh, abscoh)
    x0 = np.angle(x)
    amp = np.abs(x)

    invabscoh = inverse_float_matrix(abscoh)
    #invabscoh = LA.inv(abscoh)
    inverse_gam = multiply_elementwise_dc(invabscoh, coh)
    res = optimize_lbfgs(x0, inverse_gam)
    for i in range(n_image):
        vec[i] = amp[i] * np.exp(1j * res[i])

    return vec

def inverse_float_matrix(x):
    
    res , uu = lap.spotrf(x, False, False)
    res  = lap.spotri(res)[0]
    res = np.triu(res) + np.triu(res, k=1).T

    return res

def optphase_cy(x0, inverse_gam):
    u = 0
    out = 1

    n = np.shape(x0)[0]
    x = np.exp(np.float32(x0))
    #for i in range(n):
    #    x[i] = x[i] * conjf(x[0])
    y = multiplymat12(np.conjugate(x), inverse_gam)
    u = multiplymat11(y, x)
    out = np.abs(log(u))
    return  out

def multiplymat12(x, y):
    s1 = np.shape(x)[0]
    s2 = np.shape(y)[1]
    out = np.zeros(s2, dtype=np.complex64)

    for i in range(s2):
        for t in range(s1):
            out[i] += x[t] * y[t,i]
    return out

def multiplymat11(x, y):
    out = 0
    for i in range(np.shape(x)[0]):
            out += x[i] * y[i]
    return out

def optimize_lbfgs(x0, inverse_gam):
    out = np.zeros(np.shape(x0)[0], dtype=np.float32)
    res = minimize(optphase_cy, x0, args=inverse_gam, method='L-BFGS-B', options={'gtol': 1e-6, 'disp': False}).x

    return out

def EMI_phase_estimation_cy(coh, abscoh):
    """ Estimates the phase values based on EMI decomosition (Homa Ansari, 2018 paper) """
    n = np.shape(coh)[0]
    vec = np.zeros(n, dtype=np.complex64)

    #invabscoh = inverse_float_matrix(abscoh)
    invabscoh = LA.inv(abscoh)
    M = multiply_elementwise_dc(invabscoh, coh)
    #eigen_value, eigen_vector = lap.cheevd(M)[0:2]
    eigen_value, eigen_vector = LA.eigh(M)[0:2]
    for i in range(n):
            vec[i] = eigen_vector[i, 0]
    return vec

def EVD_phase_estimation_cy(coh):
    """ Estimates the phase values based on eigen value decomosition """
    n = np.shape(coh)[0]
    vec = np.zeros(n, dtype=np.complex64)

    #eigen_value, eigen_vector = lap.cheevd(coh)[0:2]
    eigen_value, eigen_vector = LA.eigh(coh)[0:2]

    for i in range(n):
        vec[i] = eigen_vector[i, n-1]

    return vec

def phase_linking_process_cy(ccg_sample, stepp, method, lag):
    """Inversion of phase based on a selected method among PTA, EVD and EMI """
    
    n1 = np.shape(ccg_sample)[1]

    coh_mat = est_corr_cy(ccg_sample)
    max_coh_index = get_reference_index(coh_mat)
    
    if method.decode('utf-8') == 'SBW':
        coh_mat = mask_diag(coh_mat, lag)

    if method.decode('utf-8') == 'PTA' or method.decode('utf-8') == 'sequential_PTA':
        status, abscoh = regularize_matrix_cy(np.abs(coh_mat))
        if status == 0:
            res = PTA_L_BFGS_cy(coh_mat, abscoh)
        else:
            res = EVD_phase_estimation_cy(coh_mat)
    elif method.decode('utf-8') == 'EMI' or method.decode('utf-8') == 'sequential_EMI':
        status, abscoh = regularize_matrix_cy(np.abs(coh_mat))
        if status == 0:
          
            res = EMI_phase_estimation_cy(coh_mat, abscoh)
        else:
          
            res = EVD_phase_estimation_cy(coh_mat)
    else:
    
        res = EVD_phase_estimation_cy(coh_mat)

    index = 0
    if stepp > 0:
        if method.decode('utf-8') == 'SBW':
            index = 0
        else:
            index = stepp - 1

    xi = np.exp(-1j * np.angle(res[index]))

    for t in range(np.shape(coh_mat)[0]):
        res[t] = np.exp(1j * np.angle(res[t])) * xi
        
    quality = gam_pta_c(np.angle(coh_mat), res)

    return res, 0, quality, max_coh_index


def gam_pta_c(ph_filt, vec):
    """ Returns squeesar PTA coherence between the initial and estimated phase vectors.
    :param ph_filt: np.angle(coh) before inversion
    :param vec_refined: refined complex vector after inversion
    """

    n = np.shape(vec)[0]
    ang_vec = np.angle(vec)
    temp = 0

    for i in range(n-1):
        for k in range(i + 1, n):
            temp += np.exp(1j * (ph_filt[i,k] - (ang_vec[i] - ang_vec[k])))

    temp_coh = np.real(temp) * 2 /(n**2 - n)

    return temp_coh


def get_shp_row_col_c(data, input_slc, def_sample_rows, def_sample_cols, azimuth_window,  range_window,  reference_row,
                      reference_col, distance_threshold, shp_test):
    
    n_image = np.shape(input_slc)[0]
    ref = np.zeros(n_image, dtype=np.float32)
    test = np.zeros(n_image, dtype=np.float32)

    row_0 = data[0]
    col_0 = data[1]
    
    length = np.shape(input_slc)[1]
    width = np.shape(input_slc)[2]
    t1 = data[0] + def_sample_rows[0]
    t2 = data[0] + def_sample_rows[azimuth_window-1]
    if t2 > length:
        t2 = length
    if t1 < 0:
        t1 = 0

    s_rows = t2 - t1
    ref_row = reference_row - (t1 - (data[0] + def_sample_rows[0]))

    sample_rows = np.zeros(s_rows, dtype=np.int32)
    for i in range(s_rows):
        sample_rows[i] = i + t1

    t1 = data[1] + def_sample_cols[0]
    t2 = data[1] + def_sample_cols[range_window-1]
    if t2 > width:
        t2 = width
    if t1 < 0:
        t1 = 0

    s_cols = t2 - t1
    ref_col = reference_col - (t1 - (data[1] + def_sample_cols[0]))

    sample_cols = np.zeros(s_cols, dtype=np.int32)
    for i in range(s_cols):
        sample_cols[i] = i + t1

    for i in range(n_image):
        ref[i] = np.abs(input_slc[i, row_0, col_0])
    sorting(ref)
    distance = np.zeros((s_rows, s_cols), dtype='long')
    
    if shp_test == b'glrt':
        for t1 in range(s_rows):
            for t2 in range(s_cols):
                test = np.zeros(n_image, dtype=np.float32)
                for temp in range(n_image):
                    test[temp] = np.abs(input_slc[temp, sample_rows[t1], sample_cols[t2]])
                sorting(test)
                distance[t1, t2] = glrt_cy(ref, test, distance_threshold)

    elif shp_test == b'ks':
        for t1 in range(s_rows):
            for t2 in range(s_cols):
                test = np.zeros(n_image, dtype=np.float32)
                for temp in range(n_image):
                    test[temp] = np.abs(input_slc[temp, sample_rows[t1], sample_cols[t2]])
                sorting(test)
                distance[t1, t2] = ks2smapletest_cy(ref, test, distance_threshold)

    elif shp_test == b'ad':
        for t1 in range(s_rows):
            for t2 in range(s_cols):
                test = np.zeros(n_image, dtype=np.float32)
                for temp in range(n_image):
                    test[temp] = np.abs(input_slc[temp, sample_rows[t1], sample_cols[t2]])
                sorting(test)
                distance[t1, t2] = ADtest_cy(ref, test, distance_threshold)

    elif shp_test == b'ttest':
        for t1 in range(s_rows):
            for t2 in range(s_cols):
                test = np.zeros(n_image, dtype=np.float32)
                for temp in range(n_image):
                    test[temp] = np.abs(input_slc[temp, sample_rows[t1], sample_cols[t2]])
                sorting(test)
                distance[t1, t2] = ttest_indtest_cy(ref, test, distance_threshold)
   
    #ks_label = clabel(distance, connectivity=2)
    #ref_label = ks_label[ref_row, ref_col]
    
    #temp = count(ks_label, ref_label)
    temp = count(distance, 1)
    shps = np.zeros(temp, dtype=np.complex64)

    temp = 0
    for t1 in range(s_rows):
        for t2 in range(s_cols):
            #if ks_label[t1, t2] == ref_label:
            if distance[t1, t2] == 1:
                shps[temp] = sample_rows[t1] + 1j * sample_cols[t2]
                temp += 1
    return shps

def  ADtest_cy(S1, S2, threshold):
    testobj = anderson_ksamp([S1, S2])
    test = testobj.significance_level
    if test >= threshold:
        res = 1
    else:
        res = 0

    return res

def sorting(x):
    x.sort()
    return

def  compute_glrt_test_stat(scale_1,  scale_2):
    """Compute the GLRT test statistic."""
    scale_pooled = (scale_1 + scale_2) / 2
    return 2 * log(scale_pooled) - log(scale_1) - log(scale_2)

def glrt_alpha(S1, S2):


    scale_1 = (np.var(S1) + (np.mean(S1))**2) / 2
    scale_2 = (np.var(S2) + (np.mean(S2))**2) / 2

    return compute_glrt_test_stat(scale_1, scale_2)

def glrt_cy(S1, S2, threshold):
    alpha = glrt_alpha(S1, S2)
    if alpha <= threshold:
        res = 1
    else:
        res = 0
    return res

def  concat_cy(x, y):
    n1 = np.shape(x)[0]
    n2 = np.shape(y)[0]
    out = np.zeros((n1 + n2), dtype=np.float32)
    for i in range(n1):
        out[i] = x[i]
        out[i + n1] = y[i]
    return out

def  ecdf_distance(data1, data2):
    data_all = concat_cy(data1, data2)
    sorting(data_all)
    distance = searchsorted_max(data1, data2, data_all)

    return distance

def searchsorted_max(x1, x2, y):
    nx = np.shape(x1)[0]
    ny = np.shape(y)[0]
    outtmp = 0
    out = 0
    i = 0
    temp1 = 0
    temp2 = 0

    for i in range(ny):

        t1 = temp1
        t2 = temp2

        if y[i] >= x1[nx - 1]:
            temp1 = nx

        if y[i] >= x2[nx - 1]:
            temp2 = nx

        while t1 < nx:
            if t1 == 0 and y[i] < x1[t1]:
                temp1 = 0
                t1 = nx
            elif x1[t1 - 1] <= y[i] and y[i] < x1[t1]:
                temp1 = t1
                t1 = nx
            else:
                t1 += 1

        while t2 < nx:
            if t2 == 0 and y[i] < x2[t2]:
                temp2 = 0
                t2 = nx
            elif x2[t2 - 1] <= y[i] and y[i] < x2[t2]:
                temp2 = t2
                t2 = nx
            else:
                t2 += 1

        outtmp = abs(temp1 - temp2) / nx
        if outtmp > out:
            out = outtmp
    return out

def ks2smapletest_cy(S1, S2, threshold):
    distance = ecdf_distance(S1, S2)
    if distance <= threshold:
        res = 1
    else:
        res = 0
    return res

def  ks2smapletest_py(S1, S2, threshold):
    distance = ecdf_distance(S1, S2)
    if distance <= threshold:
        res = 1
    else:
        res = 0
    return res

def ttest_indtest_cy(S1, S2, threshold):
    testobj = ttest_ind(S1, S2, equal_var=False)
    test = testobj[1]
    
    if test >= threshold:
        res = 1
    else:
        res = 0

    return res


def count(x, value):
    n1 = np.shape(x)[0]
    n2 = np.shape(x)[1]
    out = 0
    for i in range(n1):
        for t in range(n2):
            if x[i, t] == value:
                out += 1
    return out