#cython: wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3

cimport cython
import os
import numpy as np
cimport numpy as cnp
import csv
from numpy import linalg as LA
#from scipy.linalg import lapack as lap
from libc.math cimport sqrt, exp, isnan, logf
from scipy.optimize import minimize
from skimage.measure._ccomp import label_cython as clabel
from scipy.stats import anderson_ksamp, ttest_ind
from mintpy.utils import ptime
from mintpy.utils import readfile
import time
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)


cdef extern from "complex.h":    
    float complex cexpf(float complex z)
    float complex conjf(float complex z)
    float crealf(float complex z)
    float cimagf(float complex z)
    float cabsf(float complex z)
    float sqrtf(float z)
    float complex clogf(float complex z)
    float complex csqrtf(float complex z)
    float sqrtf(float z)
    float cargf(float complex z)
    float fmaxf(float x, float y)
    float fminf(float x, float y)

cdef inline bint isnanc(float complex x):
    cdef bint res = isnan(crealf(x)) or isnan(cimagf(x))
    return res

cdef inline float cargf_r(float complex z):
    cdef float res
    res = cargf(z)
    if isnan(res):
        res = 0
    return res

cdef inline double cargd_r(float complex z):
    cdef double res = cargf(z)
    if isnan(res):
        res = 0
    return res

cdef inline float[::1] absmat1(float complex[::1] x):
    cdef int i
    cdef int n = np.shape(x)[0]
    cdef float[::1] y = np.zeros(n, dtype=np.float32)

    for i in range(n):
            y[i] = cabsf(x[i])
    return y

cdef inline float[:, ::1] absmat2(float complex[:, ::1] x):
    cdef int i, j
    cdef int n1 = np.shape(x)[0]
    cdef int n2 = np.shape(x)[1]
    cdef float[:, ::1] y = np.zeros((n1, n2), dtype=np.float32)

    for i in range(n1):
        for j in range(n2):
            y[i, j] = cabsf(x[i, j])
    return y

cdef inline float[::1] angmat(float complex[::1] x):
    cdef int i
    cdef int n = np.shape(x)[0]
    cdef float[::1] y = np.zeros(n, dtype=np.float32)

    for i in range(n):
            y[i] = cargf_r(x[i])
    return y

cdef inline double[::1] angmatd(float complex[::1] x):
    cdef int i
    cdef int n = np.shape(x)[0]
    cdef double[::1] y = np.zeros(n, dtype=np.double)

    for i in range(n):
            y[i] = cargf_r(x[i])
    return y

cdef inline float[:, ::1] angmat2(float complex[:, ::1] x):
    cdef int i, t
    cdef int n1 = np.shape(x)[0]
    cdef int n2 = np.shape(x)[1]
    cdef float[:, ::1] y = np.zeros((n1, n2), dtype=np.float32)

    for i in range(n1):
        for t in range(n2):
            y[i, t] = cargf_r(x[i, t])
    return y

cdef inline float complex[::1] expmati(float[::1] x):
    cdef int i
    cdef int n = np.shape(x)[0]
    cdef float complex[::1] y = np.zeros(n, dtype=np.complex64)

    for i in range(n):
            y[i] = cexpf(1j * x[i])
    return y


cdef inline float complex[::1] conjmat1(float complex[::1] x):
    cdef int i
    cdef int n = np.shape(x)[0]
    cdef float complex[::1] y = np.zeros(n, dtype=np.complex64)

    for i in range(n):
            y[i] = conjf(x[i])
    return y

cdef inline float complex[:,::1] conjmat2(float complex[:,::1] x):
    cdef int i, t
    cdef int n1 = np.shape(x)[0]
    cdef int n2 = np.shape(x)[1]
    cdef float complex[:, ::1] y = np.zeros((n1, n2), dtype=np.complex64)
    for i in range(n1):
        for t in range(n2):
            y[i, t] = conjf(x[i, t])
    return y


cdef inline float complex[:,::1] multiply_elementwise_dc(float[:, :] x, float complex[:, ::1] y):
    cdef int i, t
    cdef float complex[:, ::1] out = np.zeros((np.shape(y)[0], np.shape(y)[1]), dtype=np.complex64)
    for i in range(np.shape(x)[0]):
        for t in range(np.shape(x)[1]):
            out[i, t] = y[i,t] * x[i, t]

    return out


cdef inline float complex multiplymat11(float complex[::1] x, float complex[::1] y):
    cdef float complex out = 0
    cdef int i
    for i in range(np.shape(x)[0]):
            out += x[i] * y[i]
    return out


cdef inline float complex[:,::1] multiplymat22(float complex[:, :] x, float complex[:, ::1] y):
    cdef int s1 = np.shape(x)[0]
    cdef int s2 = np.shape(y)[1]
    cdef float complex[:,::1] out = np.zeros((s1,s2), dtype=np.complex64)
    cdef int i, t, m

    for i in range(s1):
        for t in range(s2):
            for m in range(np.shape(x)[1]):
                out[i,t] += x[i,m] * y[m,t]

    return out


cdef inline float complex[::1] multiplymat12(float complex[::1] x, float complex[:, ::1] y):
    cdef int s1 = np.shape(x)[0]
    cdef int s2 = np.shape(y)[1]
    cdef float complex[::1] out = np.zeros(s2, dtype=np.complex64)
    cdef int i, t

    for i in range(s2):
        for t in range(s1):
            out[i] += x[t] * y[t,i]
    return out


cdef inline float complex[:, ::1] mask_diag(float complex[:, ::1] coh, int lag):
    cdef int n = np.shape(coh)[0]
    cdef int[:, ::1] N = np.ones((n, n), dtype=np.int32)
    cdef int[:, ::1] mask = np.triu(N, lag) + np.tril(N, -lag)
    cdef float complex[:, ::1] masked_coh = np.zeros((n, n), dtype=np.complex64)
    cdef int i, t
    for i in range(n):
        for t in range(n):
            if mask[i,t] == 0:
                masked_coh[i, t] = coh[i, t]
    return masked_coh


cdef inline float complex[::1] EVD_phase_estimation_cy(float complex[:, ::1] coh):
    """ Estimates the phase values based on eigen value decomosition """
    cdef float[::1] eigen_value
    cdef float complex[:, :] eigen_vector
    cdef int i, n = np.shape(coh)[0]
    cdef float complex[::1] vec = np.zeros(n, dtype=np.complex64)

    #eigen_value, eigen_vector = lap.cheevd(coh)[0:2]
    eigen_value, eigen_vector = LA.eigh(coh)[0:2]

    for i in range(n):
        vec[i] = eigen_vector[i, n-1]

    return vec


cdef inline float complex[::1] EMI_phase_estimation_cy(float complex[:, ::1] coh, float[:, ::1] abscoh):
    """ Estimates the phase values based on EMI decomosition (Homa Ansari, 2018 paper) """
    cdef float[:, :] invabscoh
    cdef int stat
    cdef int i, n = np.shape(coh)[0]
    cdef float complex[:, ::1] M
    cdef float[::1] eigen_value
    cdef float complex[:, :] eigen_vector
    cdef float complex[::1] vec = np.zeros(n, dtype=np.complex64)


    #invabscoh = inverse_float_matrix(abscoh)
    invabscoh = LA.inv(abscoh)
    M = multiply_elementwise_dc(invabscoh, coh)
    #eigen_value, eigen_vector = lap.cheevd(M)[0:2]
    eigen_value, eigen_vector = LA.eigh(M)[0:2]
    for i in range(n):
            vec[i] = eigen_vector[i, 0]

    return vec


cpdef inline double optphase_cy(double[::1] x0, float complex[:, ::1] inverse_gam):
    cdef int n
    cdef float complex[::1] x
    cdef float complex[::1] y
    cdef float complex u = 0
    cdef double out = 1
    cdef int i

    n = np.shape(x0)[0]
    x = expmati(np.float32(x0))
    #for i in range(n):
    #    x[i] = x[i] * conjf(x[0])
    y = multiplymat12(conjmat1(x), inverse_gam)
    u = multiplymat11(y, x)
    out = cabsf(clogf(u))
    return  out

cdef inline float[::1] optimize_lbfgs(double[::1] x0, float complex[:, ::1] inverse_gam):
    cdef double[::1] res
    cdef float[::1] out = np.zeros(np.shape(x0)[0], dtype=np.float32)
    cdef int i

    res = minimize(optphase_cy, x0, args=inverse_gam, method='L-BFGS-B', options={'gtol': 1e-6, 'disp': False}).x

    return out

cdef inline float complex[::1] PTA_L_BFGS_cy(float complex[:, ::1] coh, float[:, ::1] abscoh):
    """ Uses L-BFGS method to optimize PTA function and estimate phase values. """
    cdef int i, n_image = np.shape(coh)[0]
    cdef float complex[::1] x
    cdef float[::1] amp, res
    cdef double[::1] x0
    cdef float[:, :] invabscoh
    cdef int stat
    cdef float complex[:, ::1] inverse_gam
    cdef float complex[::1] vec = np.zeros(n_image, dtype=np.complex64)

    x = EMI_phase_estimation_cy(coh, abscoh)
    x0 = angmatd(x)
    amp = absmat1(x)

    #invabscoh = inverse_float_matrix(abscoh)
    invabscoh = LA.inv(abscoh)
    inverse_gam = multiply_elementwise_dc(invabscoh, coh)
    res = optimize_lbfgs(x0, inverse_gam)
    for i in range(n_image):
        vec[i] = amp[i] * cexpf(1j * res[i])

    return vec


cdef inline float[:,::1] outer_product(float[::1] x, float[::1] y):
    cdef int i, t, n = np.shape(x)[0]
    cdef float[:, ::1] out = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for t in range(n):
            out[i, t] = x[i] * y[t]
    return out


cdef inline float complex[:,::1] divide_elementwise(float complex[:, ::1] x, float[:, ::1] y):
    cdef int i, t
    cdef int n1 = np.shape(x)[0]
    cdef int n2 = np.shape(x)[1]
    cdef float complex[:, ::1] out = np.zeros((n1, n2), dtype=np.complex64)
    for i in range(n1):
        for t in range(n2):
            if x[i, t] == 0:
                out[i, t] = 0
            else:
                out[i, t] = x[i, t] / y[i, t]
    return out

cdef inline float complex[:, ::1] cov2corr_cy(float complex[:,::1] cov_matrix):
    """ Converts covariance matrix to correlation/coherence matrix. """
    cdef int i, n = np.shape(cov_matrix)[0]
    cdef float [::1] v = np.zeros(n, dtype=np.float32)
    cdef float [:, ::1] outer_v
    cdef float complex[:, ::1] corr_matrix

    for i in range(n):
        v[i] = sqrt(cabsf(cov_matrix[i, i]))

    outer_v = outer_product(v, v)
    corr_matrix = divide_elementwise(cov_matrix, outer_v)

    return corr_matrix

cdef inline float complex[:,::1] transposemat2(float complex[:, :] x):
    cdef int i, j
    cdef int n1 = np.shape(x)[0]
    cdef int n2 = np.shape(x)[1]
    cdef float complex[:, ::1] y = np.zeros((n2, n1), dtype=np.complex64)
    for i in range(n1):
        for j in range(n2):
            y[j, i] = x[i, j]
    return y

cdef inline float complex[:,::1] est_corr_cy(float complex[:,::1] ccg):
    """ Estimate Correlation matrix from an ensemble."""
    cdef int i, t
    cdef float complex[:,::1] cov_mat, corr_matrix

    cov_mat = multiplymat22(ccg,  conjmat2(transposemat2(ccg)))

    for i in range(np.shape(cov_mat)[0]):
        for t in range(np.shape(cov_mat)[1]):
            cov_mat[i, t] /= np.shape(ccg)[1]
    corr_matrix = cov2corr_cy(cov_mat)

    for i in range(np.shape(cov_mat)[0]):
        for t in range(np.shape(cov_mat)[1]):
            corr_matrix[i, t] = np.nan_to_num(corr_matrix[i, t])

    return corr_matrix

cdef inline float complex[:,::1] est_cov_cy(float complex[:,::1] ccg):
    """ Estimate Correlation matrix from an ensemble."""
    cdef int i, t
    cdef float complex[:,::1] cov_mat

    cov_mat = multiplymat22(ccg,  conjmat2(transposemat2(ccg)))

    for i in range(np.shape(cov_mat)[0]):
        for t in range(np.shape(cov_mat)[1]):
            cov_mat[i, t] /= np.shape(ccg)[1]

    return cov_mat

cpdef float complex[:,::1] est_cov_py(float complex[:,::1] ccg):
    """ Estimate Correlation matrix from an ensemble."""

    cdef int i, t
    cdef float complex[:,::1] cov_mat

    cov_mat = multiplymat22(ccg,  conjmat2(transposemat2(ccg)))

    for i in range(np.shape(cov_mat)[0]):
        for t in range(np.shape(cov_mat)[1]):
            cov_mat[i, t] /= np.shape(ccg)[1]

    return cov_mat
    
cpdef float complex[:, ::1] est_corr_py(float complex[:,::1] ccg):
    """ Estimate Correlation matrix from an ensemble."""

    cdef int i, t
    cdef float complex[:,::1] cov_mat, corr_matrix

    cov_mat = multiplymat22(ccg,  conjmat2(transposemat2(ccg)))

    for i in range(np.shape(cov_mat)[0]):
        for t in range(np.shape(cov_mat)[1]):
            cov_mat[i, t] /= np.shape(ccg)[1]
    corr_matrix = cov2corr_cy(cov_mat)

    return corr_matrix

cdef inline float sum1d(float[::1] x):
    cdef int i, n = np.shape(x)[0]
    cdef float out = 0
    for i in range(n):
        out += x[i]
    return out

cdef int get_reference_index(float complex[:, ::1] coh):
    cdef int i, index, n = np.shape(coh)[0]
    cdef float[:, ::1] abscoh = absmat2(coh)
    cdef float[::1] meanval = np.zeros(n, dtype=np.float32)
    cdef float maxval = 0
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

#cdef (float, int) test_PS_cy(float complex[:, ::1] CCG, float[::1] amplitude):
cdef (float, int) test_PS_cy(float[::1] amplitude):
    """ checks if the pixel is PS """
    
    cdef float temp_quality, amp_dispersion #, top_percentage, s
    #cdef float complex[:, ::1] coh_mat = est_corr_cy(CCG)
    #cdef int i, t, ns = np.shape(coh_mat)[0]
    #cdef float[::1] Eigen_value
    #cdef cnp.ndarray[float complex, ndim=2] Eigen_vector
    #cdef float complex[::1] vec = np.zeros(ns, dtype=np.complex64)
    #cdef float complex x0
    #cdef int index = 0
        
    # amp_diff_dispersion = np.std(amplitude_diff)/np.mean(amplitude)
    amp_dispersion = np.std(amplitude)/np.mean(amplitude)
    if amp_dispersion > 1:
        amp_dispersion = 1

    if amp_dispersion < 0.25:
        temp_quality = 1
    else:
        temp_quality = 0.01
    return temp_quality, 0

    # if amp_dispersion < 0.42:
    #     coh_mat = est_corr_cy(CCG)
    #     index = get_reference_index(coh_mat)
    #     ns = np.shape(coh_mat)[0]
    #     #Eigen_value, Eigen_vector = lap.cheevd(coh_mat)[0:2]
    #     Eigen_value, Eigen_vector = LA.eigh(coh_mat)[0:2]

    #     s = 0
    #     for i in range(ns):
    #         s += abs(Eigen_value[i])**2

    #     s = sqrt(s)
    #     top_percentage = Eigen_value[ns-1]*(100 / s)

    #     if top_percentage > 95:
    #         temp_quality = 1
    #     else:
    #         x0 = cexpf(-1j * cargf_r(Eigen_vector[0, ns - 1]))

    #         for i in range(ns):
    #             vec[i] = Eigen_vector[i, ns-1] * x0

    #         temp_quality = gam_pta_c(angmat2(coh_mat), vec)
    #         if temp_quality == 1:
    #             temp_quality = 0.95
                
    # else:
    #     temp_quality = 0.01

    # return temp_quality, index #, vec, amp_dispersion, Eigen_value[ns-1], Eigen_value[ns-2], top_percentage

cdef inline float norm_complex(float complex[::1] x):
    cdef int n = np.shape(x)[0]
    cdef int i
    cdef float out = 0
    for i in range(n):
        out += cabsf(x[i])**2
    out = sqrt(out)
    return out

cdef inline float complex compress_slcs(float complex[::1] org_vec, float complex[::1] ref_vec):
    cdef float complex out = 0
    cdef int i, n = np.shape(org_vec)[0]
    cdef int f, N = np.shape(ref_vec)[0]
    cdef float complex[::1] vm = np.zeros(n, dtype=np.complex64)
    f = N - n

    for i in range(n):
        vm[i] = cexpf(-1j * cargf_r(ref_vec[i + f]))

    normm = norm_complex(vm)

    for i in range(n):
            out += org_vec[i] * (vm[i])/normm
    return out


cdef inline float complex[::1] squeeze_images(float complex[::1] x, float complex[:, ::1] ccg, int step, int ministack_size):
    cdef int n = np.shape(x)[0]
    cdef int s = np.shape(ccg)[1]
    cdef float complex[::1] vm = np.zeros(ministack_size, dtype=np.complex64) # np.zeros(n-step, dtype=np.complex64)
    cdef int i, t
    cdef float normm = 0
    cdef float complex[::1] out = np.zeros(s, dtype=np.complex64)

    for i in range(ministack_size):
        vm[i] = cexpf(-1j * cargf_r(x[i + step]))

    normm = norm_complex(vm)

    for t in range(s):
        for i in range(ministack_size):
            out[t] += ccg[i + step, t] * (vm[i])/normm

    return out

cdef inline int is_semi_pos_def_chol_cy(float[:, ::1] x):
    """ Checks the positive semi definitness of a matrix. desired: res=0 """
    cdef int res
    try:
        LA.cholesky(x)
        res = 0
    except:
        res = 1
    return res


cdef inline tuple regularize_matrix_cy(float[:, ::1] M):
    """ Regularizes a matrix to make it positive semi definite. """
    cdef int status = 1
    cdef int i, t = 0
    cdef float[:, ::1] N = np.zeros((np.shape(M)[0], np.shape(M)[1]), dtype=np.float32)
    cdef float en = 1e-6

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

# cdef inline tuple phase_linking_process_cy(float complex[:, ::1] ccg_sample, int stepp, bytes method, bint squeez, int lag, int ministack_size):
cdef inline tuple phase_linking_process_cy(float complex[:, ::1] ccg_sample, int stepp, bytes method, int lag):
    """Inversion of phase based on a selected method among PTA, EVD and EMI """

    cdef float complex[:, ::1] coh_mat
    cdef float[:, ::1] abscoh
    cdef float complex[::1] res
    cdef int i, t, n1 = np.shape(ccg_sample)[1]
    cdef float complex[::1] squeezed
    cdef float quality, avg, maxvalue, average_row_coh
    cdef int status, max_coh_index, index
    cdef float complex xi

    coh_mat = est_corr_cy(ccg_sample)
    max_coh_index = get_reference_index(coh_mat)
    
    if method.decode('utf-8') == 'SBW':
        coh_mat = mask_diag(coh_mat, lag)

    if method.decode('utf-8') == 'PTA' or method.decode('utf-8') == 'sequential_PTA':
        status, abscoh = regularize_matrix_cy(absmat2(coh_mat))
        if status == 0:
            res = PTA_L_BFGS_cy(coh_mat, abscoh)
        else:
            res = EVD_phase_estimation_cy(coh_mat)
    elif method.decode('utf-8') == 'EMI' or method.decode('utf-8') == 'sequential_EMI':
        status, abscoh = regularize_matrix_cy(absmat2(coh_mat))
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

    xi = cexpf(-1j * cargf(res[index]))

    for t in range(np.shape(coh_mat)[0]):
        res[t] = cexpf(1j * cargf(res[t])) * xi
        
    quality = gam_pta_c(angmat2(coh_mat), res)

    return res, 0, quality, max_coh_index


cpdef tuple phase_linking_process_py(float complex[:, ::1] ccg_sample, int stepp, bytes method, bint squeez, int lag, int ministack_size):
    """Inversion of phase based on a selected method among PTA, EVD and EMI """

    cdef float complex[:, ::1] coh_mat
    cdef float[:, ::1] abscoh
    cdef float complex[::1] res
    cdef int n1 = np.shape(ccg_sample)[1]
    cdef float complex[::1] squeezed
    cdef float quality, avg, maxvalue, average_row_coh
    cdef int status, max_coh_index, index
    cdef float complex xi
    cdef int i, t

    coh_mat = est_corr_cy(ccg_sample)
    max_coh_index = 0
    maxvalue = 0.0
    for i in range(np.shape(coh_mat)[0]):
        avg = 0
        for t in range(np.shape(coh_mat)[1]):
            avg = avg + cabsf(coh_mat[i, t])
        average_row_coh = avg / np.shape(coh_mat)[1]
        if average_row_coh >= maxvalue:
            max_coh_index = i
    
    if method.decode('utf-8') == 'SBW':
        coh_mat = mask_diag(coh_mat, lag)

    if method.decode('utf-8') == 'PTA' or method.decode('utf-8') == 'sequential_PTA' or method.decode('utf-8')=='SBW':
        status, abscoh = regularize_matrix_cy(absmat2(coh_mat))
        if status == 0:
            res = PTA_L_BFGS_cy(coh_mat, abscoh)
        else:
            res = EVD_phase_estimation_cy(coh_mat)
    elif method.decode('utf-8') == 'EMI' or method.decode('utf-8') == 'sequential_EMI':
        status, abscoh = regularize_matrix_cy(absmat2(coh_mat))
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

    xi = cexpf(-1j * cargf(res[index]))

    for t in range(np.shape(coh_mat)[0]):
        res[t] = cexpf(1j * cargf(res[t])) * xi

    quality = gam_pta_c(angmat2(coh_mat), res)

    if squeez:
        squeezed = squeeze_images(res, ccg_sample, stepp, ministack_size)
        return res, squeezed, quality
    else:
        return res, 0, quality, max_coh_index


cdef inline tuple real_time_phase_linking(float complex[:,::1] compressed, float complex[:,::1] old_images,
                                          float complex[:,::1] new_images, int mini_stack_size, bytes method,
                                          bint existing_compressed):

    cdef int num_compressed, num_newim = np.shape(new_images)[0]
    cdef int num_oldim = np.shape(old_images)[0]
    cdef int noval, t, i, lineo, line, starting_line = 0
    cdef int max_coh_index, num_samples = np.shape(new_images)[1]
    cdef bint existing_compressed_n, do_squeeze =  False
    cdef float complex[:, ::1] new_stack
    cdef float complex[::1] vec_refined, squeezed_images
    cdef float complex[:, ::1] res_compressed, res_old_images
    cdef float temp_quality
    num_compressed = 0

    existing_compressed_n = False
    if existing_compressed:
        existing_compressed_n = True

    if num_oldim + num_newim == 2 * mini_stack_size:
        starting_line = num_oldim - (mini_stack_size - num_newim)

    if (num_oldim + num_newim) == 2 * mini_stack_size - 1:
        do_squeeze = True

    if existing_compressed:
        num_compressed = np.shape(compressed)[0]


    new_stack = np.zeros((num_compressed + num_oldim + num_newim - starting_line, num_samples), dtype=np.complex64)
    res_old_images = np.zeros((num_oldim + num_newim - starting_line, num_samples), dtype=np.complex64)

    line = 0
    lineo = 0
    if existing_compressed:
        for i in range(num_compressed):
            for t in range(num_samples):
                new_stack[i, t] = compressed[i, t]
            line += 1

    for i in range(starting_line, num_oldim):
        for t in range(num_samples):
            new_stack[line, t] = old_images[i, t]
            res_old_images[lineo, t] = old_images[i, t]
        line += 1
        lineo += 1

    for i in range(num_newim):
        for t in range(num_samples):
            new_stack[line, t] = new_images[i, t]
            res_old_images[lineo, t] = new_images[i, t]
        line += 1
        lineo += 1

    if do_squeeze:
        vec_refined, squeezed_images, temp_quality, max_coh_index = phase_linking_process_py(new_stack, num_compressed, b'real_time', do_squeeze, 0, mini_stack_size)
    else:
        vec_refined, noval, temp_quality, max_coh_index = phase_linking_process_py(new_stack, num_compressed, b'real_time', do_squeeze, 0, mini_stack_size)


    if do_squeeze is False:
        res_compressed = np.zeros((num_compressed, num_samples), dtype=np.complex64)
        for i in range(num_compressed):
            for t in range(num_samples):
                res_compressed[i, t] = compressed[i, t]

    if existing_compressed is True and do_squeeze is True:
        res_compressed = np.zeros((1 + num_compressed, num_samples), dtype=np.complex64)
        for t in range(num_samples):
            for i in range(num_compressed):
                res_compressed[i, t] = compressed[i, t]
            res_compressed[num_compressed, t] = squeezed_images[t]
    elif do_squeeze is True and existing_compressed is False:
        res_compressed = np.zeros((1, num_samples), dtype=np.complex64)
        for t in range(num_samples):
            res_compressed[0, t] = squeezed_images[t]
        existing_compressed_n = True

    return vec_refined, temp_quality, res_compressed, res_old_images, np.shape(res_old_images)[0], existing_compressed_n, max_coh_index


cdef sequential_phase_linking(cnp.ndarray[float complex, ndim=3] patch_slc_images, bytes out_folder, 
int default_mini_stack_size, int total_num_mini_stacks, int[:, ::1] coords, bytes phase_linking_method,
cnp.ndarray[float complex, ndim=3] rslc_ref, cnp.ndarray[float, ndim=2] tempCoh, cnp.ndarray[int, ndim=2] mask_ps, 
cnp.ndarray[float complex, ndim=3] SHP, cnp.ndarray[int, ndim=2] NUMSHP):

    
    cdef int box_length = np.shape(rslc_ref)[1]
    cdef int box_width = np.shape(rslc_ref)[2]
    cdef cnp.ndarray[int, ndim=2] reference_index = np.zeros((box_length, box_width), dtype=np.int32)
    cdef int num_points = np.shape(coords)[0]
    cdef float complex[:, ::1] CCG 
    cdef cnp.ndarray[float complex, ndim=3] squeezed_images = np.zeros((total_num_mini_stacks, box_length, box_width), dtype=np.complex64)
    cdef int row, col, num_shp, first_line, last_line, num_lines, n_image = np.shape(rslc_ref)[0]
    cdef (int, int) data
    cdef float complex[::1] vec, vec_refined = np.zeros(n_image, dtype=np.complex64)
    cdef int i, m, p, t
    cdef float complex x0
    cdef str patch
    
    
    for i in range(total_num_mini_stacks):
        prog_bar = ptime.progressBar(maxValue=num_points)

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
                    row = int(crealf(SHP[data[0], data[1], m]))
                    col = int(cimagf(SHP[data[0], data[1], m]))
                    if i > 0:
                        for p in range(i):
                            CCG[p, m] = squeezed_images[p, row, col]
                    for p in range(num_lines):
                        CCG[p + i, m] = patch_slc_images[first_line + p, row, col]
                
                vec_refined, noval, temp_quality, max_coh_index = phase_linking_process_cy(CCG, 0, phase_linking_method, 0)
                amp_refined = mean_along_axis_x(absmat2(CCG))
                squeezed_images[i, data[0], data[1]] = compress_slcs(vec, vec_refined)
                
                if i == 0:
                    reference_index[data[0], data[1]] = max_coh_index

                for m in range(first_line, last_line):
                    rslc_ref[m, data[0], data[1]] = amp_refined[m + i - first_line] * cexpf(1j * cargf(vec_refined[m + i - first_line])) 

                tempCoh[data[0], data[1]] += temp_quality/total_num_mini_stacks
            
            patch = os.path.basename(out_folder.decode('UTF-8'))
            prog_bar.update(t, every=200, suffix='{}/{} DS pixels ministack {}, {}'.format(t, num_points, i, patch))
    
    np.save(out_folder.decode('UTF-8') + '/phase_ref.npy', rslc_ref)
    np.save(out_folder.decode('UTF-8') + '/tempCoh.npy', tempCoh)
    np.save(out_folder.decode('UTF-8') + '/compressed.npy', squeezed_images)
    np.save(out_folder.decode('UTF-8') + '/reference_index.npy', reference_index)

        
    return 


cdef incremental_iterate(cnp.ndarray[float complex, ndim=3] patch_slc_images, bytes out_folder, 
int default_mini_stack_size, int first_new_image, int[:, ::1] coords, bytes phase_linking_method,
cnp.ndarray[float complex, ndim=3] rslc_ref, cnp.ndarray[float, ndim=2] tempCoh, cnp.ndarray[int, ndim=2] mask_ps, 
cnp.ndarray[float complex, ndim=3] SHP, cnp.ndarray[int, ndim=2] NUMSHP):

    cdef int box_length = np.shape(rslc_ref)[1]
    cdef int box_width = np.shape(rslc_ref)[2]
    cdef int num_points = np.shape(coords)[0]
    cdef int num_new = np.shape(rslc_ref)[0] - first_new_image
    cdef float complex[:, ::1] CCG 
    cdef bint do_compress = False
    cdef cnp.ndarray[float complex, ndim=3] new_compressed = np.zeros((1, box_length, box_width), dtype=np.complex64)
    cdef cnp.ndarray[float complex, ndim=3] squeezed_images = np.zeros((1, box_length, box_width), dtype=np.complex64)
    cdef int num_images_in_stack, num_compressed, num_old_images
    cdef int row, col, num_shp
    cdef (int, int) data
    cdef float complex[::1] vec, vec_refined 
    cdef int i, m, p, t
    cdef float complex x0


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
                    row = int(crealf(SHP[data[0], data[1], m]))
                    col = int(cimagf(SHP[data[0], data[1], m]))
                    for p in range(num_compressed):
                        CCG[p, m] = squeezed_images[p, row, col]
                    for p in range(num_old_images + 1):
                        CCG[p + num_compressed, m] = patch_slc_images[first_old_image + p, row, col]
                
                vec_refined, noval, temp_quality, max_coh_index = phase_linking_process_cy(CCG, 0, phase_linking_method, 0)
                amp_refined = mean_along_axis_x(absmat2(CCG))
                if do_compress == True:
                    new_compressed[num_compressed + 1, data[0], data[1]] = compress_slcs(vec, vec_refined)

                if num_compressed == 0:
                    x0 = cexpf(-1j * cargf(vec_refined[0]))
                else:
                    x0 = cexpf(-1j * cargf(vec_refined[num_compressed]))

                rslc_ref[i, data[0], data[1]] = amp_refined[num_images_in_stack - 1] * cexpf(1j * cargf(vec_refined[num_images_in_stack - 1])) * x0
                tempCoh[data[0], data[1]] = (tempCoh[data[0], data[1]] + temp_quality)/2
    

            if do_compress == True:
                squeezed_images = np.zeros((num_compressed + 1, box_length, box_width), dtype=np.complex64)
                squeezed_images[:, :, :] = new_compressed[:, :, :]
                num_compressed += 1

            first_old_image += 1
            num_old_images = i - first_old_image
        
        prog_bar.update(i, every=1, suffix='image {}/{} incremental {} pixels patch {}'.format(i, np.shape(rslc_ref)[0], num_points, os.path.basename(out_folder.decode('UTF-8'))))

    np.save(out_folder.decode('UTF-8') + '/phase_ref.npy', rslc_ref)
    np.save(out_folder.decode('UTF-8') + '/tempCoh.npy', tempCoh)
    np.save(out_folder.decode('UTF-8') + '/compressed.npy', squeezed_images)
     
    return


cdef inline tuple real_time_phase_linking_iterate(float complex[:,::1] full_stack_complex_samples,
                                                  int mini_stack_default_size, bytes method, int num_archived, 
                                                  int[:,::1] ps_mask):
    """ phase linking of each pixel in real time adding each acquired image to the stack """

    cdef float complex value0 = 1+0j
    cdef float complex [:, ::1] old_images, compressed, new_image, compressed_arch, archive_stack
    cdef float complex [::1] vec_refined, ph_new 
    cdef list ph_tmp = []
    cdef int num_samples = np.shape(full_stack_complex_samples)[1]
    cdef int max_coh_index, max_coh_index_first, t, i, n_ph, num_oldim, num_img = np.shape(full_stack_complex_samples)[0]
    cdef int num_archived_stacks, num_new_images, first_new_image, first_old_image, last_old_image, num_old_images
    cdef bint existing_compressed = False
    cdef float temp_quality
    cdef float amp_refined
    cdef float[::1] amp_arch
    max_coh_index_first = 0


    compressed = np.zeros((1, num_samples), dtype=np.complex64)
    first_old_image = 0
    
    if num_archived > 0:
        num_archived_stacks = num_archived // mini_stack_default_size
        first_new_image = num_archived_stacks * mini_stack_default_size
        if num_archived_stacks > 1:
            first_old_image = (num_archived_stacks - 1) * mini_stack_default_size
            last_old_image = first_old_image + mini_stack_default_size
        else:
            first_old_image = 0
            last_old_image = num_archived
        num_old_images = mini_stack_default_size
        archive_stack = np.zeros((num_archived, num_samples), dtype=np.complex64)
        for t in range(num_samples):
            for i in range(num_archived):
                archive_stack[i, t] = full_stack_complex_samples[i, t]

        vec_refined, compressed_arch, temp_quality, max_coh_index = sequential_phase_linking_cy(archive_stack, b'sequential_EMI',
                                                                            mini_stack_default_size, num_archived_stacks)
        if np.shape(compressed_arch)[0] == 1:
            max_coh_index_first = max_coh_index

        if num_archived_stacks > 1:
            compressed = np.zeros((num_archived_stacks-1, num_samples), dtype=np.complex64)
            for t in range(num_archived_stacks - 1):
                for i in range(num_samples):
                    compressed[t, i] = compressed_arch[t, i]
            existing_compressed = True

        amp_arch = mean_along_axis_x(absmat2(archive_stack))
        for t in range(1, num_archived):
            vec_refined[t] = amp_arch[t] * cexpf(1j * cargf(vec_refined[t]))

        for i in range(num_archived):
            ph_tmp.append(vec_refined[i])


    else:
        first_new_image = 1
        num_old_images = 1
        ph_tmp.append(value0)



    old_images = np.zeros((num_old_images, num_samples), dtype=np.complex64)
    for t in range(num_samples):
        for i in range(num_old_images):
            old_images[i, t] = full_stack_complex_samples[i + first_old_image, t]
    
    vec_refined = np.zeros(num_img, dtype=np.complex64)
    for i in range(first_new_image):
            vec_refined[i] = ph_tmp[i]

    for t in range(first_new_image, num_img):
        new_image = np.zeros((1, num_samples), dtype=np.complex64)
        for i in range(num_samples):
            new_image[0, i] = full_stack_complex_samples[t, i]

        ph_new, temp_quality, compressed, old_images, num_oldim, existing_compressed, max_coh_index = real_time_phase_linking(compressed, old_images,
                                                                                          new_image, mini_stack_default_size,
                                                                                          method, existing_compressed)

        if first_new_image==1 and t-first_new_image == mini_stack_default_size:
            max_coh_index_first = max_coh_index

        n_ph = np.shape(ph_new)[0]

        amp_refined = mean_along_axis_x(absmat2(new_image))[0]
        ph_new[n_ph-1] = amp_refined * cexpf(1j * cargf(ph_new[n_ph-1]))

        vec_refined[t] = ph_new[n_ph - 1]


    return vec_refined, compressed, temp_quality, max_coh_index_first


cdef inline tuple sequential_phase_linking_cy(float complex[:,::1] full_stack_complex_samples,
                                        bytes method, int mini_stack_default_size,
                                        int total_num_mini_stacks):
    """ phase linking of each pixel sequentially and applying a datum shift at the end """

    cdef int i, t, sstep, first_line, last_line, num_lines
    cdef int max_coh_index, max_coh_index_first, a1, a2, ns = np.shape(full_stack_complex_samples)[1]
    cdef int n_image = np.shape(full_stack_complex_samples)[0]
    cdef float complex[::1] vec_refined = np.zeros((n_image), dtype=np.complex64)
    cdef float complex[:, ::1] mini_stack_complex_samples
    cdef float complex[::1] res, squeezed_images_0
    cdef float temp_quality, quality
    cdef float complex[:, ::1] squeezed_images = np.zeros((total_num_mini_stacks,
                                                           np.shape(full_stack_complex_samples)[1]),
                                                           dtype=np.complex64)

    quality = 0
    temp_quality = 0
    max_coh_index_first = 0

    for sstep in range(total_num_mini_stacks):

        first_line = sstep * mini_stack_default_size
        if sstep == total_num_mini_stacks - 1:
            last_line = n_image
        else:
            last_line = first_line + mini_stack_default_size
        num_lines = last_line - first_line

        if sstep == 0:

            mini_stack_complex_samples = np.zeros((num_lines, ns), dtype=np.complex64)

            for i in range(ns):
                for t in range(num_lines):
                    mini_stack_complex_samples[t, i] = full_stack_complex_samples[t, i]

            res, squeezed_images_0, temp_quality, max_coh_index_first = phase_linking_process_py(mini_stack_complex_samples, sstep, method, True, 0, mini_stack_default_size)
        else:

            mini_stack_complex_samples = np.zeros((sstep + num_lines, ns), dtype=np.complex64)

            for i in range(ns):
                for t in range(sstep):
                    mini_stack_complex_samples[t, i] = squeezed_images[t, i]
                for t in range(num_lines):
                    mini_stack_complex_samples[t + sstep, i] = full_stack_complex_samples[first_line + t, i]

            res, squeezed_images_0, temp_quality, max_coh_index = phase_linking_process_py(mini_stack_complex_samples, sstep, method, True, 0, mini_stack_default_size)

        quality += temp_quality

        for i in range(num_lines):
            vec_refined[first_line + i] = res[sstep + i]

        for i in range(np.shape(squeezed_images_0)[0]):
                squeezed_images[sstep, i] = squeezed_images_0[i]

    quality /= total_num_mini_stacks

    return vec_refined, squeezed_images, quality, max_coh_index_first


cpdef tuple sequential_phase_linking_py(float complex[:,::1] full_stack_complex_samples,
                                        bytes method, int mini_stack_default_size,
                                        int total_num_mini_stacks):
    """ phase linking of each pixel sequentially and applying a datum shift at the end """

    cdef int i, t, sstep, first_line, last_line, num_lines
    cdef int max_coh_index, max_coh_index_first, a1, a2, ns = np.shape(full_stack_complex_samples)[1]
    cdef int n_image = np.shape(full_stack_complex_samples)[0]
    cdef float complex[::1] vec_refined = np.zeros((n_image), dtype=np.complex64)
    cdef float complex[:, ::1] mini_stack_complex_samples
    cdef float complex[::1] res, squeezed_images_0
    cdef float  temp_quality, quality
    cdef float complex[:, ::1] squeezed_images = np.zeros((total_num_mini_stacks,
                                                           np.shape(full_stack_complex_samples)[1]),
                                                           dtype=np.complex64)

    quality = 0
    temp_quality = 0
    max_coh_index_first = 0

    for sstep in range(total_num_mini_stacks):

        first_line = sstep * mini_stack_default_size
        if sstep == total_num_mini_stacks - 1:
            last_line = n_image
        else:
            last_line = first_line + mini_stack_default_size
        num_lines = last_line - first_line

        if sstep == 0:

            mini_stack_complex_samples = np.zeros((num_lines, ns), dtype=np.complex64)

            for i in range(ns):
                for t in range(num_lines):
                    mini_stack_complex_samples[t, i] = full_stack_complex_samples[t, i]

            res, squeezed_images_0, temp_quality, max_coh_index = phase_linking_process_py(mini_stack_complex_samples, sstep, method, True, 0, mini_stack_default_size)
        else:

            mini_stack_complex_samples = np.zeros((sstep + num_lines, ns), dtype=np.complex64)

            for i in range(ns):
                for t in range(sstep):
                    mini_stack_complex_samples[t, i] = squeezed_images[t, i]
                for t in range(num_lines):
                    mini_stack_complex_samples[t + sstep, i] = full_stack_complex_samples[first_line + t, i]

            res, squeezed_images_0, temp_quality, max_coh_index = phase_linking_process_py(mini_stack_complex_samples, sstep, method, True, 0, mini_stack_default_size)

        quality += temp_quality

        for i in range(num_lines):
            vec_refined[first_line + i] = res[sstep + i]

        for i in range(np.shape(squeezed_images_0)[0]):
                squeezed_images[sstep, i] = squeezed_images_0[i]

        if sstep == 0:
            max_coh_index_first = max_coh_index

    quality /= total_num_mini_stacks

    return vec_refined, squeezed_images, quality, max_coh_index_first



cdef inline float complex[::1] datum_connect_cy(float complex[:, ::1] squeezed_images, float complex[::1] vector_refined, int mini_stack_size):
    """

    Parameters
    ----------
    squeezed_images: a 2D matrix in format of squeezed_images * num_of_samples
    vector_refined: n*1 refined complex vector

    Returns
    -------

    """
    cdef float complex[::1] datum_shift
    cdef float complex[::1] new_vector_refined
    cdef int step, i, first_line, last_line
    cdef bytes method = b'EMI'

    datum_shift =phase_linking_process_py(squeezed_images, 0, method, False, 0, mini_stack_size)[0]
    new_vector_refined = np.zeros((np.shape(vector_refined)[0]), dtype=np.complex64)

    for step in range(np.shape(datum_shift)[0]):
        first_line = step * mini_stack_size
        if step == np.shape(datum_shift)[0] - 1:
            last_line = np.shape(vector_refined)[0]
        else:
            last_line = first_line + mini_stack_size

        for i in range(last_line - first_line):
            new_vector_refined[i + first_line] = vector_refined[i + first_line] * datum_shift[step]

    return new_vector_refined

cpdef float complex[::1] datum_connect_py(float complex[:, ::1] squeezed_images, float complex[::1] vector_refined, int mini_stack_size):

    """
    Parameters
    ----------
    squeezed_images: a 2D matrix in format of squeezed_images * num_of_samples
    vector_refined: n*1 refined complex vector

    Returns
    -------

    """
    cdef float[::1] datum_shift
    cdef float complex[::1] new_vector_refined
    cdef int step, i, first_line, last_line
    cdef bytes method = b'EMI'

    datum_shift = phase_linking_process_py(squeezed_images, 0, method, False, 0, mini_stack_size)[0]
    new_vector_refined = np.zeros((np.shape(vector_refined)[0]), dtype=np.complex64)

    for step in range(np.shape(datum_shift)[0]):
        first_line = step * mini_stack_size
        if step == np.shape(datum_shift)[0] - 1:
            last_line = np.shape(vector_refined)[0]
        else:
            last_line = first_line + mini_stack_size

        for i in range(last_line - first_line):
            new_vector_refined[i + first_line] = vector_refined[i + first_line] * datum_shift[step]

    return new_vector_refined

@cython.cdivision(False)
cdef float searchsorted_max(cnp.ndarray[float, ndim=1] x1, cnp.ndarray[float, ndim=1] x2, cnp.ndarray[float, ndim=1] y):
    cdef int nx = np.shape(x1)[0]
    cdef int ny = np.shape(y)[0]
    cdef float outtmp = 0
    cdef float out = 0
    cdef int t1, t2, i = 0
    cdef int temp1 = 0
    cdef int temp2 = 0

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


cdef void sorting(cnp.ndarray[float, ndim=1] x):
    x.sort()
    return


cdef float ecdf_distance(cnp.ndarray[float, ndim=1] data1, cnp.ndarray[float, ndim=1] data2):
    cdef cnp.ndarray[float, ndim=1] data_all = concat_cy(data1, data2)
    cdef float distance
    sorting(data_all)
    distance = searchsorted_max(data1, data2, data_all)

    return distance


cdef inline float ks_lut_cy(int N1, int N2, float alpha):
    cdef float N = (N1 * N2) / (N1 + N2)
    cdef float[::1] distances = np.arange(0.01, 1, 0.001, dtype=np.float32)
    cdef float value, pvalue, critical_distance = 0.1
    cdef int i, t
    for i in range(np.shape(distances)[0]):
        value = distances[i]*(sqrt(N) + 0.12 + 0.11/sqrt(N))
        pvalue = 0
        for t in range(1, 101):
            pvalue += ((-1)**(t-1))*exp(-2*(value**2)*(t**2))
        pvalue = 2 * pvalue
        if pvalue > 1:
            pvalue = 1
        if pvalue < 0:
            pvalue = 0
        if pvalue <= alpha:
            critical_distance = distances[i]
            break
    return critical_distance


cdef cnp.ndarray[float, ndim=1] concat_cy(cnp.ndarray[float, ndim=1] x, cnp.ndarray[float, ndim=1] y):
    cdef int n1 = np.shape(x)[0]
    cdef int n2 = np.shape(y)[0]
    cdef cnp.ndarray[float, ndim=1] out = np.zeros((n1 + n2), dtype=np.float32)
    cdef int i
    for i in range(n1):
        out[i] = x[i]
        out[i + n1] = y[i]
    return out


cdef int count(cnp.ndarray[long, ndim=2]  x, long value):
    cdef int n1 = np.shape(x)[0]
    cdef int n2 = np.shape(x)[1]
    cdef int i, t, out = 0
    for i in range(n1):
        for t in range(n2):
            if x[i, t] == value:
                out += 1
    return out


cdef float complex[::1] get_shp_row_col_c((int, int) data, float complex[:, :, ::1] input_slc,
                        cnp.ndarray[int, ndim=1] def_sample_rows, cnp.ndarray[int, ndim=1] def_sample_cols,
                        int azimuth_window, int range_window, int reference_row,
                        int reference_col, float distance_threshold, bytes shp_test):

    cdef int row_0, col_0, i, temp, ref_row, ref_col, t1, t2, s_rows, s_cols
    cdef long ref_label
    cdef int width, length, n_image = np.shape(input_slc)[0]
    cdef int[::1] sample_rows, sample_cols
    cdef cnp.ndarray[long, ndim=2] ks_label, distance
    cdef float complex[::1] shps
    cdef cnp.ndarray[float, ndim=1] ref = np.zeros(n_image, dtype=np.float32)
    cdef cnp.ndarray[float, ndim=1] test = np.zeros(n_image, dtype=np.float32)

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
        ref[i] = cabsf(input_slc[i, row_0, col_0])
    sorting(ref)
    distance = np.zeros((s_rows, s_cols), dtype='long')
    
    if shp_test == b'glrt':
        for t1 in range(s_rows):
            for t2 in range(s_cols):
                test = np.zeros(n_image, dtype=np.float32)
                for temp in range(n_image):
                    test[temp] = cabsf(input_slc[temp, sample_rows[t1], sample_cols[t2]])
                sorting(test)
                distance[t1, t2] = glrt_cy(ref, test, distance_threshold)

    elif shp_test == b'ks':
        for t1 in range(s_rows):
            for t2 in range(s_cols):
                test = np.zeros(n_image, dtype=np.float32)
                for temp in range(n_image):
                    test[temp] = cabsf(input_slc[temp, sample_rows[t1], sample_cols[t2]])
                sorting(test)
                distance[t1, t2] = ks2smapletest_cy(ref, test, distance_threshold)

    elif shp_test == b'ad':
        for t1 in range(s_rows):
            for t2 in range(s_cols):
                test = np.zeros(n_image, dtype=np.float32)
                for temp in range(n_image):
                    test[temp] = cabsf(input_slc[temp, sample_rows[t1], sample_cols[t2]])
                sorting(test)
                distance[t1, t2] = ADtest_cy(ref, test, distance_threshold)

    elif shp_test == b'ttest':
        for t1 in range(s_rows):
            for t2 in range(s_cols):
                test = np.zeros(n_image, dtype=np.float32)
                for temp in range(n_image):
                    test[temp] = cabsf(input_slc[temp, sample_rows[t1], sample_cols[t2]])
                sorting(test)
                distance[t1, t2] = ttest_indtest_cy(ref, test, distance_threshold)
   

    ks_label = clabel(distance, connectivity=2)
    ref_label = ks_label[ref_row, ref_col]

    temp = count(ks_label, ref_label)
    shps = np.zeros(temp, dtype=np.complex64)

    temp = 0
    for t1 in range(s_rows):
        for t2 in range(s_cols):
            if ks_label[t1, t2] == ref_label:
                shps[temp] = sample_rows[t1] + 1j * sample_cols[t2]
                temp += 1
    return shps

cdef inline float[::1] mean_along_axis_x(float[:, ::1] x):
    cdef int i, t, n = np.shape(x)[0]
    cdef float[::1] out = np.zeros(n, dtype=np.float32)
    cdef float temp = 0
    for i in range(n):
        temp = 0
        for t in range(np.shape(x)[1]):
            temp += x[i, t]
        out[i] = temp/np.shape(x)[1]
    return out


cdef inline float gam_pta_c(float[:, ::1] ph_filt, float complex[::1] vec):
    """ Returns squeesar PTA coherence between the initial and estimated phase vectors.
    :param ph_filt: np.angle(coh) before inversion
    :param vec_refined: refined complex vector after inversion
    """

    cdef int i, k, n = np.shape(vec)[0]
    cdef float[::1] ang_vec = angmat(vec)
    cdef float temp_coh 
    cdef float complex temp = 0

    for i in range(n-1):
        for k in range(i + 1, n):
            temp += cexpf(1j * (ph_filt[i,k] - (ang_vec[i] - ang_vec[k])))

    temp_coh = crealf(temp) * 2 /(n**2 - n)

    return temp_coh


cdef float complex[:, ::1] normalize_samples(float complex[:, ::1] X):
    cdef float[:, ::1] amp = absmat2(X)
    cdef float[::1] norma = np.zeros(np.shape(X)[1], dtype=np.float32)
    cdef float complex[:, ::1] y = np.zeros((np.shape(X)[0], np.shape(X)[1]), dtype=np.complex64)
    cdef int i, t, n1 = np.shape(X)[0]
    cdef int n2 = np.shape(X)[1]
    cdef double temp = 0
    for t in range(n2):
        temp = 0
        for i in range(n1):
            temp += cabsf(X[i, t]) ** 2
        norma[t] = sqrt(temp)

    for t in range(n2):
        for i in range(n1):
            y[i, t] = X[i, t]/norma[t]
    return y

cdef cnp.ndarray[int, ndim=1] get_big_box_cy(cnp.ndarray[int, ndim=1] box, int range_window, int azimuth_window, int width, int length):
    cdef cnp.ndarray[int, ndim=1] big_box = np.arange(4, dtype=np.int32)
    big_box[0] = box[0] - range_window
    big_box[1] = box[1] - azimuth_window
    big_box[2] = box[2] + range_window
    big_box[3] = box[3] + azimuth_window

    if big_box[0] < 0:
        big_box[0] = 0
    if big_box[1] < 0:
        big_box[1] = 0
    if big_box[2] > width:
        big_box[2] = width
    if big_box[3] > length:
        big_box[3] = length

    return big_box

cdef tuple time_referencing(cnp.ndarray[float complex, ndim=3] patch_slc_images, int n_image, int y, int x):
    cdef float complex[::1] vec_refined = np.zeros(n_image, dtype=np.complex64)
    cdef float[::1] amp_refined =  np.zeros(n_image, dtype=np.float32)
    cdef float complex x0

    x0 = cexpf(-1j * cargf(patch_slc_images[0, y, x]))

    for m in range(n_image):
        vec_refined[m] = cexpf(1j * cargf(patch_slc_images[m, y, x]))  * x0
        amp_refined[m] = cabsf(patch_slc_images[m, y, x])

    return vec_refined, amp_refined



def process_patch_c(cnp.ndarray[int, ndim=1] box, int range_window, int azimuth_window, int width, int length, int n_image,
                    object slcStackObj, float distance_threshold, cnp.ndarray[int, ndim=1] def_sample_rows,
                    cnp.ndarray[int, ndim=1] def_sample_cols, int reference_row, int reference_col,
                    bytes phase_linking_method, int total_num_mini_stacks, int default_mini_stack_size,
                    int ps_shp, bytes shp_test, bytes out_dir, int lag, bytes mask_file, int num_archived):

  
    cdef cnp.ndarray[float complex, ndim=3] patch_slc_images = slcStackObj.read(datasetName='slc', box=box, print_msg=False)
    cdef int first_line, last_line, num_lines, row, col
    cdef int box_width = box[2] - box[0]
    cdef int box_length = box[3] - box[1]
    cdef cnp.ndarray[int, ndim=2] reference_index = np.zeros((box_length, box_width), dtype=np.int32)
    cdef cnp.ndarray[float complex, ndim=3] rslc_ref = np.zeros((n_image, box_length, box_width), dtype=np.complex64)
    cdef cnp.ndarray[float, ndim=2] tempCoh = np.zeros((box_length, box_width), dtype=np.float32)
    cdef cnp.ndarray[int, ndim=2] mask_ps = np.zeros((box_length, box_width), dtype=np.int32)
    cdef cnp.ndarray[float complex, ndim=3] SHP = np.zeros((box_length, box_width, range_window*azimuth_window), dtype=np.complex64)
    cdef cnp.ndarray[int, ndim=2] NUMSHP = np.zeros((box_length, box_width), dtype=np.int32)
    cdef int row1 = 0 
    cdef int row2 = box_length 
    cdef int col1 = 0 
    cdef int col2 = box_width 
    cdef int[:, ::1] coords = np.zeros((box_length*box_width, 2), dtype=np.int32)
    cdef int noval, num_points, num_shp, i, t, m, p
    cdef (int, int) data
    cdef float complex[::1] shp
    cdef float complex[:, ::1] CCG, coh_mat 
    cdef cnp.ndarray[float complex, ndim=3] squeezed_images 
    cdef float complex[::1] vec, vec_refined = np.zeros(n_image, dtype=np.complex64)
    cdef float[::1] amp_refined =  np.zeros(n_image, dtype=np.float32)
    cdef bint noise = False
    cdef float temp_quality, temp_quality_full, denom
    cdef object prog_bar
    cdef bytes out_folder
    cdef int max_coh_index, index = box[4]
    cdef float time0 = time.time()
    cdef float complex x0
    cdef float mi, se, amp_disp, eigv1, eigv2
    cdef int[:, ::1] mask = np.ones((box_length, box_width), dtype=np.int32)
    
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
                coords[m, 0] = i
                coords[m, 1] = t 
                m += 1
    
    num_points = m   

    if num_points == 0:
        np.save(out_folder.decode('UTF-8') + '/num_shp.npy', NUMSHP)
        np.save(out_folder.decode('UTF-8') + '/shp.npy', SHP)
        np.save(out_folder.decode('UTF-8') + '/tempCoh.npy', tempCoh)
        np.save(out_folder.decode('UTF-8') + '/mask_ps.npy', mask_ps)
        np.save(out_folder.decode('UTF-8') + '/phase_ref.npy', rslc_ref)
        np.save(out_folder.decode('UTF-8') + '/reference_index.npy', reference_index)
        np.save(out_folder.decode('UTF-8') + '/flag.npy', [1])
        return
    
    print('\nFinding PS pixels PATCH {}'.format(index))
    prog_bar = ptime.progressBar(maxValue=num_points)
    
    if not os.path.exists(out_folder.decode('UTF-8') + '/shp.npy'):
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

                # CCG = np.zeros((n_image, num_shp), dtype=np.complex64)
                # for t in range(num_shp):
                #     row = int(crealf(SHP[data[0], data[1], m]))
                #     col = int(cimagf(SHP[data[0], data[1], m]))
                #     for m in range(n_image):
                #         CCG[m, t] = patch_slc_images[m, row, col]
                

                #temp_quality, reference_index[data[0], data[1]] = test_PS_cy(CCG, amp_refined)
                temp_quality, reference_index[data[0], data[1]] = test_PS_cy(amp_refined)

                if temp_quality == 1:
                    mask_ps[data[0], data[1]] = 1

                
                x0 = cexpf(-1j * cargf(vec_refined[0]))
                for m in range(n_image):
                    rslc_ref[m, data[0], data[1]] = amp_refined[m] * cexpf(1j * cargf(vec_refined[m])) * x0

                if temp_quality < 0:
                    temp_quality = 0
                
                tempCoh[data[0], data[1]] = temp_quality         # Average temporal coherence from mini stacks
            
            prog_bar.update(i, every=200, suffix='{}/{} PS pixels, {}'.format(i, num_points, os.path.basename(out_folder.decode('UTF-8'))))
        

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
                x0 = cexpf(-1j * cargf(vec_refined[0]))
                for m in range(n_image):
                    rslc_ref[m, data[0], data[1]] = amp_refined[m] * cexpf(1j * cargf(vec_refined[m])) * x0


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
                row = int(crealf(SHP[data[0], data[1], m]))
                col = int(cimagf(SHP[data[0], data[1], m]))
                for p in range(n_image):
                    CCG[p, m] = patch_slc_images[p, row, col]

            vec_refined, noval, temp_quality, reference_index[data[0], data[1]] = phase_linking_process_cy(CCG, 0, phase_linking_method, lag)
            amp_refined = mean_along_axis_x(absmat2(CCG))

            
            for m in range(n_image):
                rslc_ref[m, data[0], data[1]] = amp_refined[m] * cexpf(1j * cargf(vec_refined[m])) #* x0
            
            tempCoh[data[0], data[1]] = temp_quality

            prog_bar.update(i, every=200, suffix='{}/{} DS pixels, patch {}'.format(i, num_points, index))
            
        np.save(out_folder.decode('UTF-8') + '/phase_ref.npy', rslc_ref)
        np.save(out_folder.decode('UTF-8') + '/tempCoh.npy', tempCoh)
        np.save(out_folder.decode('UTF-8') + '/reference_index.npy', reference_index)

    np.save(out_folder.decode('UTF-8') + '/flag.npy', [1])


    mi, se = divmod(time.time()-time0, 60)
    print('    Phase inversion of PATCH_{:04.0f} is Completed in {:02.0f} mins {:02.0f} secs\n'.format(index, mi, se))

    return

cdef float read_cutoff_csv_glrt(int N, float Alpha):
    # Replace with the actual filename
    cdef str filename = os.path.dirname(__file__) + f"/glrt_cutoffs.csv"
    cdef object file, reader
    cdef dict row
    cdef int n
    cdef float alpha, cutoff

    print(__file__)

    with open(filename, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            n = int(row["N"])
            alpha = np.float32(row["alpha"])
            if (n, alpha) == (N, Alpha):
                cutoff = np.float32(row["cutoff"])
                break
    return cutoff

cdef float compute_glrt_test_stat(float scale_1, float scale_2):
    """Compute the GLRT test statistic."""
    cdef float scale_pooled = (scale_1 + scale_2) / 2
    return 2 * logf(scale_pooled) - logf(scale_1) - logf(scale_2)


cdef float glrt_alpha(cnp.ndarray[float, ndim=1] S1, cnp.ndarray[float, ndim=1] S2):

    cdef float scale1, scale2

    scale_1 = (np.var(S1) + (np.mean(S1))**2) / 2
    scale_2 = (np.var(S2) + (np.mean(S2))**2) / 2

    return compute_glrt_test_stat(scale_1, scale_2)

cdef int glrt_cy(cnp.ndarray[float, ndim=1] S1, cnp.ndarray[float, ndim=1] S2, float threshold):
    cdef int res
    cdef float alpha = glrt_alpha(S1, S2)
    if alpha <= threshold:
        res = 1
    else:
        res = 0
    return res


cdef int ks2smapletest_cy(cnp.ndarray[float, ndim=1] S1, cnp.ndarray[float, ndim=1] S2, float threshold):
    cdef int res
    cdef float distance = ecdf_distance(S1, S2)
    if distance <= threshold:
        res = 1
    else:
        res = 0
    return res

cpdef int ks2smapletest_py(cnp.ndarray[float, ndim=1] S1, cnp.ndarray[float, ndim=1] S2, float threshold):
    cdef int res
    cdef float distance = ecdf_distance(S1, S2)
    if distance <= threshold:
        res = 1
    else:
        res = 0
    return res

cdef int ttest_indtest_cy(cnp.ndarray[float, ndim=1] S1, cnp.ndarray[float, ndim=1] S2, float threshold):
    cdef object testobj = ttest_ind(S1, S2, equal_var=False)
    cdef float test = testobj[1]
    cdef int res
    if test >= threshold:
        res = 1
    else:
        res = 0

    return res


cdef int ADtest_cy(cnp.ndarray[float, ndim=1] S1, cnp.ndarray[float, ndim=1] S2, float threshold):
    cdef object testobj = anderson_ksamp([S1, S2])
    cdef float test = testobj.significance_level
    cdef int res
    if test >= threshold:
        res = 1
    else:
        res = 0

    return res

cdef inline float[:,::1] transposemat(float[:, ::1] x):
    cdef int i, j
    cdef int n1 = np.shape(x)[0]
    cdef int n2 = np.shape(x)[1]
    cdef float[:, ::1] y = np.zeros((n2, n1), dtype=np.float32)
    for i in range(n1):
        for j in range(n2):
            y[j, i] = x[i, j]
    return y

cdef inline float[:, ::1] concatmat(float[:, :] x, float[:, :] y):
    cdef int i, j
    cdef int n1 = np.shape(x)[0]
    cdef int n2 = np.shape(x)[1]
    cdef float[:, ::1] out = np.zeros((2 * n1, n2), dtype=np.float32)
    for i in range(n2):
        for j in range(n1):
            out[j, i] = x[j, i]
            out[j + n1, i] = y[j, i]

    return out

cdef inline float[::1] concatmat11(float[:] x, float[::1] y):
    cdef int i, j
    cdef int n1 = np.shape(x)[0]
    cdef float[::1] out = np.zeros(2 * n1, dtype=np.float32)
    for i in range(n1):
        out[i] = x[i]
        out[i + n1] = y[i]
    return out

cdef inline float[:,::1] multiplymat22_float(float[:, ::1] x, float[:, ::1] y):
    cdef int s1 = np.shape(x)[0]
    cdef int s2 = np.shape(y)[1]
    cdef float[:,::1] out = np.zeros((s1,s2), dtype=np.float32)
    cdef int i, t, m

    for i in range(s1):
        for t in range(s2):
            for m in range(np.shape(x)[1]):
                out[i,t] += x[i,m] * y[m,t]

    return out

cdef inline float[::1] multiplymat21_float(float[:, ::1] x, float[::1] y):
    cdef int s1 = np.shape(x)[0]
    cdef int s2 = np.shape(x)[1]
    cdef float[::1] out = np.zeros(s1, dtype=np.float32)
    cdef int i, t, m

    for i in range(s1):
        for m in range(s2):
            out[i] += x[i,m] * y[m]
    return out

cdef inline float[::1] multiplymat21_fa(float[:, :] x, float[::1] y):
    cdef int s1 = np.shape(x)[0]
    cdef int s2 = np.shape(x)[1]
    cdef float[::1] out = np.zeros(s1, dtype=np.float32)
    cdef int i, t, m

    for i in range(s1):
        for m in range(s2):
            out[i] += x[i,m] * y[m]
    return out

cdef inline float[:, ::1] inv_diag_mat(float[::1] x):
    cdef int i, ns = np.shape(x)[0]
    cdef float[:, ::1] out = np.zeros((ns, ns), dtype=np.float32)
    cdef float maxval
    for i in range(ns):
        out[i, i] = 1/x[i]
    maxval = np.max(out)
    for i in range(ns):
        out[i, i] = out[i, i] / maxval
    return out

cdef inline float[::1] calc_residuals(float[::1] ifg, float[:,::1] G, float[::1] X):
    cdef int i, s1 = np.shape(ifg)[0]
    cdef float[::1] ifgp, res = np.zeros(s1, dtype=np.float32)
    ifgp = multiplymat21_float(G, X)
    for i in range(s1):
        res[i] = abs(ifg[i] - ifgp[i])
        if res[i] < 1e-5:
            res[i] = 1e-5
    return res

cdef inline float calculate_diff_res_max(float[::1] x, float[::1] y):
    cdef int i, s1 = np.shape(y)[0]
    cdef float[::1] res = np.zeros(s1, dtype=np.float32)
    cdef float out
    for i in range(s1):
       res[i] = abs(x[i] - y[i])
    out = max(res)
    return out

cpdef tuple invert_L1_norm_c(cnp.ndarray[float, ndim=2] R, cnp.ndarray[float, ndim=2] Alpha,
                             cnp.ndarray[float, ndim=1] y, int max_iter, float smooth_factor):
    cdef int i, ii, s1 = np.shape(y)[0], s2=2*s1, s3 = np.shape(R)[1]
    cdef float[::1] ifg, X, ifg0=np.zeros(s1, dtype=np.float32)
    cdef float[:, ::1] G, W
    cdef float e1
    if smooth_factor > 0:
        ifg = concatmat11(y, ifg0)
        G = concatmat(R, Alpha)
        W = np.eye(s2, dtype=np.float32)
    else:
        ifg = np.zeros(s1, dtype=np.float32)
        G = np.zeros((s1, s3), dtype=np.float32)
        for i in range(s1):
            ifg[i] = y[i]
            for ii in range(s3):
                G[i, ii] = R[i, ii]
        W = np.eye(s1, dtype=np.float32)
    X, e1 = iterate_L1_norm(ifg, G, W, max_iter)
    return X, e1

cdef inline tuple iterate_L1_norm(float[::1] ifg, float[:,::1] G, float[:,::1] W, int max_iter):
    cdef int i, ii, s2=np.shape(ifg)[0]
    cdef float[::1] Coef, X, res
    cdef float[::1] res1 = np.ones(s2, dtype=np.float32)
    cdef float[:, ::1] W1
    cdef float [:, :] inv_Q
    cdef float e1, diff_res

    inv_Q = LA.pinv2(multiplymat22_float(multiplymat22_float(transposemat(G), W), G ))
    Coef = multiplymat21_float(multiplymat22_float(transposemat(G), W), ifg)
    X = multiplymat21_fa(inv_Q, Coef)
    res = calc_residuals(ifg, G, X)
    diff_res = calculate_diff_res_max(res, res1)
    for ii in range(max_iter):
        if diff_res <= 0.5:
            break
        W1 = inv_diag_mat(res)
        inv_Q = LA.pinv2(multiplymat22_float(multiplymat22_float(transposemat(G), W1), G ))
        Coef = multiplymat21_float(multiplymat22_float(transposemat(G), W1), ifg)
        X = multiplymat21_fa(inv_Q, Coef)
        res1 = calc_residuals(ifg, G, X)
        diff_res = calculate_diff_res_max(res, res1)
        for i in range(s2):
            res[i] = res1[i]
    e1 = np.sum(res)
    return X, e1