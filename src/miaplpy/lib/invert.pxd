cimport numpy as cnp
cimport cython


cdef void write_hdf5_block_3D(object, cnp.ndarray[float, ndim=3], bytes, list)
cdef void write_hdf5_block_2D_int(object, cnp.ndarray[int, ndim=2], bytes, list)
cdef void write_hdf5_block_2D_float(object, cnp.ndarray[float, ndim=2], bytes, list)

cdef class CPhaseLink:
    cdef object inps, slcStackObj
    cdef bytes work_dir, phase_linking_method, shp_test
    cdef bytes slc_stack, RSLCfile
    cdef int range_window, azimuth_window, patch_size, n_image, width, length, num_archived
    cdef int shp_size, mini_stack_default_size, num_box, total_num_mini_stacks
    cdef float distance_thresh
    cdef bint sequential
    cdef dict metadata
    cdef list all_date_list
    cdef float[::1] prep_baselines
    cdef int[::1] sample_rows, sample_cols
    cdef int reference_row, reference_col
    cdef float complex[:, :, ::1] patch_slc_images
    cdef int ps_shp, rglooks, azlooks
    cdef readonly list box_list
    cdef readonly bytes out_dir
    cdef readonly int time_lag
    cdef bytes mask_file




