# Author: Tom Furnival
# License: GPLv3

# cython: language_level=3
# distutils: language = c++

import numpy as np
cimport numpy as np
cimport cython
from libcpp cimport bool
from libc.stdint cimport uint16_t, uint32_t, uint64_t, int64_t

np.import_array()

cdef extern from "numpy/arrayobject.h":
    void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)
    void PyArray_CLEARFLAGS(np.ndarray arr, int flags)


cdef extern from "<armadillo>" namespace "arma" nogil:
    ctypedef int uword

    cdef cppclass Cube[T]:
        const uword n_rows
        const uword n_cols
        const uword n_slices
        const uword n_elem

        Cube() nogil

        Cube(uword n_rows, uword n_cols, uword n_slices) nogil

        Cube(T* aux_mem,
            uword n_rows,
            uword n_cols,
            uword n_slices,
            bool copy_aux_mem,
            bool strict) nogil

        T *memptr() nogil


cdef extern from "../src/utils.hpp":
    void SetMemState[T](T& m, int state)
    size_t GetMemState[T](T& m)
    float* GetMemory(Cube[float]& m)
    double* GetMemory(Cube[double]& m)


cdef Cube[uint16_t] numpy_to_cube_u16(np.ndarray[uint16_t, ndim=3] X):
    return Cube[uint16_t](<uint16_t*> X.data, X.shape[0], X.shape[1], X.shape[2], False, False)


cdef Cube[uint32_t] numpy_to_cube_u32(np.ndarray[uint32_t, ndim=3] X):
    return Cube[uint32_t](<uint32_t*> X.data, X.shape[0], X.shape[1], X.shape[2], False, False)


cdef Cube[uint64_t] numpy_to_cube_u64(np.ndarray[uint64_t, ndim=3] X):
    return Cube[uint64_t](<uint64_t*> X.data, X.shape[0], X.shape[1], X.shape[2], False, False)


cdef Cube[float] numpy_to_cube_f(np.ndarray[float, ndim=3] X):
    return Cube[float](<float*> X.data, X.shape[0], X.shape[1], X.shape[2], False, False)


cdef Cube[double] numpy_to_cube_d(np.ndarray[double, ndim=3] X):
    return Cube[double](<double*> X.data, X.shape[0], X.shape[1], X.shape[2], False, False)


cdef np.ndarray[np.float_t, ndim=3] numpy_from_cube_f(Cube[float] &m) except +:
    cdef np.npy_intp dims[3]
    dims[0] = <np.npy_intp> m.n_slices
    dims[1] = <np.npy_intp> m.n_cols
    dims[2] = <np.npy_intp> m.n_rows
    cdef np.ndarray[np.float_t, ndim=3] arr = np.PyArray_SimpleNewFromData(3, &dims[0], np.NPY_FLOAT, GetMemory(m))

    if GetMemState[Cube[float]](m) == 0:
        SetMemState[Cube[float]](m, 1)
        PyArray_ENABLEFLAGS(arr, np.NPY_OWNDATA)

    return arr


cdef np.ndarray[np.double_t, ndim=3] numpy_from_cube_d(Cube[double] &m) except +:
    cdef np.npy_intp dims[3]
    dims[0] = <np.npy_intp> m.n_slices
    dims[1] = <np.npy_intp> m.n_cols
    dims[2] = <np.npy_intp> m.n_rows
    cdef np.ndarray[np.double_t, ndim=3] arr = np.PyArray_SimpleNewFromData(3, &dims[0], np.NPY_DOUBLE, GetMemory(m))

    if GetMemState[Cube[double]](m) == 0:
        SetMemState[Cube[double]](m, 1)
        PyArray_ENABLEFLAGS(arr, np.NPY_OWNDATA)

    return arr


cdef extern from "../src/pguresvt.hpp":
    cdef uint32_t c_pgure "PGURESVT"[T1, T2] (Cube[T2] &, Cube[T1] &,
                                              uint32_t, uint32_t, uint32_t, uint32_t,
                                              int64_t, uint32_t, uint32_t,
                                              int64_t, int64_t, bool, bool, bool,
                                              double, double, double, double, double)


def pguresvt_16(np.ndarray[np.uint16_t, ndim=3] input_images,
                uint32_t trajectory_length = 15,
                uint32_t patch_size = 4,
                uint32_t patch_overlap = 1,
                uint32_t motion_window = 7,
                int64_t motion_filter = 5,
                uint32_t noise_method = 4,
                uint32_t max_iter = 500,
                int64_t n_jobs = -1,
                int64_t random_seed = -1,
                bool optimize_pgure = True,
                bool exponential_weighting = True,
                bool motion_estimation = True,
                double lambda1 = 0.0,
                double noise_alpha = -1.0,
                double noise_mu = -1.0,
                double noise_sigma = -1.0,
                double tol = 1e-7):

    cdef uint32_t result

    cdef np.ndarray[np.double_t, ndim=3] X
    cdef Cube[double] _X
    _X = Cube[double]()

    result = c_pgure[uint16_t, double](
        _X,
        numpy_to_cube_u16(input_images),
        trajectory_length,
        patch_size,
        patch_overlap,
        motion_window,
        motion_filter,
        noise_method,
        max_iter,
        n_jobs,
        random_seed,
        optimize_pgure,
        exponential_weighting,
        motion_estimation,
        lambda1,
        noise_alpha,
        noise_mu,
        noise_sigma,
        tol,
    )

    X = numpy_from_cube_d(_X)

    return X, result