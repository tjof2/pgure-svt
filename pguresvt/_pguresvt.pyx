# Copyright 2015-2020 Tom Furnival
#
# This file is part of PGURE-SVT.
#
# PGURE-SVT is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PGURE-SVT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PGURE-SVT.  If not, see <http://www.gnu.org/licenses/>.

# cython: language_level=3
# distutils: language = c++

import numpy as np
cimport numpy as np
cimport cython
from libcpp cimport bool
from libc.stdint cimport uint16_t, uint32_t, uint64_t

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


cdef Cube[uint16_t] numpy_to_mat_u16(np.ndarray[uint16_t, ndim=3] X):
    if not X.flags.f_contiguous:
        X = X.copy(order="F")
    return Cube[uint16_t](<uint16_t*> X.data, X.shape[0], X.shape[1], X.shape[2], False, False)


cdef Cube[uint32_t] numpy_to_mat_u32(np.ndarray[uint32_t, ndim=3] X):
    if not X.flags.f_contiguous:
        X = X.copy(order="F")
    return Cube[uint32_t](<uint32_t*> X.data, X.shape[0], X.shape[1], X.shape[2], False, False)


cdef Cube[uint64_t] numpy_to_mat_u64(np.ndarray[uint64_t, ndim=3] X):
    if not X.flags.f_contiguous:
        X = X.copy(order="F")
    return Cube[uint64_t](<uint64_t*> X.data, X.shape[0], X.shape[1], X.shape[2], False, False)


cdef Cube[float] numpy_to_mat_f(np.ndarray[float, ndim=3] X):
    if not X.flags.f_contiguous:
        X = X.copy(order="F")
    return Cube[float](<float*> X.data, X.shape[0], X.shape[1], X.shape[2], False, False)


cdef Cube[double] numpy_to_mat_d(np.ndarray[double, ndim=3] X):
    if not X.flags.f_contiguous:
        X = X.copy(order="F")
    return Cube[double](<double*> X.data, X.shape[0], X.shape[1], X.shape[2], False, False)


cdef np.ndarray[np.float_t, ndim=3] numpy_from_cube_f(Cube[float] &m) except +:
    cdef np.npy_intp dims[3]
    dims[0] = <np.npy_intp> m.n_slices
    dims[1] = <np.npy_intp> m.n_cols
    dims[2] = <np.npy_intp> m.n_rows
    cdef np.ndarray[np.float_t, ndim=3] arr = np.PyArray_SimpleNewFromData(3, &dims[0], np.NPY_FLOAT, GetMemory(m))

    if GetMemState[Cube[float]](m) == 0:
        SetMemStateCube[Cube[float]](m, 1)
        PyArray_ENABLEFLAGS(arr, np.NPY_OWNDATA)

    return arr


cdef np.ndarray[np.double_t, ndim=3] numpy_from_cube_d(Cube[double] &m) except +:
    cdef np.npy_intp dims[3]
    dims[0] = <np.npy_intp> m.n_slices
    dims[1] = <np.npy_intp> m.n_cols
    dims[2] = <np.npy_intp> m.n_rows
    cdef np.ndarray[np.double_t, ndim=3] arr = np.PyArray_SimpleNewFromData(3, &dims[0], np.NPY_DOUBLE, GetMemory(m))

    if GetMemState[Cube[double]](m) == 0:
        SetMemStateCube[Cube[double]](m, 1)
        PyArray_ENABLEFLAGS(arr, np.NPY_OWNDATA)

    return arr


