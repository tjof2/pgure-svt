# PGURE-SVT Denoising
#
#	Author:	Tom Furnival
#	Email:	tjof2@cam.ac.uk
#
#	Copyright (C) 2015-16 Tom Furnival
#
#	This program uses Singular Value Thresholding (SVT) [1], combined
#	with an unbiased risk estimator (PGURE) to denoise a video sequence
#	of microscopy images [2]. Noise parameters for a mixed Poisson-Gaussian
#	noise model are automatically estimated during the denoising.
#
#	References:
#	[1] "Unbiased Risk Estimates for Singular Value Thresholding and
#		Spectral Estimators", (2013), Candes, EJ et al.
#		http://dx.doi.org/10.1109/TSP.2013.2270464
#
#	[2]	"An Unbiased Risk Estimator for Image Denoising in the Presence
#		of Mixed Poisson-Gaussian Noise", (2014), Le Montagner, Y et al.
#		http://dx.doi.org/10.1109/TIP.2014.2300821
#
#   This file is part of PGURE-SVT.
#
#   PGURE-SVT is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   PGURE-SVT is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with PGURE-SVT. If not, see <http://www.gnu.org/licenses/>.

import ctypes, os
import numpy as np
from numpy.ctypeslib import ndpointer

class SVT(object):
    """       
    Parameters
    ----------
    patchsize : integer
        The dimensions of the patch in pixels 
        to form a Casorati matrix (default = 4)

    length : integer
        Length in frames of the block to form
        a Casorati matrix. Must be odd (default = 15)
        
    optimize : bool
        Whether to optimize PGURE or just denoise
        according to given threshold (default = True)
        
    threshold : float
        Threshold to use if not optimizing PGURE
        (default = 0.15)
        
    alpha : float
        Level of noise gain, if negative then
        estimated online (default = -1)
        
    mu : float
        Level of noise offset, if negative then
        estimated online (default = -1)
        
    sigma : float
        Level of Gaussian noise, if negative then
        estimated online (default = -1)
        
    arpssize : integer
        Size of neighbourhood for ARPS search
        (default = 7 pixels)
        
    tol : float
        Tolerance of PGURE optimizers
        (default = 1E-7)
        
    median : integer
        Size of initial median filter
        (default = 5 pixels)
    
    """
    
    def __init__(self, 
                patchsize=4,
                patchoverlap=1,
                length=15,
                optimize=True,
                threshold=-1.,
                alpha=-1., 
                mu=-1., 
                sigma=-1.,                
                arpssize=7, 
                tol=1e-7,
                median=5,
                hotpixelthreshold=10
                ):
                    
        self.patchsize = patchsize
        self.overlap = patchoverlap
        self.length = length
        self.optimize = optimize
        self.threshold = threshold
        self.alpha = alpha
        self.mu = mu
        self.sigma = sigma
        self.arpssize = arpssize
        self.tol = tol
        self.median = median
        self.hotpixelthreshold = hotpixelthreshold
        
        # Setup ctypes function
        libpath = os.path.dirname(os.path.abspath(__file__)) + '/build/libpguresvt.so'
        self._PGURESVT = ctypes.cdll.LoadLibrary(libpath).PGURESVT 
        self._PGURESVT.restype = ctypes.c_int      
        self._PGURESVT.argtypes = [ndpointer(ctypes.c_double, flags="F"),
                                   ndpointer(ctypes.c_double, flags="F"),
                                   ndpointer(ctypes.c_int),
                                   ctypes.c_int, 
                                   ctypes.c_int,
                                   ctypes.c_int,
                                   ctypes.c_bool,
                                   ctypes.c_double,
                                   ctypes.c_double, 
                                   ctypes.c_double, 
                                   ctypes.c_double,
                                   ctypes.c_int,
                                   ctypes.c_double,
                                   ctypes.c_int
                                   ctypes.c_double]

        self.Y = None
        
    def denoise(self, X):
        """Denoise the data X
        
        Parameters
        ----------
        X : array [nx, ny, time]
            The image sequence to be denoised
            
        Returns
        -------
        self : object
            Returns the instance itself
        
        """   
        
        self._denoise(X)
        return self
        
    def _denoise(self, X):
        """Denoise the data X
        
        Parameters
        ----------
        X : array [nx, ny, time]
            The image sequence to be denoised
            
        Returns
        -------
        Y : array [nx, ny, time]
            Returns the denoised sequence
        
        """
        X = self._check_array(X)
        dims = np.asarray(X.shape).astype(np.int32)
        Y = np.zeros(X.shape, dtype=np.double, order='F')
        result = self._PGURESVT(X,
                                Y,
                                dims,
                                self.patchsize,
                                self.overlap,
                                self.length,
                                self.optimize,
                                self.threshold,
                                self.alpha,
                                self.mu,
                                self.sigma,
                                self.arpssize,
                                self.tol,
                                self.median
                                self.hotpixelthreshold)
        self.Y = Y
        return Y

    def _check_array(self, X):
        """Sanity-checks the data and parameters.
        
        Parameters
        ----------
        X : array [nx, ny, time]
            The data as an array
            
        Returns
        -------
        x : array [nx, ny, time]
            Returns the array in Fortran-order (column-major)
        
        """  
        x = np.copy(X.astype(np.double), order='F')
        return x
   

