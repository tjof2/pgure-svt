"""	PGURE-SVT Denoising

	Author:	Tom Furnival
	Email:	tjof2@cam.ac.uk

	Copyright (C) 2015-16 Tom Furnival

	This program uses Singular Value Thresholding (SVT) [1], combined
	with an unbiased risk estimator (PGURE) to denoise a video sequence
	of microscopy images [2]. Noise parameters for a mixed Poisson-Gaussian
	noise model are automatically estimated during the denoising.

	References:
	[1] 	"Unbiased Risk Estimates for Singular Value Thresholding and
			Spectral Estimators", (2013), Candes, EJ et al.
			http://dx.doi.org/10.1109/TSP.2013.2270464

	[2]		"An Unbiased Risk Estimator for Image Denoising in the Presence
			of Mixed Poisson–Gaussian Noise", (2014), Le Montagner, Y et al.
			http://dx.doi.org/10.1109/TIP.2014.2300821
"""

import ctypes, os
import numpy as np
from numpy.ctypeslib import ndpointer

class PGURESVT(object):

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
    
    Attributes
    ----------
    """
    def __init__(self, 
                patchsize=4, 
                length=15,
                optimize=True,
                threshold=0.15,
                alpha=-1., 
                mu=-1., 
                sigma=-1.,                
                arpssize=7, 
                tol=1e-7,
                median=5
                ):
        
        # Setup ctypes function
        libpath = os.path.dirname(os.path.abspath(__file__)) + '/pguresvt.so'
        self._PGURESVT = ctypes.cdll.LoadLibrary(libpath).PGURE_SVT
        self._PGURESVT.restype = ctypes.c_int
        self._PGURESVT.argtypes = []
                          

  
   

