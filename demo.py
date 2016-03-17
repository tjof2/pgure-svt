"""	PGURE-SVT Denoising

	Author:	Tom Furnival
	Email:	tjof2@cam.ac.uk

	Copyright (C) 2015-16 Tom Furnival

	Demo script showing how to use PGURE-SVT in conjunction
     with the HyperSpy multi-dimensional data analysis toolbox.
     
     [1] 	http://www.hyperspy.org
 
"""

import hyperspy.api as hs
from hyperspy.hspy import *
import matplotlib.pyplot as plt
#import pguresvt

# Load the file name
imgseq = hs.load('example.dm4', stack=True)
X = imgseq.data

# Initialize with default parameters
svt = pguresvt.SVT()

# Run the denoising
Y = SVT.denoise(X)

# Plot
imgseq.plot(navigator='slider')