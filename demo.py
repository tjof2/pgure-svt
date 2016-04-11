#	PGURE-SVT Denoising
#
#	Author:	Tom Furnival
#	Email:	tjof2@cam.ac.uk
#
#	Copyright (C) 2015-16 Tom Furnival
#
#	Demo script showing how to use PGURE-SVT in conjunction
#   with the HyperSpy multi-dimensional data analysis toolbox.
#     
#   [1] http://www.hyperspy.org
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
