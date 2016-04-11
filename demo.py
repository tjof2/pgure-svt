#	PGURE-SVT Denoising
#
#	Author:	Tom Furnival
#	Email:	tjof2@cam.ac.uk
#
#	Copyright (C) 2015-16 Tom Furnival
#
#	Demo script showing how to use PGURE-SVT
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

import numpy as np
import pguresvt

# Generate a random array
X =  np.random.randint(65535, size=(64, 64, 20))

# Initialize with default parameters
svt = pguresvt.SVT(patchsize=4,
                   length=15,
                   optimize=False,
                   threshold=0.1,
                   tol=1e-6)

# Run the denoising
Y = svt.denoise(X)


