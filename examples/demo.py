# -*- coding: utf-8 -*-
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

import hyperspy.api as hs
from pguresvt import hspysvt

# Load example dataset
s = hs.load("./example.tif")

# Take the first 15 frames, and plot the result
s = s.inav[:15]
s.plot(navigator="slider")

# Denoise the HyperSpy signal directly.
svt = hspysvt.HSPYSVT(patchsize=4, length=15, optimize=True, tol=1e-6)
s_denoised = svt.denoise(movie)

# Plot denoised data
s_denoised.plot(navigator="slider")
