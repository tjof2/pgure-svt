# Author: Tom Furnival
# License: GPLv3

import hyperspy.api as hs
from pguresvt import hspysvt

# Load example dataset
s = hs.load("./example.tif")

# Take the first 15 frames, and plot the result
s = s.inav[:15]
s.plot(navigator="slider")

# Denoise the HyperSpy signal directly
svt = hspysvt.HSPYSVT(patchsize=4, length=15, optimize=True, tol=1e-6)
s_denoised = svt.denoise(s)

# Plot denoised data
s_denoised.plot(navigator="slider")
