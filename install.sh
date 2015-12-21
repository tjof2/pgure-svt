#!/bin/bash

#----------------------------
#
#	PGURE-SVT Denoising
#
#	Author:	Tom Furnival
#	Email:	tjof2@cam.ac.uk
#	Date:	25/10/2015
#
#----------------------------

echo ""
echo "Compiling PGURE-SVT..."
echo ""

rm -r bin/
rm cmake_install.cmake CMakeCache.txt
mkdir bin

cmake .
make


