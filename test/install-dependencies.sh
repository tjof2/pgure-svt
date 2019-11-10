#!/bin/sh
#   This script builds from source:
#       - OpenBLAS 0.2.16
#       - Armadillo 7.200
#       - NLopt 2.4.2
#
# Copyright 2015-2019 Tom Furnival
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

set -ex
##########################################
wget http://github.com/xianyi/OpenBLAS/archive/v0.2.16.tar.gz
tar -xzvf v0.2.16.tar.gz
cd OpenBLAS-0.2.16
make NO_AFFINITY=1 > log-file 2>&1
sudo make install > log-file 2>&1
##########################################
# Armadillo 6.600
wget http://sourceforge.net/projects/arma/files/armadillo-7.950.1.tar.xz
tar -xvf armadillo-7.950.1.tar.xz
cd armadillo-7.950.1
cmake .
make
sudo make install
##########################################
# NLopt 2.4.2
wget http://ab-initio.mit.edu/nlopt/nlopt-2.4.2.tar.gz
tar -xzvf nlopt-2.4.2.tar.gz
cd nlopt-2.4.2
./configure --enable-shared
make
sudo make install
##########################################
