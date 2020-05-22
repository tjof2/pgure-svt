#!/bin/sh
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

set -ex
mkdir build_deps
cd build_deps

# Versions to install
ARMA_VER="9.880.1"
NLOP_VER="2.6.2"

# Armadillo
wget http://sourceforge.net/projects/arma/files/armadillo-${ARMA_VER}.tar.xz
tar -xvf armadillo-${ARMA_VER}.tar.xz > arma.log 2>&1
cd armadillo-${ARMA_VER}
cmake .
make
sudo make install
cd ../

# NLopt
wget https://github.com/stevengj/nlopt/archive/v${NLOP_VER}.tar.gz
tar -xzvf v${NLOP_VER}.tar.gz > nlopt.log 2>&1
cd nlopt-${NLOP_VER}
cmake .
make
sudo make install
cd ../

# Tidy-up
cd ../
sudo rm -rf build_deps/
sudo ldconfig
