#!/bin/sh
# Author: Tom Furnival
# License: GPLv3

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
