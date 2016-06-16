#!/bin/sh
#
#   This script builds from source:
#       - OpenBLAS 0.2.16
#       - Armadillo 7.200
#       - NLopt 2.4.2
#
set -ex
##########################################
wget http://github.com/xianyi/OpenBLAS/archive/v0.2.16.tar.gz
tar -xzvf v0.2.16.tar.gz
cd OpenBLAS-0.2.16
make NO_AFFINITY=1
sudo make install
##########################################
# Armadillo 6.600
wget http://sourceforge.net/projects/arma/files/armadillo-7.200.1.tar.xz
tar -xzvf armadillo-7.200.1.tar.xz
cd armadillo-7.200.1
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

