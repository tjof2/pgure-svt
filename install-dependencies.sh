#!/bin/sh
#
#   This script builds from source:
#       - OpenBLAS 0.2.16
#       - Armadillo 6.600
#       - NLopt 2.4.2
#
set -ex
##########################################
wget http://github.com/xianyi/OpenBLAS/archive/v0.2.16.tar.gz
tar -xzvf OpenBLAS-0.2.16.tar.gz
cd OpenBLAS-0.2.16
make
sudo make install
##########################################
# Armadillo 6.600
wget http://sourceforge.net/projects/arma/files/armadillo-6.600.4.tar.gz
tar -xzvf armadillo-6.600.4.tar.gz
cd armadillo-6.600.4
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

