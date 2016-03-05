/***************************************************************************

	Copyright (C) 2015-16 Tom Furnival
	
***************************************************************************/

#ifndef SVT_H
#define SVT_H

// C++ headers
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <vector>

// OpenMP library
#include <omp.h>

// Armadillo library
#include <armadillo>

void CubeReshape(arma::cube *u, arma::mat *G);

void SVDDecompose(arma::cube *u, std::vector<arma::mat> *U, std::vector<arma::vec> *S, std::vector<arma::mat> *V, arma::icube *sequencePatches, int Bs, int Nx, int Ny, int T);

void SVDReconstruct(double lambda, arma::cube *vout, std::vector<arma::mat> *U, std::vector<arma::vec> *S, std::vector<arma::mat> *V, arma::icube *sequencePatches, int Bs, int Nx, int Ny, int T);

#endif
