/***************************************************************************

	Copyright (C) 2015 Tom Furnival
	
***************************************************************************/

#ifndef MSE_H
#define MSE_H

// C++ headers
#include <iostream>
#include <random>
#include <stdlib.h>
#include <vector>

// Armadillo library
#include <armadillo>

// NLopt library
#include <nlopt.h>

// Own header
#include "svt.hpp"

double CalculateMSE(unsigned n, const double *x, double *grad, void *p);

void MSEOptimize(double *lambda, double *risk, void *p, double tol, double start, double bound, int eval);

struct MSESearchParameters {
	int Nx;
	int Ny;
	int T;
	int Bs;
	
	// Number of function evaluations
	int count;
	
	// Measured signal SVD
	std::vector<arma::mat> *U;
	std::vector<arma::vec> *S;
	std::vector<arma::mat> *V;

	// Clean signal  as (n^2 x T) Casorati matrix
	arma::mat *H;
	
	// Patch motions
	arma::icube *sequencePatches;
};

#endif
