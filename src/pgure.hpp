/***************************************************************************

	Copyright (C) 2015-16 Tom Furnival
	
***************************************************************************/

#ifndef PGURE_H
#define PGURE_H

// C++ headers
#include <iostream>
#include <iomanip> 
#include <random>
#include <stdlib.h>
#include <vector>

// Armadillo library
#include <armadillo>

// NLopt library
#include <nlopt.h>

// Own header
#include "svt.hpp"

arma::cube GenerateRandomPerturbation(int N, int Nx, int Ny, int T);

double CalculatePGURE(unsigned n, const double *x, double *grad, void *p);

void PGUREOptimize(double *lambda, double *risk, void *p, double tol, double start, double bound, int eval);

struct PGureSearchParameters {
	int Nx;
	int Ny;
	int T;
	int Bs;
	
	// Number of function evaluations
	int count;

	// Mixed noise parameters
	double alpha;
	double mu;
	double sigma;

	// Perturbation amplitudes
	double eps1;
	double eps2;
	
	// Measured signal as (n^2 x T) Casorati matrix
	arma::mat *G;

	// Random samples for stochastic evaluation
	arma::cube *delta1;
	arma::cube *delta2;
	
	// SVD results for SVT thresholding and reconstruction
	std::vector<arma::mat> *U;
	std::vector<arma::vec> *S;
	std::vector<arma::mat> *V;
	std::vector<arma::mat> *U1;
	std::vector<arma::vec> *S1;
	std::vector<arma::mat> *V1;
	std::vector<arma::mat> *U2p;
	std::vector<arma::vec> *S2p;
	std::vector<arma::mat> *V2p;
	std::vector<arma::mat> *U2m;
	std::vector<arma::vec> *S2m;
	std::vector<arma::mat> *V2m;
	
	// Patch motions
	arma::icube *sequencePatches;
};

#endif
