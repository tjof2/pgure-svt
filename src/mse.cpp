/***************************************************************************

	Copyright (C) 2015 Tom Furnival
	
	File: mse.cpp
	
	Optimization of MSE between denoised and ground-truth
	image sequences for comparison with PGURE
			 	
***************************************************************************/

#include "mse.hpp"

double CalculateMSE(unsigned n, const double *x, double *grad, void *p) {
	
	struct MSESearchParameters * params = (struct MSESearchParameters *)p;
	int Nx = params->Nx;
	int Ny = params->Ny;
	int T = params->T;
	int Bs = params->Bs;

	arma::mat *H = params->H;
	
	++params->count;
	
	std::vector<arma::mat> *U = params->U;
	std::vector<arma::vec> *S = params->S;
	std::vector<arma::mat> *V = params->V;
	
	arma::icube *sequencePatches = params->sequencePatches;

	arma::cube u(Nx, Ny, T);
	SVDReconstruct(x[0], &u, U, S, V, sequencePatches, Bs, Nx, Ny, T);
	
	arma::mat G(Nx*Ny, T);
	CubeReshape(&u, &G);

	arma::mat residual = arma::abs(G - *H);
	double mse = arma::mean(arma::mean(arma::square(residual)));	

	return mse;
}

void MSEOptimize(double *lambda, double *risk, void *p, double tol, double start, double bound, int eval) {

	double startingStep = start / 2;

	// Optimize MSE
	nlopt_opt opt;
	opt = nlopt_create(NLOPT_LN_BOBYQA, 1);
	nlopt_set_min_objective(opt, CalculateMSE, p);
	double lb[1], ub[1], x[1], dx[1];
	lb[0] = 0.;
	ub[0] = bound;
	x[0] = start;	
	dx[0] = startingStep;
	nlopt_set_lower_bounds(opt, lb);
	nlopt_set_upper_bounds(opt, ub);
	nlopt_set_initial_step(opt, dx);
	nlopt_set_ftol_rel(opt, tol);
	nlopt_set_maxeval(opt, 0);
			
	// Objective value
	double minf;
			
	// Run the optimizer
	int status = nlopt_optimize(opt, x, &minf);

	// Error checking for NLopt
	switch( status ) {
		case -5:
			std::cout<<"NLOPT_FORCED_STOP. Using previous lambda."<<std::endl;
			break;
		case -4:
			std::cout<<"NLOPT_ROUNDOFF_LIMITED"<<std::endl;
			// Set new lambda
			*lambda = x[0];
			break;
		case -3:
			std::cout<<"NLOPT_OUT_OF_MEMORY. Using previous lambda."<<std::endl;
			break;
		case -2:
			std::cout<<"NLOPT_INVALID_ARGS. Using previous lambda."<<std::endl;
			break;
		case -1:
			std::cout<<"NLOPT_FAILURE. Using previous lambda."<<std::endl;
			break;
		default:
			// Set new lambda
			*lambda = x[0];
			break;
	}	

	*risk = minf;

	return;	
}

