/***************************************************************************

	Copyright (C) 2015 Tom Furnival
	
	File: pgure.cpp
	
	Optimization of PGURE between noisy and denoised image sequences.
	PGURE is an extension of the formula presented in [1].
	
	References:
	[1]		"An Unbiased Risk Estimator for Image Denoising in the Presence 
			of Mixed Poisson–Gaussian Noise", (2014), Le Montagner, Y et al.
			http://dx.doi.org/10.1109/TIP.2014.2300821 
			 	
***************************************************************************/

#include "pgure.hpp"

// Perturbations used in empirical calculation of d'f(y) and d''f(y)
arma::cube GenerateRandomPerturbation(int N, int Nx, int Ny, int T) {

	std::mt19937 rand_engine;
	arma::cube delta(Nx,Ny,T);

	if(N == 1) {
		std::bernoulli_distribution binary_dist(0.5);
		delta.imbue( [&]() {
			bool bernRand = binary_dist(rand_engine);
			if( bernRand == true ) {
				return -1;
			}
			else {
				return 1;
			}
		} );
	}
	else if(N == 2) {
		double kappa = 1.;
		double vP = (1/2)+(kappa/2)/std::sqrt(kappa*kappa+4);
		double vQ = 1 - vP;
		std::bernoulli_distribution binary_dist(vP);
		delta.imbue( [&]() {
			bool bernRand = binary_dist(rand_engine);
			if( bernRand == true ) {
				return -1 * std::sqrt(vQ/vP);
			}
			else {
				return std::sqrt(vP/vQ);
			}
		} );
	}

	return delta;
}

// Calculate PGURE
double CalculatePGURE(unsigned n, const double *x, double *grad, void *p) {
	
	struct PGureSearchParameters * params = (struct PGureSearchParameters *)p;
	int Nx = params->Nx;
	int Ny = params->Ny;
	int T = params->T;
	int Bs = params->Bs;

	double alpha = params->alpha;
	double sigma = params->sigma;
	double mu = params->mu;

	double eps1 = params->eps1;
	double eps2 = params->eps2;

	arma::mat *G = params->G;
	
	++params->count;
	
	std::vector<arma::mat> *U = params->U;
	std::vector<arma::vec> *S = params->S;
	std::vector<arma::mat> *V = params->V;
	std::vector<arma::mat> *U1 = params->U1;
	std::vector<arma::vec> *S1 = params->S1;
	std::vector<arma::mat> *V1 = params->V1;
	std::vector<arma::mat> *U2p = params->U2p;
	std::vector<arma::vec> *S2p = params->S2p;
	std::vector<arma::mat> *V2p = params->V2p;
	std::vector<arma::mat> *U2m = params->U2m;
	std::vector<arma::vec> *S2m = params->S2m;
	std::vector<arma::mat> *V2m = params->V2m;

	arma::cube *delta1 = params->delta1;
	arma::cube *delta2 = params->delta2;
	
	arma::icube *sequencePatches = params->sequencePatches;
	
	arma::cube uhat(Nx, Ny, T), u1(Nx, Ny, T), u2p(Nx, Ny, T), u2m(Nx, Ny, T);

	SVDReconstruct(x[0], &uhat, U, S, V, sequencePatches, Bs, Nx, Ny, T);
	SVDReconstruct(x[0], &u1, U1, S1, V1, sequencePatches, Bs, Nx, Ny, T);
	SVDReconstruct(x[0], &u2p, U2p, S2p, V2p, sequencePatches, Bs, Nx, Ny, T);
	SVDReconstruct(x[0], &u2m, U2m, S2m, V2m, sequencePatches, Bs, Nx, Ny, T);
	
	arma::mat Ghat(Nx*Ny, T), G1(Nx*Ny, T), G2p(Nx*Ny, T), G2m(Nx*Ny, T), Gdelta1(Nx*Ny, T), Gdelta2(Nx*Ny, T);
	CubeReshape(&uhat, &Ghat);
	CubeReshape(&u1, &G1);
	CubeReshape(&u2p, &G2p);
	CubeReshape(&u2m, &G2m);
	CubeReshape(delta1, &Gdelta1);
	CubeReshape(delta2, &Gdelta2);

	// Modified from [1] to include mean/offset
	double pgURE;
	pgURE = arma::mean(arma::mean(arma::square(arma::abs(Ghat - *G))))
			- (alpha + mu) * arma::mean(arma::mean(*G))
			+ 2/eps1 * arma::mean(arma::mean(Gdelta1 % (alpha * (*G) - alpha*mu + sigma*sigma) % (G1 - Ghat))) 
			- 2*sigma*sigma*alpha/(eps2*eps2) * arma::mean(arma::mean(Gdelta2 % (G2p - 2*Ghat + G2m))) 
			+ 2 * mu * arma::mean(arma::mean(Ghat))
			+ mu/(Nx*Ny*T)
			- sigma*sigma;
			
	return pgURE;
}

// Optimization function using NLopt and BOBYQA gradient-free algorithm
void PGUREOptimize(double *lambda, double *risk, void *p, double tol, double start, double bound, int eval) {

	double startingStep = start / 2;

	// Optimize MSE
	nlopt_opt opt;
	opt = nlopt_create(NLOPT_LN_BOBYQA, 1);
	nlopt_set_min_objective(opt, CalculatePGURE, p);
	double lb[1], ub[1], x[1], dx[1], xtolabs[1];
	lb[0] = 0.;
	ub[0] = bound;
	x[0] = start;	
	dx[0] = startingStep;
	xtolabs[0] = 1E-12;
	nlopt_set_lower_bounds(opt, lb);
	nlopt_set_upper_bounds(opt, ub);
	nlopt_set_initial_step(opt, dx);
	nlopt_set_ftol_rel(opt, tol);
	nlopt_set_xtol_abs(opt, xtolabs);
	nlopt_set_maxeval(opt, eval);

	// Objective value
	double minf;
			
	// Run the optimizer
	int status = nlopt_optimize(opt, x, &minf);

	// Error checking for NLopt
	switch( status ) {
		case -5:
			std::cout<<"   NLOPT_FORCED_STOP. Using previous lambda."<<std::endl;
			break;
		case -4:
			std::cout<<"   NLOPT_ROUNDOFF_LIMITED. Using previous lambda."<<std::endl;
			break;
		case -3:
			std::cout<<"   NLOPT_OUT_OF_MEMORY. Using previous lambda."<<std::endl;
			break;
		case -2:
			std::cout<<"   NLOPT_INVALID_ARGS. Using previous lambda."<<std::endl;
			break;
		case -1:
			std::cout<<"   NLOPT_FAILURE. Using previous lambda."<<std::endl;
			break;
		default:
			// Set new lambda
			*lambda = x[0];
			break;
	}	
	
	*risk = minf;

	return;	
}

