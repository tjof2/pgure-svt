/***************************************************************************

	Copyright (C) 2015 Tom Furnival
	
	File: tools.cpp
	
	Library of miscellaneous functions:
		1. Comparison metrics including PSNR and correlation distance
		2. Generate noise according to a mixed Poisson-Gaussian model
			 	
***************************************************************************/

#include "tools.hpp"

// Mersenne twister RNG
std::mt19937 engine;

// Little function to convert string "0"/"1" to boolean
bool strToBool(std::string const& s) {
     return s != "0";
};


/////////////////////////////
//						   //
//   COMPARISON METRICS    //
//						   //
/////////////////////////////

// PSNR metric for two matrices
double MSECalc(arma::mat *A, arma::mat *B) {
	int n = (*A).n_elem;
	double mse = std::abs(arma::norm(*A - *B,"fro"));
	return -10 * std::log10(mse*mse/n);
}

// Normalized PSNR metric for two matrices
double MSENormCalc(arma::mat *A, arma::mat *B) {
	double normA = arma::norm(*A - arma::mean(arma::mean(*A)),"fro");
	double normB = arma::norm(*B - arma::mean(arma::mean(*B)),"fro");
	double denom = normA*normA + normB*normB;
	double mse = arma::norm((*A - arma::mean(arma::mean(*A))) - (*B - arma::mean(arma::mean(*B))),"fro");
	
	return -10 * std::log10(mse*mse/(2*denom));
}

// Correlation distance metric for two matrices
double CorrCalc(arma::mat *A, arma::mat *B) {
	arma::mat Amean = *A - arma::mean(arma::mean(*A));
	arma::mat Bmean = *B - arma::mean(arma::mean(*B));	
	double corr = arma::accu(Amean % Bmean) / std::sqrt( arma::accu(Amean % Amean) * arma::accu(Bmean % Bmean) );
	return corr;
}

/////////////////////////////
//						   //
//     GENERATE NOISE      //
//						   //
/////////////////////////////

// Add image noise according to Poisson-Gaussian noise model
void AddImageNoise(arma::mat *input, arma::mat *output, double alpha, double mu, double sigma) {
	int x = (*input).n_rows;
	int y = (*input).n_cols;

#pragma omp parallel for
	for(int j = 0; j<(x*y); j++) {
		int xi = j % x;
		int xj = j / y;

		double invalue = (*input)(xi, xj);
		double outvalue;

		if(invalue == 0.) {
			outvalue = 0.;
			if( sigma != 0.) {
				std::normal_distribution<double> distributionNormal(mu, sigma);
				outvalue += distributionNormal(engine);
			}
			else {
				outvalue += mu;
			}
		}
		else {
			if( alpha != 0. ) {
				std::poisson_distribution<int> distributionPoisson(invalue / alpha);
				outvalue = alpha * distributionPoisson(engine);
			}
			else {
				outvalue = invalue;
			}
			if( sigma != 0.) {
				std::normal_distribution<double> distributionNormal(mu, sigma);
				outvalue += distributionNormal(engine);
			}
			else {
				outvalue += mu;
			}
		}
		(*output)(xi, xj) = outvalue;
	}

	return;
}

