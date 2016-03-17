/***************************************************************************

	Copyright (C) 2015 Tom Furnival
	
***************************************************************************/

#ifndef TOOLS_H
#define TOOLS_H

// C++ headers
#include <iostream>
#include <random>
#include <omp.h>
#include <stdlib.h>

// Armadillo library
#include <armadillo>

bool strToBool(std::string const& s);

double MSECalc(arma::mat *A, arma::mat *B);

double MSENormCalc(arma::mat *A, arma::mat *B);

double CorrCalc(arma::mat *A, arma::mat *B);

void AddImageNoise(arma::mat *input, arma::mat *output, double alpha, double mu, double sigma);

#endif
