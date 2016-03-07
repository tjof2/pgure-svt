/***************************************************************************

	Copyright (C) 2015-16 Tom Furnival
	
***************************************************************************/

#ifndef ARPS_H
#define ARPS_H

// C++ headers
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <vector>

// OpenMP library
#include <omp.h>

// Armadillo library
#include <armadillo>

void ARPSMotionEstimation(arma::mat *frame1, arma::mat *frame2, arma::imat *refpatches, arma::imat *curpatches, arma::imat *motion, int curFr, int Bs, int wind);

#endif
