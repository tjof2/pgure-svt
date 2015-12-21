/***************************************************************************

	Copyright (C) 2015 Tom Furnival
	
***************************************************************************/

#ifndef NOISE_H
#define NOISE_H

#include <iostream>
#include <stdlib.h>
#include <vector>

// Armadillo library
#include <armadillo>

double RobustVarEstimate(arma::vec *row);

double RobustMeanEstimate(arma::vec *row, int wtype);

void WeightFunction(arma::vec *x, arma::vec *w, int wtype);

double InterqDist(arma::vec *row);

arma::vec WLSFit(arma::vec *x, arma::vec *y, int wtype);

double ComputeMode(arma::vec a);

arma::vec RestrictArray(arma::vec *a, int Is, int Ie);

arma::mat ConvolveFIR(arma::mat *in, arma::mat *ker);

void EstimateNoiseParams(arma::cube *input, arma::cube *quadtree, double *alpha, double *mu, double *sigma, int size);

bool SplitBlockQ(arma::mat *img, int size);

std::vector<arma::umat> QuadTree(arma::mat *img, int size, std::vector<arma::umat> treeDelete, int part);

#endif

