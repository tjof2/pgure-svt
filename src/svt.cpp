/***************************************************************************

	Copyright (C) 2015-16 Tom Furnival

	File: svt.cpp

	SVT calculation based on [1].

	References:
	[1] 	"Unbiased Risk Estimates for Singular Value Thresholding and
			Spectral Estimators", (2013), Candes, EJ et al.
			http://dx.doi.org/10.1109/TSP.2013.2270464

***************************************************************************/

#include "svt.hpp"

// Reshape a cube to a Casorati matrix
void CubeReshape(arma::cube *u, arma::mat *G) {
		// Reshape to n^2 x T Casorati matrix
		int x = (*u).n_rows;
		int y = (*u).n_cols;
		int z = (*u).n_slices;
		arma::cube Gtemp = *u;
		Gtemp.reshape(x*y, z, 1);
		*G = Gtemp.slice(0);
		return;
}

// Perform SVD on each block in the image sequence
void SVDDecompose(arma::cube *u, std::vector<arma::mat> *U, std::vector<arma::vec> *S, std::vector<arma::mat> *V, arma::icube *sequencePatches, int Bs, int Nx, int Ny, int T) {

	int vecSize = (Nx-Bs+1)*(Ny-Bs+1);

	arma::mat Ublock(Bs*Bs,T);
	arma::vec Sblock(T);
	arma::mat Vblock(T,T);

	// Do the local SVDs
#pragma omp parallel for private(Ublock, Sblock, Vblock)
	for(int it = 0; it < vecSize; it++) {

		// Extract block
		arma::mat block(Bs*Bs,T);
		for(int k = 0; k < T; k++) {
			int newy = (*sequencePatches)(0,it,k);
			int newx = (*sequencePatches)(1,it,k);
			block.col(k) = arma::vectorise((*u)(arma::span(newy,newy+Bs-1),arma::span(newx,newx+Bs-1),arma::span(k,k)));
		}

		// Do the SVD
		arma::svd_econ(Ublock, Sblock, Vblock, block);
		(*U)[it] = Ublock;
		(*S)[it] = Sblock;
		(*V)[it] = Vblock;
	}

	return;
}

// Reconstruct block in the image sequence after thresholding
void SVDReconstruct(double lambda, arma::cube *vout, std::vector<arma::mat> *U, std::vector<arma::vec> *S, std::vector<arma::mat> *V, arma::icube *sequencePatches, int Bs, int Nx, int Ny, int T) {

	int vecSize = (Nx-Bs+1)*(Ny-Bs+1);

	arma::mat block(Bs*Bs,T);
	arma::mat Ublock(Bs*Bs,T);
	arma::vec Sblock(T);
	arma::mat Vblock(T,T);

	arma::cube v = arma::zeros<arma::cube>(Nx,Ny,T);
	arma::cube weights = arma::zeros<arma::cube>(Nx,Ny,T);

#pragma omp parallel for shared(v, weights) private(block, Ublock, Sblock, Vblock)
	for(int it = 0; it < vecSize; it++) {
  		Ublock = (*U)[it];
  		Sblock = (*S)[it];
  		Vblock = (*V)[it];

	  	// Basic singular value thresholding
  		//arma::vec Snew = arma::sign(Sblock) % arma::max(arma::abs(Sblock) - lambda, arma::zeros<arma::vec>(T));

	  	// Gaussian-weighted singular value thresholding
  		arma::vec wvec = arma::abs(Sblock.max() * arma::exp(- lambda * arma::square(Sblock)/2));

  		// Apply threshold
  		arma::vec Snew = arma::sign(Sblock) % arma::max(arma::abs(Sblock) - wvec, arma::zeros<arma::vec>(T));

  		// Reconstruct from SVD
  		block = Ublock * diagmat(Snew) * Vblock.t();

		// Deal with block weights (TODO: currently all weights = 1)
		for(int k = 0; k < T; k++) {
			int newy = (*sequencePatches)(0,it,k);
			int newx = (*sequencePatches)(1,it,k);
			v(arma::span(newy,newy+Bs-1),arma::span(newx,newx+Bs-1), arma::span(k,k)) += arma::reshape(block.col(k), Bs, Bs);
			weights(arma::span(newy,newy+Bs-1),arma::span(newx,newx+Bs-1), arma::span(k,k)) += arma::ones<arma::mat>(Bs,Bs);
		}
	}

	// Include the weighting (avoid errors dividing by zero for blocks that aren't used)
	v /= weights;
	v.elem( find_nonfinite(v) ).zeros();
	*vout = v;

	return;
}



