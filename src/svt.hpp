/***************************************************************************

	Copyright (C) 2015-16 Tom Furnival

	SVT calculation based on [1].

	References:
	[1] 	"Unbiased Risk Estimates for Singular Value Thresholding and
			Spectral Estimators", (2013), Candes, EJ et al.
			http://dx.doi.org/10.1109/TSP.2013.2270464

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

class SVT {
	public:
		SVT() {};
		~SVT() {};
		
		void Initialize(const arma::icube &sequencePatches, int w, int h, int l, int blocksize, int blockoverlap) {
			patches = sequencePatches;
			
			Nx = w;
			Ny = h;
			T = l;
			
			Bs = blocksize;
			Bo = blockoverlap;
			
			vecSize = (1+(Nx-Bs)/Bo)*(1+(Ny-Bs)/Bo);
			
			// Memory allocation			
			U.resize(vecSize);
			S.resize(vecSize);
			V.resize(vecSize);			
			for(int it = 0; it < vecSize; it++) {
				U[it] = arma::zeros<arma::mat>(Bs*Bs,T);
				S[it] = arma::zeros<arma::vec>(T);
				V[it] = arma::zeros<arma::mat>(T,T);
			}			
			return;
		};
		
		// Perform SVD on each block in the image sequence
		void Decompose(const arma::cube &u) {
			// Do the local SVDs
			arma::mat block, Ublock, Vblock;
			arma::vec Sblock;
		
			block.set_size(Bs*Bs,T);
			Ublock.set_size(Bs*Bs,T);
			Sblock.set_size(T);
			Vblock.set_size(T,T); 
			
			#pragma omp parallel for private(block, Ublock, Sblock, Vblock)
			for(int it = 0; it < vecSize; it++) {
				// Extract block
				for(int k = 0; k < T; k++) {
					int newy = patches(0,it,k);
					int newx = patches(1,it,k);
					block.col(k) = arma::vectorise(u(arma::span(newy,newy+Bs-1),arma::span(newx,newx+Bs-1),arma::span(k)));
				}

				// Do the SVD
				arma::svd_econ(Ublock, Sblock, Vblock, block);
				U[it] = Ublock;
				S[it] = Sblock;
				V[it] = Vblock;
			}
			return;
		};
		
		// Reconstruct block in the image sequence after thresholding
		arma::cube Reconstruct(double lambda) {
			arma::cube v = arma::zeros<arma::cube>(Nx,Ny,T);
			arma::cube weights = arma::zeros<arma::cube>(Nx,Ny,T);
			
			arma::mat block, Ublock, Vblock;
			arma::vec Sblock;
		
			block.set_size(Bs*Bs,T);
			Ublock.set_size(Bs*Bs,T);
			Sblock.set_size(T);
			Vblock.set_size(T,T); 
			
			#pragma omp parallel for shared(v, weights) private(block, Ublock, Sblock, Vblock)
			for(int it = 0; it < vecSize; it++) {
		  		Ublock = U[it];
		  		Sblock = S[it];
		  		Vblock = V[it];

			  	// Basic singular value thresholding
		  		//arma::vec Snew = arma::sign(Sblock) % arma::max(arma::abs(Sblock) - lambda, arma::zeros<arma::vec>(T));

			  	// Gaussian-weighted singular value thresholding
		  		arma::vec wvec = arma::abs(Sblock.max() * arma::exp(-1 * lambda * arma::square(Sblock)/2));

		  		// Apply threshold
		  		arma::vec Snew = arma::sign(Sblock) % arma::max(arma::abs(Sblock) - wvec, arma::zeros<arma::vec>(T));

		  		// Reconstruct from SVD
		  		block = Ublock * diagmat(Snew) * Vblock.t();

				// Deal with block weights (TODO: currently all weights = 1)
				for(int k = 0; k < T; k++) {
					int newy = patches(0,it,k);
					int newx = patches(1,it,k);
					v(arma::span(newy,newy+Bs-1),arma::span(newx,newx+Bs-1), arma::span(k,k)) += arma::reshape(block.col(k), Bs, Bs);
					weights(arma::span(newy,newy+Bs-1),arma::span(newx,newx+Bs-1), arma::span(k,k)) += arma::ones<arma::mat>(Bs,Bs);
				}
			}

			// Include the weighting (avoid errors dividing by zero for blocks that aren't used)
			v /= weights;
			v.elem( find_nonfinite(v) ).zeros();
			return v;
		}
	
	private:
	
		int Nx, Ny, T, Bs, Bo, vecSize;
		arma::icube patches;
	
		// Collate U, S, V
		std::vector<arma::mat> U, V;
		std::vector<arma::vec> S;		
};

#endif
