/***************************************************************************

	Copyright (C) 2015 Tom Furnival
	
	File: arps.cpp
	
	Perform Adaptive Rood Pattern Search (ARPS) for motion estimation [1]. 
	Based on MATLAB code by Aroh Barjatya [2].
	
	References:
	[1] 	"Adaptive rood pattern search for fast block-matching motion 
			estimation", (2002), Nie, Y and Kai-Kuang, M
			http://dx.doi.org/10.1109/TIP.2002.806251  

	[2]		http://uk.mathworks.com/matlabcentral/fileexchange/8761-block-matching-algorithms-for-motion-estimation
	
***************************************************************************/

#include "arps.hpp"

// Adaptive Rood Pattern Search (ARPS) method
void ARPSMotionEstimation(arma::mat *frame1, arma::mat *frame2, arma::imat *refpatches, arma::imat *curpatches, arma::imat *motion, int curFr, int Bs, int wind) {

	int Nx = (*frame1).n_rows;
	int Ny = (*frame1).n_cols;
	int vecSize = (Nx-Bs+1)*(Ny-Bs+1);	

#pragma omp parallel for
	for(int it = 0; it < vecSize; it++) {

		arma::vec costs = arma::ones<arma::vec>(6) * 1E8;
		arma::umat chkMat = arma::zeros<arma::umat>(2*wind+1, 2*wind+1);
		arma::imat LDSP = arma::zeros<arma::imat>(6,2);
		arma::imat SDSP = arma::zeros<arma::imat>(5,2);
		SDSP(0,0) = 0; SDSP(0,1) = -1;
		SDSP(1,0) = -1; SDSP(1,1) = 0;
		SDSP(2,0) = 0; SDSP(2,1) = 0;
		SDSP(3,0) = 1; SDSP(3,1) = 0;
		SDSP(4,0) = 0; SDSP(4,1) = 1;
				
		LDSP.rows(arma::span(0,4)) = SDSP;	
		
		int i = it % (Nx-Bs+1);
		int j = it / (Ny-Bs+1);

		int x = j;
		int y = i;

		arma::mat refblock = (*frame1)(arma::span(i,i+Bs-1),arma::span(j,j+Bs-1));
		costs(2) = std::pow(arma::norm(refblock - (*frame2)(arma::span(i,i+Bs-1),arma::span(j,j+Bs-1)),"fro"),2)/(Bs*Bs);
		chkMat(wind,wind) = 1;

		int stepSize, maxIdx;
		if(j == 0) {
			stepSize = 2;
			maxIdx = 5;
		}
		else {
			int ytmp = std::abs((*motion)(0,it));
			int xtmp = std::abs((*motion)(1,it));			
			
			stepSize = (xtmp <= ytmp) ? ytmp : xtmp;
			
			if((xtmp == stepSize && ytmp == 0) || (xtmp == 0 && ytmp == stepSize)) {
				maxIdx = 5;
			}
			else {
				maxIdx = 6;
				LDSP(5,0) = (*motion)(1,it); LDSP(5,1) = (*motion)(0,it);
			}
		}
		LDSP(0,0) = 0; LDSP(0,1) = -stepSize;
		LDSP(1,0) = -stepSize; LDSP(1,1) = 0;
		LDSP(2,0) = 0; LDSP(2,1) = 0;
		LDSP(3,0) = stepSize; LDSP(3,1) = 0;
		LDSP(4,0) = 0; LDSP(4,1) = stepSize;

		double pMot = 0.0;
		
		// Do the LDSP
		for(int k = 0; k < maxIdx; k++) {
			int refBlkVer = y + LDSP(k,1);
			int refBlkHor = x + LDSP(k,0);						
			if( refBlkHor < 0 || refBlkHor+Bs-1 >= Ny || refBlkVer < 0 || refBlkVer+Bs-1 >= Nx ) { continue; }
			else if( k == 2 || stepSize == 0 ) { continue; }
			else {
				if(curFr == 0) {
					costs(k) = std::pow(arma::norm(refblock - (*frame2)(arma::span(refBlkVer,refBlkVer+Bs-1),arma::span(refBlkHor,refBlkHor+Bs-1)),"fro"),2)/(Bs*Bs);			
				}
				else if(curFr < 0) {
					arma::ivec predpos = (*refpatches).col(it) - (*motion).col(it);
					costs(k) = std::pow(arma::norm(refblock - (*frame2)(arma::span(refBlkVer,refBlkVer+Bs-1),arma::span(refBlkHor,refBlkHor+Bs-1)),"fro"),2)/(Bs*Bs) + pMot * std::sqrt(std::pow(predpos(0)-refBlkVer,2)+std::pow(predpos(1)-refBlkHor,2));		
				}
				else if(curFr > 0) {
					arma::ivec predpos = (*refpatches).col(it) + (*motion).col(it);
					costs(k) = std::pow(arma::norm(refblock - (*frame2)(arma::span(refBlkVer,refBlkVer+Bs-1),arma::span(refBlkHor,refBlkHor+Bs-1)),"fro"),2)/(Bs*Bs) + pMot * std::sqrt(std::pow(predpos(0)-refBlkVer,2)+std::pow(predpos(1)-refBlkHor,2));		
				}		
				chkMat(LDSP(k,1)+wind, LDSP(k,0)+wind) = 1;				
			}
		}

		arma::uvec point = arma::find(costs == costs.min());
		x += LDSP(point(0), 0);
		y += LDSP(point(0), 1);
		double cost = costs.min();
		costs.ones();
		costs *= 1E8;
		costs(2) = cost;	

		// Do the SDSP
		int doneFlag = 0;
		do {
			for(int k = 0; k < 5; k++) {
				int refBlkVer = y + SDSP(k,1);
				int refBlkHor = x + SDSP(k,0);	
				
				if( refBlkHor < 0 || refBlkHor+Bs-1 >= Ny || refBlkVer < 0 || refBlkVer+Bs-1 >= Nx ) { continue; }
				else if( k == 2 ) { continue; }
				else if( refBlkHor < j-wind || refBlkHor > j+wind || refBlkVer < i-wind || refBlkVer > i+wind ) { continue; }
				else if( chkMat(y-i+SDSP(k,1)+wind, x-j+SDSP(k,0)+wind) == 1) { continue; }
				else {
					if(curFr == 0) {
						costs(k) = std::pow(arma::norm(refblock - (*frame2)(arma::span(refBlkVer,refBlkVer+Bs-1),arma::span(refBlkHor,refBlkHor+Bs-1)),"fro"),2)/(Bs*Bs);			
					}
					else if(curFr < 0) {
						arma::ivec predpos = (*refpatches).col(it) - (*motion).col(it);
						costs(k) = std::pow(arma::norm(refblock - (*frame2)(arma::span(refBlkVer,refBlkVer+Bs-1),arma::span(refBlkHor,refBlkHor+Bs-1)),"fro"),2)/(Bs*Bs) + pMot * std::sqrt(std::pow(predpos(0)-refBlkVer,2)+std::pow(predpos(1)-refBlkHor,2));		
					}
					else if(curFr > 0) {
						arma::ivec predpos = (*refpatches).col(it) + (*motion).col(it);
						costs(k) = std::pow(arma::norm(refblock - (*frame2)(arma::span(refBlkVer,refBlkVer+Bs-1),arma::span(refBlkHor,refBlkHor+Bs-1)),"fro"),2)/(Bs*Bs) + pMot * std::sqrt(std::pow(predpos(0)-refBlkVer,2)+std::pow(predpos(1)-refBlkHor,2));		
					}
					chkMat(y-i+SDSP(k,1)+wind, x-j+SDSP(k,0)+wind) = 1;
				}
			}
			point = arma::find(costs == costs.min());
			cost = costs.min();
			
			if(point(0) == 2) {
				doneFlag = 1;
			}
			else {				
				x += SDSP(point(0), 0);
				y += SDSP(point(0), 1);
				costs.ones();
				costs *= 1E8;
				costs(2) = cost;
			}
		} while(doneFlag == 0);

		int ystep = y - i;
		int xstep = x - j;
		
		(*motion)(0,it) = ystep;
		(*motion)(1,it) = xstep;

		(*curpatches)(0,it) = y;
		(*curpatches)(1,it) = x;
	}
	return;
}


