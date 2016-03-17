/***************************************************************************

    Copyright ( C) 2015-16 Tom Furnival

    Perform Adaptive Rood Pattern Search ( ARPS) for motion estimation [1].
    Based on MATLAB code by Aroh Barjatya [2].

    References:
    [1]     "Adaptive rood pattern search for fast block-matching motion
            estimation", ( 2002), Nie, Y and Kai-Kuang, M
            http://dx.doi.org/10.1109/TIP.2002.806251

    [2]     http://uk.mathworks.com/matlabcentral/fileexchange/8761-block-matching-algorithms-for-motion-estimation

***************************************************************************/

#ifndef ARPS_H
#define ARPS_H

// OpenMP library
#include <omp.h>

// C++ headers
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

// Armadillo library
#include <armadillo>

class MotionEstimator {
 public:
        MotionEstimator() {}
        ~MotionEstimator() {}

        void Estimate(const arma::cube &A,
                      int iter,
                      int timewindow,
                      int num_images,
                      int blocksize,
                      int blockoverlap,
                      int MotionP) {
            Nx = A.n_rows;
            Ny = A.n_cols;
            T = A.n_slices;
            wind = MotionP;
            Bs = blocksize;
            Bo = blockoverlap;
            vecSize = (1+(Nx-Bs)/Bo)*(1+(Ny-Bs)/Bo);

            patches = arma::zeros<arma::icube>( 2, vecSize, 2*timewindow+1);
            motions = arma::zeros<arma::icube>( 2, vecSize, 2*timewindow);

            // Perform motion estimation
            // Complicated for cases near beginning and end of sequence
            if ( iter < timewindow) {
                // Populate reference frame coordinates
                for ( int i = 0; i < vecSize; i++) {
                    patches( 0, i, iter) = i % (1+(Ny-Bs)/Bo);
                    patches( 1, i, iter) = i / (1+(Nx-Bs)/Bo);
                }
                // Perform motion estimation
                // Go forwards
                for (int i = 0; i < T-iter-1; i++) {
                    ARPSMotionEstimation( A, i, iter+i, iter+i+1, iter+i);
                }
                // Go backwards
                for (int i = -1; i >= -iter; i--) {
                    ARPSMotionEstimation( A, i, iter+i+1, iter+i, iter+i+1);
                }
            }
            else if (iter >= (num_images - timewindow)) {
                int endseqFrame = iter - (num_images - T);
                // Populate reference frame coordinates
                for ( int i = 0; i < vecSize; i++) {
                    patches( 0, i, endseqFrame) = i % (1+(Ny-Bs)/Bo);
                    patches( 1, i, endseqFrame) = i / (1+(Nx-Bs)/Bo);
                }
                // Perform motion estimation
                // Go forwards
                for ( int i = 0; i < 2*timewindow - endseqFrame; i++) {
                    ARPSMotionEstimation( A, i, endseqFrame+i, endseqFrame+i+1, endseqFrame+i);
                }
                // Go backwards
                for ( int i = -1; i >= -endseqFrame; i--) {
                    if ( 2*timewindow == endseqFrame) {
                        ARPSMotionEstimation( A, i, endseqFrame+i+1, endseqFrame+i, endseqFrame+i);
                    }
                    else {
                        ARPSMotionEstimation( A, i, endseqFrame+i+1, endseqFrame+i, endseqFrame+i+1);
                    }
                }
            }
            else {
                // Populate reference frame coordinates
                for ( int i = 0; i < vecSize; i++) {
                    patches( 0, i, timewindow) = i % ( 1+( Ny-Bs)/Bo);
                    patches( 1, i, timewindow) = i / ( 1+( Nx-Bs)/Bo);
                }
                // Perform motion estimation
                // Go forwards
                for ( int i = 0; i < timewindow; i++) {
                    ARPSMotionEstimation( A, i, timewindow+i, timewindow+i+1, timewindow+i);
                }
                // Go backwards
                for ( int i = -1; i >= -timewindow; i--) {
                    ARPSMotionEstimation( A, i, timewindow+i+1, timewindow+i, timewindow+i+1);
                }
            }
            return;
        }

        arma::icube GetEstimate() {
            return patches;
        }

 private:
        arma::icube patches, motions;
        int Nx, Ny, T, Bs, Bo, vecSize, wind;

        // Adaptive Rood Pattern Search ( ARPS) method
        void ARPSMotionEstimation(const arma::cube &A,
                                  int curFr,
                                  int iARPS1,
                                  int iARPS2,
                                  int iARPS3) {
            #pragma omp parallel for
            for ( int it = 0; it < vecSize; it++) {
                arma::vec costs = arma::ones<arma::vec>( 6) * 1E8;
                arma::umat chkMat = arma::zeros<arma::umat>( 2*wind+1, 2*wind+1);
                arma::imat LDSP = arma::zeros<arma::imat>( 6, 2);
                arma::imat SDSP = arma::zeros<arma::imat>( 5, 2);
                SDSP( 0, 0) = 0;
                SDSP( 0, 1) = -1;
                SDSP( 1, 0) = -1;
                SDSP( 1, 1) = 0;
                SDSP( 2, 0) = 0;
                SDSP( 2, 1) = 0;
                SDSP( 3, 0) = 1;
                SDSP( 3, 1) = 0;
                SDSP( 4, 0) = 0;
                SDSP( 4, 1) = 1;
                LDSP.rows( arma::span( 0, 4)) = SDSP;

                int i = it % (1+(Nx-Bs)/Bo);
                int j = it / (1+(Ny-Bs)/Bo);
 
                int x = j;
                int y = i;

                arma::cube refblock = A( arma::span( i, i+Bs-1), arma::span( j, j+Bs-1), arma::span( iARPS1));
                arma::cube newblock = A( arma::span( i, i+Bs-1), arma::span( j, j+Bs-1), arma::span( iARPS2));
                costs( 2) = std::pow( arma::norm( refblock.slice( 0) - newblock.slice( 0), "fro"), 2)/( Bs*Bs);
                chkMat( wind, wind) = 1;

                int stepSize, maxIdx;
                if ( j == 0) {
                    stepSize = 2;
                    maxIdx = 5;
                }
                else {
                    int ytmp = std::abs( motions( 0, it, iARPS3));
                    int xtmp = std::abs( motions( 1, it, iARPS3));
                    stepSize = ( xtmp <= ytmp) ? ytmp : xtmp;
                    if ( ( xtmp == stepSize && ytmp == 0) || ( xtmp == 0 && ytmp == stepSize)) {
                        maxIdx = 5;
                    }
                    else {
                        maxIdx = 6;
                        LDSP( 5, 0) = motions( 1, it, iARPS3);
                        LDSP( 5, 1) = motions( 0, it, iARPS3);
                    }
                }
                LDSP( 0, 0) = 0;
                LDSP( 0, 1) = -stepSize;
                LDSP( 1, 0) = -stepSize;
                LDSP( 1, 1) = 0;
                LDSP( 2, 0) = 0;
                LDSP( 2, 1) = 0;
                LDSP( 3, 0) = stepSize;
                LDSP( 3, 1) = 0;
                LDSP( 4, 0) = 0;
                LDSP( 4, 1) = stepSize;

                // Currently not used, but motion estimation can be predictive
                // if this value is larger than 0!
                double pMot = 0.0;
                // Do the LDSP
                for ( int k = 0; k < maxIdx; k++) {
                    int refBlkVer = y + LDSP( k, 1);
                    int refBlkHor = x + LDSP( k, 0);
                    if ( refBlkHor < 0 || refBlkHor+Bs-1 >= Ny || refBlkVer < 0 || refBlkVer+Bs-1 >= Nx ) { continue; }
                    else if ( k == 2 || stepSize == 0 ) { continue; }
                    else {
                        arma::cube powblock = A( arma::span( refBlkVer, refBlkVer+Bs-1), arma::span( refBlkHor, refBlkHor+Bs-1), arma::span( iARPS2));
                        if ( curFr == 0) {
                            costs( k) = std::pow( arma::norm( refblock.slice( 0) - powblock.slice( 0), "fro"), 2)/( Bs*Bs);
                        }
                        else if ( curFr < 0) {
                            arma::ivec predpos = arma::vectorise( patches( arma::span(), arma::span( it), arma::span( iARPS1)) - motions( arma::span(), arma::span( it), arma::span(iARPS3)));
                            costs( k) = std::pow( arma::norm( refblock.slice( 0) - powblock.slice( 0), "fro"), 2)/( Bs*Bs) + pMot * std::sqrt( std::pow( predpos( 0)-refBlkVer, 2)+std::pow( predpos( 1)-refBlkHor, 2));
                        }
                        else if ( curFr > 0) {
                            arma::ivec predpos = arma::vectorise( patches( arma::span(), arma::span( it), arma::span( iARPS1)) + motions( arma::span(), arma::span( it), arma::span(iARPS3)));
                            costs( k) = std::pow( arma::norm( refblock.slice( 0) - powblock.slice( 0), "fro"), 2)/( Bs*Bs) + pMot * std::sqrt( std::pow( predpos( 0)-refBlkVer, 2)+std::pow( predpos( 1)-refBlkHor, 2));
                        }
                        chkMat( LDSP( k, 1)+wind, LDSP( k, 0)+wind) = 1;
                    }
                }

                arma::uvec point = arma::find( costs == costs.min());
                x += LDSP( point( 0), 0);
                y += LDSP( point( 0), 1);
                double cost = costs.min();
                costs.ones();
                costs *= 1E8;
                costs( 2) = cost;

                // Do the SDSP
                int doneFlag = 0;
                do {
                    for ( int k = 0; k < 5; k++) {
                        int refBlkVer = y + SDSP( k, 1);
                        int refBlkHor = x + SDSP( k, 0);

                        if ( refBlkHor < 0 || refBlkHor+Bs-1 >= Ny || refBlkVer < 0 || refBlkVer+Bs-1 >= Nx ) { continue; }
                        else if ( k == 2 ) { continue; }
                        else if ( refBlkHor < j-wind || refBlkHor > j+wind || refBlkVer < i-wind || refBlkVer > i+wind ) { continue; }
                        else if ( chkMat( y-i+SDSP( k, 1)+wind, x-j+SDSP( k, 0)+wind) == 1) { continue; }
                        else {
                            arma::cube powblock = A( arma::span( refBlkVer, refBlkVer+Bs-1), arma::span( refBlkHor, refBlkHor+Bs-1), arma::span( iARPS2));
                            if ( curFr == 0) {
                                costs( k) = std::pow( arma::norm( refblock.slice( 0) - powblock.slice( 0), "fro"), 2)/( Bs*Bs);
                            }
                            else if ( curFr < 0) {
                                arma::ivec predpos = arma::vectorise( patches( arma::span(), arma::span( it), arma::span( iARPS1)) - motions( arma::span(), arma::span( it), arma::span(iARPS3)));
                                costs( k) = std::pow( arma::norm( refblock.slice( 0) - powblock.slice( 0), "fro"), 2)/( Bs*Bs) + pMot * std::sqrt( std::pow( predpos( 0)-refBlkVer, 2)+std::pow( predpos( 1)-refBlkHor, 2));
                            }
                            else if ( curFr > 0) {
                                arma::ivec predpos = arma::vectorise( patches( arma::span(), arma::span( it), arma::span( iARPS1)) + motions( arma::span(), arma::span( it), arma::span(iARPS3)));
                                costs( k) = std::pow( arma::norm( refblock.slice( 0) - powblock.slice( 0), "fro"), 2)/( Bs*Bs) + pMot * std::sqrt( std::pow( predpos( 0)-refBlkVer, 2)+std::pow( predpos( 1)-refBlkHor, 2));
                            }
                            chkMat( y-i+SDSP( k, 1)+wind, x-j+SDSP( k, 0)+wind) = 1;
                        }
                    }
                    point = arma::find( costs == costs.min());
                    cost = costs.min();

                    if ( point( 0) == 2) {
                        doneFlag = 1;
                    }
                    else {
                        x += SDSP( point( 0), 0);
                        y += SDSP( point( 0), 1);
                        costs.ones();
                        costs *= 1E8;
                        costs( 2) = cost;
                    }
                } while( doneFlag == 0);

                int ystep = y - i;
                int xstep = x - j;

                motions( 0, it, iARPS3) = ystep;
                motions( 1, it, iARPS3) = xstep;
                patches( 0, it, iARPS2) = y;
                patches( 1, it, iARPS2) = x;
            }
            return;
        }
};

#endif
