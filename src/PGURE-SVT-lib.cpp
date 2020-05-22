/***************************************************************************

  Copyright (C) 2015-2020 Tom Furnival

  This file is part of  PGURE-SVT.

  PGURE-SVT is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  PGURE-SVT is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with PGURE-SVT. If not, see <http://www.gnu.org/licenses/>.

***************************************************************************/

#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <cstdarg>
#include <stdexcept>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <vector>
#include <armadillo>

extern "C"
{
#include "medfilter.h"
}

#include "arps.hpp"
#include "hotpixel.hpp"
#include "params.hpp"
#include "noise.hpp"
#include "pgure.hpp"
#include "parallel.hpp"
#include "utils.hpp"

// Main program
extern "C" int PGURESVT(double *X, double *Y, int *dims, int Bs, int Bo, int T,
                        bool pgureOpt, double userLambda, double alpha,
                        double mu, double sigma, int motionWindow, double tol,
                        int MedianSize, double hotpixelthreshold,
                        int numthreads)
{

  // Overall program timer
  auto overallstart = std::chrono::high_resolution_clock::now();

  // Print program header
  std::cout << std::endl
            << "PGURE-SVT Denoising" << std::endl
            << "Author: Tom Furnival" << std::endl
            << "Email:  tjof2@cam.ac.uk" << std::endl
            << std::endl;

  int NoiseMethod = 4;
  double lambda = (userLambda >= 0.) ? userLambda : 0.;

  int Nx = dims[0];
  int Ny = dims[1];
  int nImages = dims[2];

  // Copy the image sequence into the a cube
  // This is the dangerous bit - we want to avoid copying, so set
  // up the Armadillo data matrix to DIRECTLY read from auxiliary
  // memory, but be careful, this is also writable! Remember also
  // that Armadillo stores in column-major order.
  arma::cube noisySeq(X, Nx, Ny, nImages, false, false);
  arma::cube cleanSeq(Y, Nx, Ny, nImages, false, false);

  // Initialize the filtered sequence
  arma::cube filteredSeq(Nx, Ny, nImages, arma::fill::zeros);

  cleanSeq.zeros();
  filteredSeq.zeros();

  // Parameters for median filter
  int memsize = 512 * 1024;  // L2 cache size
  int filtsize = MedianSize; // Median filter size in pixels

  // Perform the initial median filtering
  auto &&mfunc = [&](int i) {
    unsigned short *Buffer = new unsigned short[Nx * Ny];
    unsigned short *FilteredBuffer = new unsigned short[Nx * Ny];
    arma::Mat<unsigned short> curslice =
        arma::conv_to<arma::Mat<unsigned short>>::from(
            noisySeq.slice(i).eval());
    inplace_trans(curslice);
    Buffer = curslice.memptr();
    ConstantTimeMedianFilter(Buffer, FilteredBuffer, Nx, Ny, Nx, Ny, filtsize,
                             1, memsize);
    arma::Mat<unsigned short> filslice(FilteredBuffer, Nx, Ny);
    inplace_trans(filslice);
    filteredSeq.slice(i) = arma::conv_to<arma::mat>::from(filslice);
    delete[] Buffer;
    delete[] FilteredBuffer;
  };
  parallel(mfunc, static_cast<unsigned long long>(nImages));

  // Initial outlier detection (for hot pixels)
  // using median absolute deviation
  pguresvt::printFixed(3, "Applying hot-pixel detector with threshold: ", hotpixelthreshold, " * MAD");
  HotPixelFilter(noisySeq, hotpixelthreshold);

  // Print table headings
  int ww = 10;
  std::cout << std::endl;
  std::cout << std::right << std::setw(5 * ww + 5)
            << std::string(5 * ww + 5, '-') << std::endl;
  std::cout << std::setw(5) << "Frame" << std::setw(ww) << "Gain"
            << std::setw(ww) << "Offset" << std::setw(ww) << "Sigma"
            << std::setw(ww) << "Lambda" << std::setw(ww) << "Time (s)"
            << std::endl;
  std::cout << std::setw(5 * ww + 5) << std::string(5 * ww + 5, '-')
            << std::endl;

  // Loop over time windows
  int framewindow = std::floor(T / 2);

  auto &&func = [&, lambda_ = lambda](int timeiter) {
    // Extract the subset of the image sequence
    arma::cube u(Nx, Ny, T), ufilter(Nx, Ny, T), v(Nx, Ny, T);
    if (timeiter < framewindow)
    {
      u = noisySeq.slices(0, 2 * framewindow);
      ufilter = filteredSeq.slices(0, 2 * framewindow);
    }
    else if (timeiter >= (nImages - framewindow))
    {
      u = noisySeq.slices(nImages - 2 * framewindow - 1,
                          nImages - 1);
      ufilter = filteredSeq.slices(nImages - 2 * framewindow - 1,
                                   nImages - 1);
    }
    else
    {
      u = noisySeq.slices(timeiter - framewindow, timeiter + framewindow);
      ufilter = filteredSeq.slices(timeiter - framewindow,
                                   timeiter + framewindow);
    }

    // Basic sequence normalization
    double inputMax = u.max();
    u /= inputMax;
    ufilter /= ufilter.max();

    // Perform noise estimation
    if (pgureOpt)
    {
      NoiseEstimator *noise = new NoiseEstimator;
      noise->Estimate(u, alpha, mu, sigma, 4, NoiseMethod, 0);
      delete noise;
    }

    // Perform motion estimation
    MotionEstimator *motion = new MotionEstimator;
    motion->Estimate(ufilter, timeiter, framewindow, nImages, Bs, motionWindow);
    arma::icube sequencePatches = motion->GetEstimate();
    delete motion;

    // Perform PGURE optimization
    int randomSeed = -1;
    bool expWeighting = true;
    PGURE *optimizer = new PGURE(u, sequencePatches, alpha, sigma, mu, Bs, Bo, randomSeed, expWeighting);
    // Determine optimum threshold value (max 1000 evaluations)
    if (pgureOpt)
    {
      auto lambda = lambda_;
      lambda = (timeiter == 0) ? arma::accu(u) / (Nx * Ny * T) : lambda;
      lambda = optimizer->Optimize(tol, lambda, u.max(), 1E3);
      v = optimizer->Reconstruct(lambda);
    }
    else
    {
      v = optimizer->Reconstruct(userLambda);
    }
    delete optimizer;

    // Rescale back to original range
    v *= inputMax;

    // Place frames back into sequence
    if (timeiter < framewindow)
    {
      cleanSeq.slice(timeiter) = v.slice(timeiter);
    }
    else if (timeiter >= (nImages - framewindow))
    {
      int endseqFrame = timeiter - (nImages - T);
      cleanSeq.slice(timeiter) = v.slice(endseqFrame);
    }
    else
    {
      cleanSeq.slice(timeiter) = v.slice(framewindow);
    }
  };
  parallel(func, static_cast<unsigned long long>(nImages));

  // Finish the table off
  std::cout << std::setw(5 * ww + 5) << std::string(5 * ww + 5, '-')
            << std::endl
            << std::endl;

  // Overall program timer
  auto overallend = std::chrono::high_resolution_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
      overallend - overallstart);
  std::cout << "Total time: " << std::setprecision(5) << (elapsed.count() / 1E6)
            << " seconds" << std::endl
            << std::endl;

  return 0;
}
