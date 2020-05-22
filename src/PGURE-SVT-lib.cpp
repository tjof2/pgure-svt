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

extern "C" int PGURESVT(double *X, double *Y, int *dims, int blockSize, int blockOverlap, int T,
                        bool optPGURE, double userLambda, double alpha,
                        double mu, double sigma, int motionWindow, double tol,
                        int MedianSize, double hotPixelThreshold,
                        int numthreads)
{
  pguresvt::print(std::cout,
                  "PGURE-SVT Denoising\n",
                  "Author: Tom Furnival\n",
                  "Email:  tjof2@cam.ac.uk\n");

  int noiseMethod = 4;
  double lambda = (userLambda >= 0.) ? userLambda : 0.;
  int randomSeed = 1;
  bool expWeighting = true;

  int Nx = dims[0];
  int Ny = dims[1];
  int nImages = dims[2];
  double OoNxNyT = 1.0 / (Nx * Ny * T);

  // Copy the image sequence into arma::cube
  // This is the dangerous bit - we want to avoid copying, so set
  // up the Armadillo data matrix to DIRECTLY read from auxiliary
  // memory, but be careful, this is also writable! Remember also
  // that Armadillo stores in column-major order.
  arma::cube noisySeq(X, Nx, Ny, nImages, false, false);
  arma::cube cleanSeq(Y, Nx, Ny, nImages, false, false);

  arma::cube filteredSeq(Nx, Ny, nImages, arma::fill::zeros);

  cleanSeq.zeros();
  filteredSeq.zeros();

  int memsize = 512 * 1024;  // L2 cache size
  int filtsize = MedianSize; // Median filter size in pixels

  // PGURE-SVT timer
  auto t1Start = std::chrono::high_resolution_clock::now();

  // Perform the initial median filtering
  auto &&mfunc = [&](int i) {
    uint16_t *Buffer = new uint16_t[Nx * Ny];
    uint16_t *FilteredBuffer = new uint16_t[Nx * Ny];
    arma::Mat<uint16_t> curSlice = arma::conv_to<arma::Mat<uint16_t>>::from(noisySeq.slice(i).eval());
    inplace_trans(curSlice);
    Buffer = curSlice.memptr();

    ConstantTimeMedianFilter(Buffer, FilteredBuffer, Nx, Ny, Nx, Ny, filtsize, 1, memsize);

    arma::Mat<uint16_t> filtSlice(FilteredBuffer, Nx, Ny);
    inplace_trans(filtSlice);
    filteredSeq.slice(i) = arma::conv_to<arma::mat>::from(filtSlice);
    delete[] Buffer;
    delete[] FilteredBuffer;
  };
  parallel(mfunc, static_cast<uint32_t>(nImages));

  // Initial outlier detection (for hot pixels)
  // using median absolute deviation
  HotPixelFilter(noisySeq, hotPixelThreshold);

  // Loop over time windows
  uint32_t frameWindow = std::floor(T / 2);

  auto &&func = [&, lambda_ = lambda](uint32_t timeIter) {
    auto lambda = lambda_;

    // Extract the subset of the image sequence
    arma::cube u(Nx, Ny, T), uFilter(Nx, Ny, T), v(Nx, Ny, T);
    if (timeIter < frameWindow)
    {
      u = noisySeq.slices(0, 2 * frameWindow);
      uFilter = filteredSeq.slices(0, 2 * frameWindow);
    }
    else if (timeIter >= (nImages - frameWindow))
    {
      u = noisySeq.slices(nImages - 2 * frameWindow - 1, nImages - 1);
      uFilter = filteredSeq.slices(nImages - 2 * frameWindow - 1, nImages - 1);
    }
    else
    {
      u = noisySeq.slices(timeIter - frameWindow, timeIter + frameWindow);
      uFilter = filteredSeq.slices(timeIter - frameWindow, timeIter + frameWindow);
    }

    // Basic sequence normalization
    double inputMax = u.max();
    u /= inputMax;
    uFilter /= uFilter.max();

    // Perform noise estimation
    if (optPGURE)
    {
      NoiseEstimator *noise = new NoiseEstimator;
      noise->Estimate(u, alpha, mu, sigma, 8, noiseMethod, 0);
      delete noise;
    }

    // Perform motion estimation
    MotionEstimator *motion = new MotionEstimator(uFilter, blockSize, timeIter, frameWindow, motionWindow, nImages);
    motion->Estimate();
    arma::icube sequencePatches = motion->GetEstimate();
    delete motion;

    // Perform PGURE optimization
    PGURE *optimizer = new PGURE(u, sequencePatches, alpha, sigma, mu, blockSize, blockOverlap, randomSeed, expWeighting);
    if (optPGURE) // Determine optimum threshold value (max 1000 evaluations)
    {
      lambda = (timeIter == 0) ? arma::accu(u) * OoNxNyT : lambda;
      lambda = optimizer->Optimize(tol, lambda, u.max(), 1000);
      v = optimizer->Reconstruct(lambda);
    }
    else
    {
      v = optimizer->Reconstruct(lambda);
    }
    delete optimizer;

    v *= inputMax; // Rescale back to original range

    if (timeIter < frameWindow) // Place frames back into sequence
    {
      cleanSeq.slice(timeIter) = v.slice(timeIter);
    }
    else if (timeIter >= (nImages - frameWindow))
    {
      int endseqFrame = timeIter - (nImages - T);
      cleanSeq.slice(timeIter) = v.slice(endseqFrame);
    }
    else
    {
      cleanSeq.slice(timeIter) = v.slice(frameWindow);
    }
  };
  parallel(func, static_cast<uint32_t>(nImages));

  // PGURE-SVT timer
  auto t1End = std::chrono::high_resolution_clock::now();
  auto t1Elapsed = std::chrono::duration_cast<std::chrono::microseconds>(t1End - t1Start);
  pguresvt::printFixed(4, "PGURE-SVT:   ", std::setw(10), t1Elapsed.count() * 1E-6, " seconds");

  return 0;
}
