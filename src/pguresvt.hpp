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

#include <cstdarg>
#include <cstdlib>
#include <cstdint>
#include <armadillo>

#include "arps.hpp"
#include "medfilter.hpp"
#include "noise.hpp"
#include "pgure.hpp"
#include "utils.hpp"

template <typename T1, typename T2>
uint32_t PGURESVT(arma::Cube<T2> &Y,
                  const arma::Cube<T1> &X,
                  const uint32_t trajLength,
                  const uint32_t blockSize,
                  const uint32_t blockOverlap,
                  const uint32_t motionWindow,
                  const uint32_t medianSize,
                  const uint32_t noiseMethod,
                  const uint32_t maxIter,
                  const int64_t nJobs,
                  const int64_t randomSeed,
                  const bool optPGURE,
                  const bool expWeighting,
                  const bool motionEstimation,
                  const double lambdaEst,
                  const double alphaEst,
                  const double muEst,
                  const double sigmaEst,
                  const double tol)
{
  uint32_t Nx = X.n_cols;
  uint32_t Ny = X.n_rows;
  uint32_t Nimgs = X.n_slices;

  Y.set_size(arma::size(X)); // Set output sequence size
  Y.zeros();

  //pguresvt::PrintFixed(1, "Input type=", typeid(T1).name(), ", Output type=", typeid(T2).name());
  //pguresvt::PrintFixed(1, "Dimensions=", arma::size(X));

  // Check trajectory length against block size
  uint32_t Nt = (blockSize * blockSize < trajLength) ? (blockSize * blockSize) - 1 : trajLength;
  uint32_t frameWindow = std::floor(Nt / 2);

  double OoNxNyNt = 1.0 / (Nx * Ny * Nt);

  double lambda0 = (lambdaEst >= 0.) ? lambdaEst : -1.;
  double alpha0 = (alphaEst >= 0.) ? alphaEst : -1.;
  double mu0 = (muEst >= 0.) ? muEst : -1.;
  double sigma0 = (sigmaEst >= 0.) ? sigmaEst : -1.;

  arma::Cube<T1> Z(Nx, Ny, Nimgs); // Median-filtered sequence
  int memSize = 512 * 1024;        // L2 cache size

  auto &&medianFunc = [&](uint32_t i) {
    uint16_t *zBuffer = new uint16_t[Nx * Ny];
    arma::Mat<uint16_t> zSlice(zBuffer, Nx, Ny);

    ConstantTimeMedianFilter(X.slice(i).memptr(), zBuffer, Nx, Ny, Nx, Nx, medianSize, 1, memSize);
    Z.slice(i) = zSlice;
  };

  pguresvt::parallel(medianFunc, static_cast<uint32_t>(0), static_cast<uint32_t>(Nimgs), nJobs); // Apply over the images

  auto &&pgureFunc = [&, lambda_ = lambda0, alpha_ = alpha0, mu_ = mu0, sigma_ = sigma0](uint32_t timeIter) {
    auto lambda = lambda_;
    auto alpha = alpha_;
    auto mu = mu_;
    auto sigma = sigma_;

    arma::Cube<T2> u = arma::Cube<T2>(Nx, Ny, Nt, arma::fill::zeros);
    arma::Cube<T2> v = arma::Cube<T2>(Nx, Ny, Nt, arma::fill::zeros);
    arma::Cube<T2> w = arma::Cube<T2>(Nx, Ny, Nt, arma::fill::zeros);
    arma::icube p = arma::icube(Nx, Ny, Nt, arma::fill::zeros);

    if (timeIter < frameWindow) // Extract the subset of the image sequence
    {
      u = arma::conv_to<arma::Cube<T2>>::from(X.slices(0, 2 * frameWindow));
      w = arma::conv_to<arma::Cube<T2>>::from(Z.slices(0, 2 * frameWindow));
    }
    else if (timeIter >= (Nimgs - frameWindow))
    {
      u = arma::conv_to<arma::Cube<T2>>::from(X.slices(Nimgs - 2 * frameWindow - 1, Nimgs - 1));
      w = arma::conv_to<arma::Cube<T2>>::from(Z.slices(Nimgs - 2 * frameWindow - 1, Nimgs - 1));
    }
    else
    {
      u = arma::conv_to<arma::Cube<T2>>::from(X.slices(timeIter - frameWindow, timeIter + frameWindow));
      w = arma::conv_to<arma::Cube<T2>>::from(Z.slices(timeIter - frameWindow, timeIter + frameWindow));
    }

    double uMax = u.max(); // Sequence normalization
    double wMax = w.max(); // ARPS relies on reasonably well-scaled data

    u /= uMax;
    w /= wMax;

    if (optPGURE) // Only estimate noise if optimizing PGURE threshold
    {
      pguresvt::NoiseEstimator *noise = new pguresvt::NoiseEstimator(noiseMethod);
      noise->Estimate(u, alpha, mu, sigma);
      delete noise;
    }

    pguresvt::MotionEstimator<T2> *motion = new pguresvt::MotionEstimator<T2>(w, blockSize, timeIter, frameWindow, motionWindow, Nimgs);
    p = motion->Estimate(motionEstimation);
    delete motion;

    pguresvt::PGURE<T2> *optimizer = new pguresvt::PGURE<T2>(u, p, alpha, sigma, mu, blockSize, blockOverlap, randomSeed, expWeighting);

    if (optPGURE) // Determine optimum threshold value
    {
      double upperBound = 1.0; // Image max is 1.0, so allow up to this
      lambda = optimizer->Optimize(tol, arma::accu(u) * OoNxNyNt, upperBound, maxIter);
    }

    v = optimizer->Reconstruct(lambda); // Reconstruct the sequence
    v *= uMax;                          // Rescale back to original range
    delete optimizer;

    if (timeIter < frameWindow) // Place frames back into sequence
    {
      Y.slice(timeIter) = v.slice(timeIter);
    }
    else if (timeIter >= (Nimgs - frameWindow))
    {
      int endseqFrame = timeIter - (Nimgs - Nt);
      Y.slice(timeIter) = v.slice(endseqFrame);
    }
    else
    {
      Y.slice(timeIter) = v.slice(frameWindow);
    }
  };

  pguresvt::parallel(pgureFunc, static_cast<uint32_t>(0), static_cast<uint32_t>(Nimgs), nJobs); // Apply over the time windows

  return 0;
}
