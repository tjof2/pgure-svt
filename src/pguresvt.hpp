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
#include "params.hpp"
#include "noise.hpp"
#include "pgure.hpp"
#include "parallel.hpp"
#include "utils.hpp"

template <typename T1, typename T2>
uint32_t PGURESVT(arma::Cube<T2> &Y,
                  const arma::Cube<T1> &X,
                  const arma::Cube<T2> &Z,
                  const uint32_t trajLength,
                  const uint32_t blockSize,
                  const uint32_t blockOverlap,
                  const uint32_t motionWindow,
                  const uint32_t noiseMethod,
                  const uint32_t numThreads,
                  const uint32_t maxIter,
                  const bool optPGURE,
                  const bool expWeighting,
                  const double lambdaEst,
                  const double alphaEst,
                  const double muEst,
                  const double sigmaEst,
                  const double tol,
                  const int randomSeed)
{
  uint32_t Nx = X.n_cols;
  uint32_t Ny = X.n_rows;
  uint32_t Nimgs = X.n_slices;

  // pguresvt::printFixed(1, "T1=", typeid(T1).name(), ", T2=", typeid(T2).name());

  // Check trajectory length against block size
  uint32_t Nt = (blockSize * blockSize < trajLength) ? (blockSize * blockSize) - 1 : trajLength;
  uint32_t frameWindow = std::floor(Nt / 2);

  double OoNxNyNt = 1.0 / (Nx * Ny * Nt);

  double lambda = (lambdaEst >= 0.) ? lambdaEst : -1.;
  double alpha = (alphaEst >= 0.) ? alphaEst : -1.;
  double mu = (muEst >= 0.) ? muEst : -1.;
  double sigma = (sigmaEst >= 0.) ? sigmaEst : -1.;

  auto &&func = [&, lambda_ = lambda](uint32_t timeIter) {
    auto lambda = lambda_;

    arma::Cube<T2> u(Nx, Ny, Nt); // Extract the subset of the image sequence
    arma::Cube<T2> v(Nx, Ny, Nt);
    arma::Cube<T2> w(Nx, Ny, Nt);

    if (timeIter < frameWindow)
    {
      u = X.slices(0, 2 * frameWindow);
      w = Z.slices(0, 2 * frameWindow);
    }
    else if (timeIter >= (Nimgs - frameWindow))
    {
      u = X.slices(Nimgs - 2 * frameWindow - 1, Nimgs - 1);
      w = Z.slices(Nimgs - 2 * frameWindow - 1, Nimgs - 1);
    }
    else
    {
      u = X.slices(timeIter - frameWindow, timeIter + frameWindow);
      w = Z.slices(timeIter - frameWindow, timeIter + frameWindow);
    }

    pguresvt::printFixed(1, "Sliced:", timeIter, ", Nt=", Nt, ", Nimgs=", Nimgs);

    double uMax = u.max(); // Basic sequence normalization
    u /= uMax;

    if (optPGURE) // Perform noise estimation
    {
      NoiseEstimator *noise = new NoiseEstimator;
      noise->Estimate(u, alpha, mu, sigma, 8, noiseMethod, 0);
      delete noise;
    }

    pguresvt::printFixed(1, "Noise:", timeIter, ", Nt=", Nt, ", Nimgs=", Nimgs);

    MotionEstimator<T2> *motion = new MotionEstimator<T2>(w, blockSize, timeIter, frameWindow, motionWindow, Nimgs);

    motion->Estimate(); // Perform motion estimation
    arma::icube p = motion->GetEstimate();
    delete motion;

    pguresvt::printFixed(1, "Motion:", timeIter, ", Nt=", Nt, ", Nimgs=", Nimgs);

    PGURE<T2> *optimizer = new PGURE<T2>(u, p, alpha, sigma, mu, blockSize, blockOverlap, randomSeed, expWeighting);

    if (optPGURE) // Determine optimum threshold value
    {
      double upperBound = 1.0; // Image max is 1.0, so allow up to this
      lambda = optimizer->Optimize(tol, arma::accu(u) * OoNxNyNt, upperBound, maxIter);
    }

    pguresvt::printFixed(1, "Optimized:", timeIter, ", Nt=", Nt, ", Nimgs=", Nimgs);

    v = optimizer->Reconstruct(lambda); // Reconstruct the sequence
    v *= uMax;                          // Rescale back to original range
    delete optimizer;

    pguresvt::printFixed(1, "Reconstructed:", timeIter, ", Nt=", Nt, ", Nimgs=", Nimgs);

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

    pguresvt::printFixed(1, "Rebuilt:", timeIter, ", Nt=", Nt, ", Nimgs=", Nimgs);
  };

  parallel(func, static_cast<uint32_t>(Nimgs)); // Apply over the time windows

  return 0;
}
