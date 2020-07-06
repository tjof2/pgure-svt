/**********************
# Author: Tom Furnival
# License: GPLv3
***********************/

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
                  arma::Mat<T2> &estimates,
                  const arma::Cube<T1> &X,
                  const uint32_t trajLength,
                  const uint32_t blockSize,
                  const uint32_t blockOverlap,
                  const uint32_t motionWindow,
                  const int64_t medianSize,
                  const uint32_t noiseMethod,
                  const uint32_t maxIter,
                  const int64_t nJobs,
                  const int64_t randomSeed,
                  const bool optimizePGURE,
                  const bool expWeighting,
                  const bool motionEstimation,
                  const double lambdaEst,
                  const double alphaEst,
                  const double muEst,
                  const double sigmaEst,
                  const double tol,
                  const bool verbose = false)
{
  uint32_t Nx = X.n_cols;
  uint32_t Ny = X.n_rows;
  uint32_t Nimgs = X.n_slices;

  Y.set_size(arma::size(X)); // Set output sequence size
  Y.zeros();

  estimates.set_size(Nimgs, 4); // Set results vector size
  estimates.zeros();

  if (verbose)
  {
    pguresvt::PrintFixed(1, "Input type=", typeid(T1).name(), ", Output type=", typeid(T2).name());
    pguresvt::PrintFixed(1, "Dimensions=", arma::size(X));
  }

  // Check trajectory length against block size
  uint32_t Nt = (blockSize * blockSize < trajLength) ? (blockSize * blockSize) - 1 : trajLength;
  uint32_t frameWindow = std::floor(Nt / 2);

  double OoNxNyNt = 1.0 / (Nx * Ny * Nt);

  double lambda0 = (lambdaEst >= 0.0) ? lambdaEst : -1.0;
  double alpha0 = (alphaEst >= 0.0) ? alphaEst : -1.0;
  double mu0 = (muEst >= 0.0) ? muEst : -1.0;
  double sigma0 = (sigmaEst >= 0.0) ? sigmaEst : -1.0;

  arma::Cube<T1> Z(Nx, Ny, Nimgs); // Median-filtered sequence
  int memSize = 1024 * 1024;       // assumes 1024 KB L2 cache size

  auto &&medianFunc = [&](uint32_t i) {
    uint16_t *zBuffer = new uint16_t[Nx * Ny];
    ConstantTimeMedianFilter(X.slice(i).memptr(), zBuffer, Nx, Ny, Nx, Nx, medianSize, 1, memSize);

    arma::Mat<uint16_t> zSlice(zBuffer, Nx, Ny);
    Z.slice(i) = zSlice;
    delete zBuffer;
  };

  if (medianSize > 0) // Apply median filter to the images prior to motion estimation
  {
    pguresvt::parallel(medianFunc, static_cast<uint32_t>(0), static_cast<uint32_t>(Nimgs), nJobs);
  }
  else // Motion estimation is applied to the unfiltered images
  {
    Z = X;
  }

  auto &&pgureFunc = [&, lambda_ = lambda0, alpha_ = alpha0, mu_ = mu0, sigma_ = sigma0](uint32_t timeIter) {
    auto lambda = lambda_;
    auto alpha = alpha_;
    auto mu = mu_;
    auto sigma = sigma_;

    arma::Cube<T2> u = arma::Cube<T2>(Nx, Ny, Nt, arma::fill::zeros);
    arma::Cube<T2> v = arma::Cube<T2>(Nx, Ny, Nt, arma::fill::zeros);
    arma::Cube<T2> w = arma::Cube<T2>(Nx, Ny, Nt, arma::fill::zeros);

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

    if (optimizePGURE) // Only estimate noise if optimizing PGURE threshold
    {
      pguresvt::NoiseEstimator *noise = new pguresvt::NoiseEstimator(noiseMethod);
      noise->Estimate(u, alpha, mu, sigma);
      delete noise;
    }

    pguresvt::MotionEstimator<T2> *motion = new pguresvt::MotionEstimator<T2>(w, blockSize, timeIter, frameWindow, motionWindow, Nimgs);
    arma::icube p = motion->Estimate(motionEstimation);
    delete motion;

    pguresvt::PGURE<T2> *optimizer = new pguresvt::PGURE<T2>(u, p, alpha, sigma, mu, blockSize, blockOverlap, randomSeed, expWeighting, optimizePGURE);

    if (optimizePGURE) // Determine optimum threshold value
    {
      double startPoint, upperBound;

      startPoint = (lambda >= 0.0) ? lambda : arma::accu(u) * OoNxNyNt; // User can provide initial guess
      startPoint = std::max(0.0, startPoint);                           // Initial guess for lambda should be positive
      upperBound = std::max(100.0, startPoint);                         // Image max is 1.0, so bound lambda a little higher than this

      lambda = optimizer->Optimize(tol, startPoint, upperBound, maxIter);
    }

    v = optimizer->Reconstruct(lambda); // Reconstruct the sequence
    v *= uMax;                          // Rescale back to original range
    delete optimizer;

    estimates(timeIter, 0) = lambda; // Update estimates
    estimates(timeIter, 1) = alpha;
    estimates(timeIter, 2) = mu;
    estimates(timeIter, 3) = sigma;

    if (timeIter < frameWindow) // Place frames back into sequence
    {
      Y.slice(timeIter) = v.slice(timeIter);
    }
    else if (timeIter >= (Nimgs - frameWindow))
    {
      Y.slice(timeIter) = v.slice(timeIter - (Nimgs - Nt));
    }
    else
    {
      Y.slice(timeIter) = v.slice(frameWindow);
    }
  };

  pguresvt::parallel(pgureFunc, static_cast<uint32_t>(0), static_cast<uint32_t>(Nimgs), nJobs); // Apply over the time windows

  return 0;
}
