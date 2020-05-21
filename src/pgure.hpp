/***************************************************************************

    Copyright (C) 2015-2020 Tom Furnival

    Optimization of PGURE between noisy and denoised image sequences.
    PGURE is an extension of the formula presented in [1].

    References:
    [1]     "An Unbiased Risk Estimator for Image Denoising in the Presence
            of Mixed Poisson–Gaussian Noise", (2014), Le Montagner, Y et al.
            http://dx.doi.org/10.1109/TIP.2014.2300821

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

#ifndef PGURE_HPP
#define PGURE_HPP

#include <cstdlib>
#include <cstdint>
#include <random>
#include <vector>
#include <armadillo>
#include <nlopt.hpp>

#include "svt.hpp"
#include "utils.hpp"

class PGURE
{
public:
  PGURE(const arma::cube &U,
        const arma::icube patches,
        const double alpha,
        const double mu,
        const double sigma,
        const uint32_t blockSize,
        const uint32_t blockOverlap,
        const int randomSeed,
        const bool expWeighting) : U(U),
                                   patches(patches),
                                   alpha(alpha),
                                   mu(mu),
                                   sigma(sigma),
                                   blockSize(blockSize),
                                   blockOverlap(blockOverlap),
                                   randomSeed(randomSeed),
                                   expWeighting(expWeighting)
  {
    Nx = U.n_rows;
    Ny = U.n_cols;
    T = U.n_slices;

    sigmasq = sigma * sigma;

    Uhat.set_size(Nx, Ny, T);
    U1.set_size(Nx, Ny, T);
    U2p.set_size(Nx, Ny, T);
    U2m.set_size(Nx, Ny, T);

    // Specify perturbations
    eps1 = U.max() * 1E-4;
    eps2 = U.max() * 1E-2;

    // Seed the engine
    if (randomSeed < 0)
    {
      std::uint_least32_t seed;
      pguresvt::sysrandom(&seed, sizeof(seed));
      rand_engine.seed(seed);
    }
    else
    {
      rand_engine.seed(randomSeed);
    }

    // Generate random samples for stochastic evaluation
    delta1.set_size(Nx, Ny, T);
    delta2.set_size(Nx, Ny, T);

    GenerateRandomPerturbations();

    U1 = U + (delta1 * eps1);
    U2p = U + (delta2 * eps2);
    U2m = U - (delta2 * eps2);

    // Initialize the block SVDs
    svt0 = new SVT(patches, Nx, Ny, T, blockSize, blockOverlap, expWeighting);
    svt1 = new SVT(patches, Nx, Ny, T, blockSize, blockOverlap, expWeighting);
    svt2p = new SVT(patches, Nx, Ny, T, blockSize, blockOverlap, expWeighting);
    svt2m = new SVT(patches, Nx, Ny, T, blockSize, blockOverlap, expWeighting);

    svt0->Decompose(U);
    svt1->Decompose(U1);
    svt2p->Decompose(U2p);
    svt2m->Decompose(U2m);
  }

  ~PGURE()
  {
    delete svt0;
    delete svt1;
    delete svt2p;
    delete svt2m;

    Uhat.reset();
    U1.reset();
    U2p.reset();
    U2m.reset();

    delta1.reset();
    delta2.reset();
  }

  arma::cube Reconstruct(const double user_lambda)
  {
    return svt0->Reconstruct(user_lambda);
  }

  double CalculatePGURE(const std::vector<double> &x, std::vector<double> &grad, void *data)
  {
    Uhat = svt0->Reconstruct(x[0]);
    U1 = svt1->Reconstruct(x[0]);
    U2p = svt2p->Reconstruct(x[0]);
    U2m = svt2m->Reconstruct(x[0]);

    // Modified from [1] to include mean/offset
    double pgURE = 1.0 / (Nx * Ny * T) * (arma::accu(arma::square(arma::abs(Uhat - U))) - (alpha + mu) * arma::accu(U) + (2 / eps1 * arma::accu(delta1 % (alpha * U - alpha * mu + sigmasq) % (U1 - Uhat))) - (2 * sigmasq * alpha / (eps2 * eps2) * arma::accu(delta2 % (U2p - 2 * Uhat + U2m))) + (2 * mu * arma::accu(Uhat)) + mu) - sigmasq;

    // Set new lambda
    lambda = x[0];

    return pgURE;
  }

  double Optimize(const double tol, const double start, const double bound, const int eval);

private:
  arma::cube U;
  arma::icube patches;
  double alpha, mu, sigma, sigmasq;
  uint32_t Nx, Ny, T, blockSize, blockOverlap;
  int randomSeed;
  bool expWeighting;

  double eps1, eps2;
  double lambda;

  SVT *svt0, *svt1, *svt2p, *svt2m;

  arma::cube Uhat, U1, U2p, U2m;
  arma::cube delta1, delta2;

  std::mt19937 rand_engine;

  // Reshape to n^2 x T Casorati matrix
  arma::mat CubeFlatten(arma::cube u)
  {
    u.reshape(u.n_rows * u.n_cols, u.n_slices, 1);
    return u.slice(0);
  }

  // Perturbations used in empirical calculation of d'f(y) and d''f(y)
  void GenerateRandomPerturbations()
  {
    std::bernoulli_distribution binary_dist1(0.5);
    delta1.imbue([&]() {
      bool bernRand = binary_dist1(rand_engine);
      if (bernRand == true)
      {
        return -1;
      }
      else
      {
        return 1;
      }
    });

    double kappa = 1.;
    double vP = 0.5 + 0.5 * kappa / std::sqrt(kappa * kappa + 4);
    double vQ = 1 - vP;
    std::bernoulli_distribution binary_dist2(vP);
    delta2.imbue([&]() {
      bool bernRand = binary_dist2(rand_engine);
      if (bernRand == true)
      {
        return -1 * std::sqrt(vQ / vP);
      }
      else
      {
        return std::sqrt(vP / vQ);
      }
    });
    return;
  }
};

// Wrapper for the PGURE optimization function
double obj_wrapper(const std::vector<double> &x, std::vector<double> &grad, void *data)
{
  PGURE *obj = static_cast<PGURE *>(data);
  return obj->CalculatePGURE(x, grad, data);
}

// Optimization function using NLopt and
// BOBYQA gradient-free algorithm
double PGURE::Optimize(const double tol, const double start, const double bound, const int eval)
{
  double minf;
  double startingStep = 0.5 * start;

  std::vector<double> x(1);
  x[0] = start;

  nlopt::opt opt(nlopt::LN_BOBYQA, 1);
  opt.set_min_objective(obj_wrapper, this);
  opt.set_maxeval(eval);
  opt.set_lower_bounds(0.);
  opt.set_upper_bounds(bound);
  opt.set_ftol_rel(tol);
  opt.set_xtol_abs(1E-12);
  opt.set_initial_step(startingStep);

  nlopt::result status = opt.optimize(x, minf);

  if (status <= 0)
  {
    // TODO: Need to implement warnings
  }
  return lambda;
}

#endif
