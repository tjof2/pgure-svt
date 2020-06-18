/**********************
# Author: Tom Furnival
# License: GPLv3
***********************/

#ifndef PGURE_HPP
#define PGURE_HPP

#include <cstdlib>
#include <cstdint>
#include <random>
#include <vector>
#include <armadillo>
#include <nlopt.hpp>

#include "pcg/pcg_random.hpp"
#include "svt.hpp"
#include "utils.hpp"

namespace pguresvt
{
  template <typename T>
  class PGURE
  {
  public:
    PGURE(const arma::Cube<T> &U,
          const arma::icube patches,
          const double alpha,
          const double mu,
          const double sigma,
          const uint32_t blockSize,
          const uint32_t blockOverlap,
          const int64_t randomSeed,
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
      Nt = U.n_slices;
      OoNxNyNt = 1.0 / (Nx * Ny * Nt);
      sigmasq = sigma * sigma;

      Uhat.set_size(Nx, Ny, Nt);
      U1.set_size(Nx, Ny, Nt);
      U2p.set_size(Nx, Ny, Nt);
      U2m.set_size(Nx, Ny, Nt);

      if (randomSeed < 0) // Seed with external entropy from std::random_device
      {
        RNG.seed(pcg_extras::seed_seq_from<std::random_device>());
      }
      else
      {
        RNG.seed(randomSeed);
      }

      eps1 = U.max() * 1E-4; // Specify perturbations
      eps2 = eps1 * 1E2;     // Heuristic: 100 * eps1

      delta1.set_size(Nx, Ny, Nt); // Generate random samples for stochastic evaluation
      delta2.set_size(Nx, Ny, Nt);

      GenerateRandomPerturbations();

      U1 = U + (delta1 * eps1);
      U2p = U + (delta2 * eps2);
      U2m = U - (delta2 * eps2);

      // Initialize the block SVDs
      svt0 = new pguresvt::SVT<T>(patches, Nx, Ny, Nt, blockSize, blockOverlap, expWeighting);
      svt1 = new pguresvt::SVT<T>(patches, Nx, Ny, Nt, blockSize, blockOverlap, expWeighting);
      svt2p = new pguresvt::SVT<T>(patches, Nx, Ny, Nt, blockSize, blockOverlap, expWeighting);
      svt2m = new pguresvt::SVT<T>(patches, Nx, Ny, Nt, blockSize, blockOverlap, expWeighting);

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

    arma::Cube<T> Reconstruct(const double user_lambda)
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
      double pgURE = OoNxNyNt * (arma::accu(arma::square(arma::abs(Uhat - U))) - (alpha + mu) * arma::accu(U) + (2 / eps1 * arma::accu(delta1 % (alpha * U - alpha * mu + sigmasq) % (U1 - Uhat))) - (2 * sigmasq * alpha / (eps2 * eps2) * arma::accu(delta2 % (U2p - 2 * Uhat + U2m))) + (2 * mu * arma::accu(Uhat)) + mu) - sigmasq;

      lambda = x[0]; // Set new lambda

      return pgURE;
    }

    double Optimize(const double tol, const double start, const double bound, const int eval);

  private:
    arma::Cube<T> U;
    arma::icube patches;
    double alpha, mu, sigma, sigmasq;
    uint32_t Nx, Ny, Nt, blockSize, blockOverlap;
    int64_t randomSeed;
    bool expWeighting;

    double OoNxNyNt;
    double eps1, eps2;
    double lambda;

    pguresvt::SVT<T> *svt0, *svt1, *svt2p, *svt2m;

    arma::Cube<T> Uhat, U1, U2p, U2m;
    arma::icube delta1;
    arma::Cube<T> delta2;

    pcg64 RNG;

    arma::Mat<T> CubeFlatten(arma::Cube<T> u) // Reshape to Casorati matrix
    {
      u.reshape(u.n_rows * u.n_cols, u.n_slices, 1);
      return u.slice(0);
    }

    void GenerateRandomPerturbations() // Perturbations used in empirical calculation of d'f(y) and d''f(y)
    {
      auto bernoulliFunc = [&](std::bernoulli_distribution &dist, auto value1, auto value2) {
        return (dist(RNG)) ? value1 : value2;
      };

      double kappa = 1.;
      double vP = 0.5 + 0.5 * kappa / std::sqrt(kappa * kappa + 4);
      double vQ = 1 - vP;
      double vQvP = std::sqrt(vQ / vP);
      double vPvQ = std::sqrt(vP / vQ);

      std::bernoulli_distribution binary_dist1(0.5);
      std::bernoulli_distribution binary_dist2(vP);

      delta1.imbue([&]() { return bernoulliFunc(binary_dist1, -1, 1); });
      delta2.imbue([&]() { return static_cast<T>(bernoulliFunc(binary_dist2, -1 * vQvP, vPvQ)); });

      return;
    }
  };

  template <typename T>
  double obj_wrapper(const std::vector<double> &x, std::vector<double> &grad, void *data) // Wrapper for optimization function
  {
    PGURE<T> *obj = static_cast<PGURE<T> *>(data);
    return obj->CalculatePGURE(x, grad, data);
  }

  template <typename T>
  double PGURE<T>::Optimize(const double tol, const double start, const double bound, const int eval) // Optimization function using BOBYQA gradient-free algorithm
  {
    double minf;
    double startingStep = 0.5 * start;

    std::vector<double> x(1);
    x[0] = start;

    nlopt::opt opt(nlopt::LN_BOBYQA, 1);
    opt.set_min_objective(obj_wrapper<T>, this);
    opt.set_maxeval(eval);
    opt.set_lower_bounds(0.);
    opt.set_upper_bounds(bound);
    opt.set_ftol_rel(tol);
    opt.set_xtol_abs(1E-12);
    opt.set_initial_step(startingStep);

    nlopt::result status = opt.optimize(x, minf);

    if (status == 5)
    {
      pguresvt::Print(std::cerr, "WARNING: optimization terminated after max_eval (", eval, ") was reached.\n",
                      "Consider increasing the max_eval or increasing the convergence tolerance (tol=", tol, ").");
    }
    return lambda;
  }

} // namespace pguresvt
#endif
