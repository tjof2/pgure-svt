/***************************************************************************

    Copyright (C) 2015-16 Tom Furnival

    Optimization of PGURE between noisy and denoised image sequences.
    PGURE is an extension of the formula presented in [1].

    References:
    [1]     "An Unbiased Risk Estimator for Image Denoising in the Presence
            of Mixed Poisson–Gaussian Noise", (2014), Le Montagner, Y et al.
            http://dx.doi.org/10.1109/TIP.2014.2300821

***************************************************************************/

#ifndef PGURE_H
#define PGURE_H

// C++ headers
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <random>
#include <vector>

// Armadillo library
#include <armadillo>

// NLopt library
#include <nlopt.hpp>

// Own header
#include "svt.hpp"

class PGURE {
 public:
        PGURE() {
            svt0 = new SVT;
            svt1 = new SVT;
            svt2p = new SVT;
            svt2m = new SVT;
        }
        ~PGURE() {
            delete svt0;
            delete svt1;
            delete svt2p;
            delete svt2m;
        }

        void Initialize(const arma::cube &u,
                        const arma::icube patches,
                        int blocksize,
                        int blockoverlap,
                        double alphaIn,
                        double muIn,
                        double sigmaIn) {
            U = u;

            Nx = u.n_rows;
            Ny = u.n_cols;
            T = u.n_slices;
            Bs = blocksize;
            Bo = blockoverlap;

            alpha = alphaIn;
            mu = muIn;
            sigma = sigmaIn;

            Uhat.set_size(Nx, Ny, T);
            U1.set_size(Nx, Ny, T);
            U2p.set_size(Nx, Ny, T);
            U2m.set_size(Nx, Ny, T);

            // Specify perturbations
            eps1 = U.max() * 1E-4;
            eps2 = U.max() * 1E-2;

            // Generate random samples for stochastic evaluation
            delta1.set_size(Nx, Ny, T);
            delta2.set_size(Nx, Ny, T);
            GenerateRandomPerturbations();
            U1 = U + (delta1 * eps1);
            U2p = U + (delta2 * eps2);
            U2m = U - (delta2 * eps2);

            // Initialize the block SVDs
            svt0->Initialize(patches, Nx, Ny, T, Bs, Bo);
            svt1->Initialize(patches, Nx, Ny, T, Bs, Bo);
            svt2p->Initialize(patches, Nx, Ny, T, Bs, Bo);
            svt2m->Initialize(patches, Nx, Ny, T, Bs, Bo);

            // Initialize the block SVDs
            svt0->Decompose(U);
            svt1->Decompose(U1);
            svt2p->Decompose(U2p);
            svt2m->Decompose(U2m);

            return;
        }

        arma::cube Reconstruct(double user_lambda) {
            return svt0->Reconstruct(user_lambda);
        }

        double CalculatePGURE(const std::vector<double> &x,
                              std::vector<double> &grad,
                              void *data) {
            Uhat = svt0->Reconstruct(x[0]);
            U1 = svt1->Reconstruct(x[0]);
            U2p = svt2p->Reconstruct(x[0]);
            U2m = svt2m->Reconstruct(x[0]);

            // Modified from [1] to include mean/offset
            int NxNyT = Nx*Ny*T;
            double pgURE;
            pgURE = arma::accu(arma::square(arma::abs(Uhat - U)))/NxNyT
                - (alpha + mu) * arma::accu(U)/NxNyT
                + 2/eps1 * arma::accu(delta1
                            % (alpha * U - alpha*mu + sigma*sigma)
                            % (U1 - Uhat))/NxNyT
                - 2*sigma*sigma*alpha/(eps2*eps2)
                            * arma::accu(delta2 % (U2p - 2*Uhat + U2m))/NxNyT
                + 2 * mu * arma::accu(Uhat)/NxNyT
                + mu/NxNyT
                - sigma*sigma;

            // Set new lambda
            lambda = x[0];

            return pgURE;
        }

        double Optimize(double tol,
                        double start,
                        double bound,
                        int eval);

 private:
        int Nx, Ny, T, Bs, Bo;
        double eps1, eps2;
        double lambda;
        double alpha, mu, sigma;

        SVT *svt0, *svt1, *svt2p, *svt2m;

        arma::cube U;
        arma::cube Uhat, U1, U2p, U2m;
        arma::cube delta1, delta2;

        std::mt19937 rand_engine;

        // Reshape to n^2 x T Casorati matrix
        arma::mat CubeFlatten(arma::cube u) {
            u.reshape(u.n_rows*u.n_cols, u.n_slices, 1);
            return u.slice(0);
        }

        // Perturbations used in empirical calculation of d'f(y) and d''f(y)
        void GenerateRandomPerturbations() {
            std::bernoulli_distribution binary_dist1(0.5);
            delta1.imbue([&]() {
                bool bernRand = binary_dist1(rand_engine);
                if (bernRand == true) {
                    return -1;
                } else {
                    return 1;
                }
            });

            double kappa = 1.;
            double vP = (1/2)+(kappa/2)/std::sqrt(kappa*kappa+4);
            double vQ = 1 - vP;
            std::bernoulli_distribution binary_dist2(vP);
            delta2.imbue([&]() {
                bool bernRand = binary_dist2(rand_engine);
                if (bernRand == true) {
                    return -1 * std::sqrt(vQ/vP);
                } else {
                    return std::sqrt(vP/vQ);
                }
            });
            return;
        }
};

// Wrapper for the PGURE optimization function
double obj_wrapper(const std::vector<double> &x,
                   std::vector<double> &grad,
                   void *data) {
  PGURE *obj = static_cast<PGURE *>(data);
  return obj->CalculatePGURE(x, grad, data);
}

// Optimization function using NLopt and
// BOBYQA gradient-free algorithm
double PGURE::Optimize(double tol,
                       double start,
                       double bound,
                       int eval) {
    double startingStep = start / 2;

    // Optimize PGURE
    nlopt::opt opt(nlopt::LN_BOBYQA, 1);
    opt.set_min_objective(obj_wrapper, this);
    opt.set_maxeval(eval);
    opt.set_lower_bounds(0.);
    opt.set_upper_bounds(bound);
    opt.set_ftol_rel(tol);
    opt.set_xtol_abs(1E-12);
    opt.set_initial_step(startingStep);

    std::vector<double> x(1);
    x[0] = start;

    // Objective value
    double minf;

    // Run the optimizer
    nlopt::result status = opt.optimize(x, minf);

    if (status <= 0) {
        // TODO(tjof2): Need to implement warnings
    }
    return lambda;
}

#endif
