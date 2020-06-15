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

#ifndef NOISE_HPP
#define NOISE_HPP

#include <cstdlib>
#include <cstdint>
#include <vector>
#include <armadillo>

namespace pguresvt
{
  class NoiseEstimator
  {
  public:
    NoiseEstimator(const uint32_t noiseMethod,
                   const uint32_t size = 8,
                   const uint32_t weightType = 0) : noiseMethod(noiseMethod),
                                                    size(size),
                                                    weightType(weightType)
    {
      // weightType: 0 - "Huber", 1 - "BiSquare"
      laplacian.fill(1.0);
      laplacian *= 0.125;
      laplacian(1, 1) = -1;
      treeDelete.resize(2);
    };

    ~NoiseEstimator(){};

    void Estimate(const arma::cube &input,
                  double &alphaIn,
                  double &muIn,
                  double &sigmaIn)
    {
      alpha = alphaIn;
      mu = muIn;
      sigma = sigmaIn;

      dSi = 0.0; // Set some parameters
      Nx = input.n_cols;
      Ny = input.n_rows;
      T = input.n_slices;

      // Perform quadtree decomposition of frames
      // to generate patches for noise estimation
      uint32_t maxVsize = Nx * Ny;
      arma::vec means = -1 * arma::ones<arma::vec>(T * maxVsize);
      arma::vec vars = -1 * arma::ones<arma::vec>(T * maxVsize);

      for (size_t i = 0; i < T; i++) // Quadtree decomposition of frames
      {
        treeDelete[0] = arma::zeros<arma::umat>(3, 1);
        treeDelete[0](2, 0) = Nx;
        treeDelete[1] = arma::zeros<arma::umat>(0, 0);

        QuadTree(input.slice(i).eval(), 0);

        arma::umat tree = treeDelete[0];
        arma::umat dele = arma::unique(arma::sort(treeDelete[1]));

        for (size_t k = dele.n_elem - 1; k > 0; k--) // Shed parents from quadtree
        {
          tree.shed_col(dele(0, k));
        }

        for (size_t n = 0; n < tree.n_cols; n++) // Extract patches for robust estimation
        {
          int x = tree(0, n);
          int y = tree(1, n);
          int s = tree(2, n);

          // Extract cube from input and reshape to column vector
          arma::cube tmpM = input.subcube(arma::span(x, x + s - 1),
                                          arma::span(y, y + s - 1),
                                          arma::span(i, i));
          tmpM.reshape(s * s, 1, 1);
          arma::vec col = tmpM.tube(0, 0, arma::size(s * s, 1));

          // Add robust patch mean and variance to array
          // Get robust mean estimate
          double meanEst = RobustMeanEstimate(col);

          arma::mat patch(s * s, 1);
          patch.col(0) = col;
          patch.reshape(s, s);
          patch = ConvolveFIR(patch); // Convolve with Laplacian operator
          patch.reshape(s * s, 1);
          col = patch.col(0);

          means(i * maxVsize + n) = meanEst;                    // Robust mean estimate
          vars(i * maxVsize + n) = RobustVarianceEstimate(col); // Robust variance estimate
        }
      }

      // Delete empty values (stored as -1)
      means = means.elem(find(means >= 0.));
      vars = vars.elem(find(vars >= 0.));

      // Get sorted indices of means
      arma::uvec indices = arma::stable_sort_index(means);
      arma::vec robustMeans = means.elem(indices);
      arma::vec robustVars = vars.elem(indices);

      arma::vec alphaBeta = WLSFit(robustMeans, robustVars); // Calculate alpha
      alpha = (alpha >= 0.) ? alpha : alphaBeta(0);          // Don't override user-defined alpha

      switch (noiseMethod) // Calculate mu and sigma
      {
      case 1: // PureDenoise@EPFL
      {

        int L = std::floor(1. * (Nx * Ny / means.n_elem));
        mu = (mu >= 0.) ? mu : ComputeMode(RestrictArray(robustMeans, 0, std::round(0.05 * L)));
        dSi = ComputeMode(RestrictArray(robustVars, 0, std::round(0.05 * L)));
        sigma = (sigma >= 0.) ? sigma : std::sqrt(dSi);
        break;
      }
      case 2: // PureDenoise@EPFL
      {

        mu = (mu >= 0.) ? mu : ComputeMode(robustMeans);
        dSi = ComputeMode(robustVars);
        sigma = (sigma >= 0.) ? sigma : std::sqrt(std::max(dSi, std::max(alphaBeta(1) + alphaBeta(0) * dSi, 0.)));
        break;
      }
      case 3: // tjof2@cam.ac.uk - assumes that the DC offset is the mode of the means
      {
        mu = (mu >= 0.) ? mu : ComputeMode(robustMeans);
        sigma = (sigma >= 0.) ? sigma : std::sqrt(std::abs(alphaBeta(1) + alphaBeta(0) * mu));
        break;
      }
      case 4: // tjof2@cam.ac.uk assumes that the DC offset is the min of the means
      default:
      {
        mu = (mu >= 0.) ? mu : robustMeans(0);
        sigma = (sigma >= 0.) ? sigma : std::sqrt(std::abs(alphaBeta(1) + alphaBeta(0) * mu));
        break;
      }
      }

      alphaIn = alpha;
      muIn = mu;
      sigmaIn = sigma;
      return;
    };

  private:
    uint32_t noiseMethod, size, weightType;
    uint32_t Nx, Ny, T;

    std::vector<arma::umat> treeDelete;

    double alpha, sigma, mu, dSi;

    // F-test lookup tables, supports up to 4096x4096 pixels
    // Default goes wth 2.5%
    const arma::uvec degOfFreePlus1 = {2, 4, 8, 16, 32, 64,
                                       128, 256, 512, 1024, 2048, 4096};
    const arma::vec fTest0100 = {5.39077, 1.97222, 1.38391, 1.17439,
                                 1.08347, 1.04087, 1.02023, 1.01006,
                                 1.00502, 1.00251, 1.00125, 1.00063};
    const arma::vec fTest0050 = {9.27663, 2.40345, 1.51833, 1.22924,
                                 1.10838, 1.05276, 1.02604, 1.01293,
                                 1.00645, 1.00322, 1.00161, 1.00081};
    const arma::vec fTest0025 = {15.4392, 2.86209, 1.64602, 1.27893,
                                 1.13046, 1.06318, 1.03110, 1.01543,
                                 1.00769, 1.00384, 1.00192, 1.00096};
    const arma::vec fTest0010 = {29.4567, 3.52219, 1.80896, 1.33932,
                                 1.15670, 1.07543, 1.03702, 1.01834,
                                 1.00913, 1.00455, 1.00227, 1.00114};

    arma::mat33 laplacian; // Discrete Laplacian operator

    bool SplitBlockQ(const arma::mat &A) // Test to see if a node should be split
    {
      uint32_t N = A.n_cols;
      if (N <= size) // First by compare to minimum allowed size of block
      {
        return false;
      }

      // Calculate pseudo-residuals for estimating variance due to noise
      uint32_t l = 5;
      arma::mat resids(N, N);
      for (size_t x = 0; x < N; x++)
      {
        for (size_t y = 0; y < N; y++)
        {
          int xp = ((x + 1) == N) ? 1 : (x + 1);
          int xm = ((x - 1) < 0) ? (N - 2) : (x - 1);
          int yp = ((y + 1) == N) ? 1 : (y + 1);
          int ym = ((y - 1) < 0) ? (N - 2) : (y - 1);
          resids(y, x) = l * A(y, x) - (A(yp, x) + A(ym, x) + A(y, xm) + A(y, xp));
        }
      }
      resids /= std::sqrt(l * l + l);

      // Perform F-test based on data variance vs. noise variance
      uint32_t R = N * N;
      double OoR = 1.0 / R;
      double OoRm1 = 1.0 / (R - 1);
      arma::vec value;
      double accuZ = arma::accu(A) * OoR;
      double Sz = arma::accu(arma::square(A - accuZ)) * OoRm1;
      double accuR = arma::accu(resids) * OoR;
      double Se = arma::accu(arma::square(resids - accuR)) * OoRm1;
      double stat = (Sz > Se) ? Sz / Se : Se / Sz;

      // Look-up value for F-test (2.5% default)
      value = fTest0025.elem(arma::find(degOfFreePlus1 == N));

      return (stat > value(0));
    };

    double InterquartileDistance(const arma::vec &A)
    {
      arma::vec sorted = sort(A);
      uint32_t N = sorted.n_elem;
      uint32_t m = std::floor((std::floor((N + 1) / 2) + 1) / 2);
      double diq = sorted(N - m - 1) - sorted(m - 1);
      return diq;
    };

    double RobustVarianceEstimate(const arma::vec &A)
    {
      double sig = 1.4826 * arma::median(arma::abs(A - arma::median(A)));
      return sig * sig;
    };

    double RobustMeanEstimate(const arma::vec &A)
    {
      uint32_t I = 1E4;
      uint32_t N = A.n_elem;
      double e = 0.;
      double tol = 1E-6;
      double d = 1.;
      double m = 0.;
      double m0 = 1E12;
      double eps = 1E-12;
      double aux = 0.;

      arma::vec w(N), r(N);
      w.ones();

      for (size_t i = 0; i < I; i++)
      {
        r = w % A;
        m = arma::accu(r);
        aux = arma::accu(w);
        m = (std::abs(aux) < eps) ? m0 : m / aux;
        r = A - m;
        e = arma::mean(arma::abs(r));
        if (std::abs(m0 - m) < tol || e < tol)
        {
          break;
        }
        m0 = m;
        d = InterquartileDistance(r) + eps;
        r *= 1. / d;
        WeightFunction(r, w);
      }
      return m;
    };

    double ComputeMode(const arma::vec &A)
    {
      uint32_t maxCount = 0;
      uint32_t count;
      double maxValue = 0.;
      double M = A.max();
      uint32_t N = A.n_elem;
      double dyn = 1. * N;
      arma::vec a = arma::round(A * dyn / M);

      for (size_t i = 0; i < N; i++)
      {
        count = 0;
        for (size_t j = 0; j < N; j++)
        {
          if (a(j) == a(i))
          {
            count++;
          }
        }
        if (count > maxCount)
        {
          maxCount = count;
          maxValue = a(i);
        }
      }
      maxValue *= M / dyn;
      return maxValue;
    };

    void WeightFunction(const arma::vec &x, arma::vec &w)
    {
      double p, pp;
      switch (weightType)
      {
      case 0:
        p = 0.75;
        for (size_t i = 0; i < x.n_elem; i++)
        {
          w(i) = (std::abs(x(i)) < p) ? 1. : p / std::abs(x(i));
        }
        break;
      case 1:
      default:
        p = 3.5;
        pp = p * p;
        for (size_t i = 0; i < x.n_elem; i++)
        {
          w(i) = (std::abs(x(i)) > p) ? 0. : (pp - x(i) * x(i)) * (pp - x(i) * x(i)) / (pp * pp);
        }
        break;
      }
      return;
    };

    arma::vec WLSFit(const arma::vec &x, const arma::vec &y)
    {
      uint32_t I = 1E4;
      uint32_t N = x.n_elem;
      double e = 0.;
      double tol = 1E-6;
      double d = 1.;
      double a0 = 1E12;
      double b0 = 1E12;
      double eps = 1E-12;
      double aux = 0.;
      double sw2, sw2x, sw2y;

      arma::vec params(2);

      arma::vec w(N), w2(N), w2x(N), w2y(N), xy(N), f(N), r(N);
      w.ones();

      for (size_t i = 0; i < I; i++)
      {
        w2 = w % w;
        w2x = w2 % x;
        w2y = w2 % y;
        xy = x % y;
        xy = w2 % xy;

        sw2 = arma::accu(w2);
        sw2x = arma::accu(w2x);
        sw2y = arma::accu(w2y);

        params(0) = sw2 * arma::accu(xy) - sw2x * sw2y;
        xy = x % x;
        xy = w2 % xy;
        aux = sw2 * arma::accu(xy) - sw2x * sw2x;

        params(0) = (std::abs(aux) < eps) ? a0 : params(0) / aux;
        params(1) = sw2y - params(0) * sw2x;
        params(1) = (std::abs(aux) < eps) ? b0 : params(1) / sw2;

        f = x * params(0) + params(1);

        r = y - f;
        e = arma::mean(arma::abs(r));

        if ((std::abs(a0 - params(0)) < tol && std::abs(b0 - params[1]) < tol) || e < tol)
        {
          break;
        }
        a0 = params[0];
        b0 = params[1];

        d = InterquartileDistance(r) + eps;
        r /= d;
        WeightFunction(r, w);
      }
      return params;
    };

    arma::vec RestrictArray(const arma::vec &a, const uint32_t Is, const uint32_t Ie)
    {
      arma::vec b(Ie - Is + 1);
      for (size_t i = Is; i <= Ie; i++)
      {
        b(i - Is) = a(i);
      }
      return b;
    };

    arma::mat ConvolveFIR(const arma::mat &in)
    {
      uint32_t N = in.n_cols;
      arma::mat out(N, N, arma::fill::zeros);
      arma::mat33 neighbours;
      for (size_t x = 0; x < N; x++)
      {
        for (size_t y = 0; y < N; y++)
        {
          int xp = ((x + 1) == N) ? 1 : (x + 1);
          int xm = ((x - 1) < 0) ? (N - 2) : (x - 1);
          int yp = ((y + 1) == N) ? 1 : (y + 1);
          int ym = ((y - 1) < 0) ? (N - 2) : (y - 1);
          neighbours << in(xm, ym) << in(x, ym) << in(xp, ym) << arma::endr
                     << in(xm, y) << in(x, y) << in(xp, y) << arma::endr
                     << in(xm, yp) << in(x, yp) << in(xp, yp) << arma::endr;

          out(y, x) = arma::accu(neighbours % (-1 * laplacian));
        }
      }
      return out;
    };

    void QuadTree(const arma::mat &A, const uint32_t part) // Recursive quadtree function
    {
      uint32_t i = treeDelete[0](0, part);
      uint32_t j = treeDelete[0](1, part);
      uint32_t s = treeDelete[0](2, part);

      arma::mat patch = A.submat(arma::span(i, i + s - 1), arma::span(j, j + s - 1));

      if (!SplitBlockQ(patch)) // Test if block should be split
      {
        return;
      }

      s /= 2; // If test returns TRUE, split
      uint32_t n = treeDelete[0].n_cols - 1;

      arma::umat::fixed<3, 4> newTreeAdd;
      newTreeAdd << i << i + s << i << i + s << arma::endr
                 << j << j << j + s << j + s << arma::endr
                 << s << s << s << s << arma::endr;
      arma::umat newTree = arma::join_horiz(treeDelete[0], newTreeAdd);

      arma::umat newDeleteAdd(1, 1);
      newDeleteAdd << part << arma::endr;
      arma::umat newDelete = arma::join_horiz(treeDelete[1], newDeleteAdd);

      treeDelete[0].set_size(arma::size(newTree));
      treeDelete[1].set_size(arma::size(newDelete));
      treeDelete[0] = newTree;
      treeDelete[1] = newDelete;

      uint32_t iter = n;
      do
      {
        QuadTree(A, iter);
        iter++;
      } while (iter < n + 4);

      return;
    };
  };

} // namespace pguresvt
#endif
