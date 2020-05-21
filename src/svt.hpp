/***************************************************************************

    Copyright (C) 2015-2020 Tom Furnival

    SVT calculation based on [1].

    References:
    [1]     "Unbiased Risk Estimates for Singular Value Thresholding and
            Spectral Estimators", (2013), Candes, EJ et al.
            http://dx.doi.org/10.1109/TSP.2013.2270464

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

#ifndef SVT_HPP
#define SVT_HPP

#include <cstdlib>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <vector>
#include <armadillo>

#include "utils.hpp"

class SVT
{
public:
  SVT() {}
  ~SVT() {}

  // Allocate memory for SVD step
  void Initialize(const arma::icube &sequencePatches,
                  const uint32_t w, const uint32_t h, const uint32_t l,
                  const uint32_t blocksize, const uint32_t blockoverlap)
  {
    patches = sequencePatches;

    Nx = w;
    Ny = h;
    T = l;

    Bs = blocksize;
    Bo = blockoverlap;
    vecSize = (1 + (Nx - Bs) / Bo) * (1 + (Ny - Bs) / Bo);
    return;
  }

  // Perform SVD on each block in the image sequence,
  // subject to the block overlap restriction
  void Decompose(const arma::cube &u)
  {
    // Do the local SVDs
    arma::mat block, Ublock, Vblock;
    arma::vec Sblock;

    block.set_size(Bs * Bs, T);
    Ublock.set_size(Bs * Bs, T);
    Sblock.set_size(T);
    Vblock.set_size(T, T);

    // Fix block overlap parameter
    arma::uvec firstpatches(vecSize);
    uint32_t kiter = 0;
    for (size_t i = 0; i < 1 + (Ny - Bs); i += Bo)
    {
      for (size_t j = 0; j < 1 + (Nx - Bs); j += Bo)
      {
        firstpatches(kiter) = i * (Ny - Bs) + j;
        kiter++;
      }
    }

    // Code must include right and bottom edges
    // of the image sequence to ensure an
    // accurate PGURE reconstruction
    arma::uvec patchesbottomedge(1 + (Ny - Bs) / Bo);
    for (size_t i = 0; i < 1 + (Ny - Bs); i += Bo)
    {
      patchesbottomedge(i / Bo) = (Ny - Bs + 1) * i + (Nx - Bs);
    }

    arma::uvec patchesrightedge(1 + (Nx - Bs) / Bo);
    for (size_t i = 0; i < 1 + (Nx - Bs); i += Bo)
    {
      patchesrightedge(i / Bo) = (Ny - Bs + 1) * (Nx - Bs) + i;
    }

    // Concatenate and find unique indices
    arma::uvec joinpatches(vecSize + 1 + (Ny - Bs) / Bo + 1 + (Nx - Bs) / Bo);
    joinpatches(arma::span(0, vecSize - 1)) = firstpatches;
    joinpatches(arma::span(vecSize, vecSize + (Ny - Bs) / Bo)) = patchesrightedge;
    joinpatches(arma::span(vecSize + (Ny - Bs) / Bo + 1,
                           vecSize + (Ny - Bs) / Bo + 1 + (Nx - Bs) / Bo)) = patchesbottomedge;
    actualpatches = arma::sort(joinpatches.elem(arma::find_unique(joinpatches)));

    // Get new vector size
    newVecSize = actualpatches.n_elem;

    // Memory allocation
    U.resize(newVecSize);
    S.resize(newVecSize);
    V.resize(newVecSize);
    for (size_t it = 0; it < newVecSize; it++)
    {
      U[it] = arma::zeros<arma::mat>(Bs * Bs, T);
      S[it] = arma::zeros<arma::vec>(T);
      V[it] = arma::zeros<arma::mat>(T, T);
    }

    for (size_t it = 0; it < newVecSize; it++)
    {
      // Extract block
      for (size_t k = 0; k < T; k++)
      {
        int newy = patches(0, actualpatches(it), k);
        int newx = patches(1, actualpatches(it), k);
        block.col(k) = arma::vectorise(u(arma::span(newy, newy + Bs - 1),
                                         arma::span(newx, newx + Bs - 1), arma::span(k)));
      }

      // Do the SVD
      arma::svd_econ(Ublock, Sblock, Vblock, block);
      U[it] = Ublock;
      S[it] = Sblock;
      V[it] = Vblock;
    }
    return;
  }

  // Reconstruct block in the image sequence after thresholding
  arma::cube Reconstruct(const double lambda)
  {
    arma::cube v = arma::zeros<arma::cube>(Nx, Ny, T);
    arma::cube weights = arma::zeros<arma::cube>(Nx, Ny, T);
    arma::vec wvec = arma::zeros<arma::vec>(T);
    arma::vec zvec = arma::zeros<arma::vec>(T);

    arma::mat block, Ublock, Vblock;
    arma::vec Sblock, Snew;

    block.set_size(Bs * Bs, T);
    Ublock.set_size(Bs * Bs, T);
    Vblock.set_size(T, T);
    Sblock.set_size(T);
    Snew.set_size(T);

    for (size_t it = 0; it < newVecSize; it++)
    {
      Ublock = U[it];
      Sblock = S[it];
      Vblock = V[it];

      if (true)
      {
        // Gaussian-weighted singular value thresholding
        wvec = arma::abs(Sblock.max() * arma::exp(-0.5 * lambda * arma::square(Sblock)));
        pguresvt::SoftThreshold(Snew, Sblock, zvec, wvec);
      }
      else
      {
        // Simple singular value thresholding
        pguresvt::SoftThreshold(Snew, Sblock, zvec, lambda);
      }

      // Reconstruct from SVD
      block = Ublock * diagmat(Snew) * Vblock.t();

      // Deal with block weights (TODO: currently all weights = 1)
      for (size_t k = 0; k < T; k++)
      {
        int newy = patches(0, actualpatches(it), k);
        int newx = patches(1, actualpatches(it), k);
        v(arma::span(newy, newy + Bs - 1),
          arma::span(newx, newx + Bs - 1),
          arma::span(k, k)) += arma::reshape(block.col(k), Bs, Bs);
        weights(arma::span(newy, newy + Bs - 1),
                arma::span(newx, newx + Bs - 1),
                arma::span(k, k)) += arma::ones<arma::mat>(Bs, Bs);
      }
    }

    // Include the weighting
    v /= weights;
    v.elem(find_nonfinite(v)).zeros();
    return v;
  }

private:
  uint32_t Nx, Ny, T, Bs, Bo, vecSize, newVecSize;
  arma::icube patches;
  arma::uvec actualpatches;

  std::vector<arma::mat> U, V;
  std::vector<arma::vec> S;
};

#endif
