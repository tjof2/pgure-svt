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
  SVT(const arma::icube &patches,
      const uint32_t Nx,
      const uint32_t Ny,
      const uint32_t T,
      const uint32_t blockSize,
      const uint32_t blockOverlap,
      const bool expWeighting) : patches(patches),
                                 Nx(Nx), Ny(Ny), T(T),
                                 blockSize(blockSize),
                                 blockOverlap(blockOverlap),
                                 expWeighting(expWeighting)
  {
    vecSize = (1 + (Nx - blockSize) / blockOverlap) * (1 + (Ny - blockSize) / blockOverlap);
  };

  ~SVT()
  {
    U.clear();
    S.clear();
    V.clear();
  };

  // Perform SVD on each block in the image sequence,
  // subject to the block overlap restriction
  void Decompose(const arma::cube &u)
  {
    // Do the local SVDs
    arma::mat block, Ublock, Vblock;
    arma::vec Sblock;

    block.set_size(blockSize * blockSize, T);
    Ublock.set_size(blockSize * blockSize, T);
    Sblock.set_size(T);
    Vblock.set_size(T, T);

    // Fix block overlap parameter
    arma::uvec firstPatches(vecSize);
    uint32_t kiter = 0;
    for (size_t i = 0; i < 1 + (Ny - blockSize); i += blockOverlap)
    {
      for (size_t j = 0; j < 1 + (Nx - blockSize); j += blockOverlap)
      {
        firstPatches(kiter) = i * (Ny - blockSize) + j;
        kiter++;
      }
    }

    // Code must include right and bottom edges
    // of the image sequence to ensure an
    // accurate PGURE reconstruction
    arma::uvec patchesBottomEdge(1 + (Ny - blockSize) / blockOverlap);
    for (size_t i = 0; i < 1 + (Ny - blockSize); i += blockOverlap)
    {
      patchesBottomEdge(i / blockOverlap) = (Ny - blockSize + 1) * i + (Nx - blockSize);
    }

    arma::uvec patchesRightEdge(1 + (Nx - blockSize) / blockOverlap);
    for (size_t i = 0; i < 1 + (Nx - blockSize); i += blockOverlap)
    {
      patchesRightEdge(i / blockOverlap) = (Ny - blockSize + 1) * (Nx - blockSize) + i;
    }

    // Concatenate and find unique indices
    arma::uvec joinPatches(vecSize + 1 + (Ny - blockSize) / blockOverlap + 1 + (Nx - blockSize) / blockOverlap);
    joinPatches(arma::span(0, vecSize - 1)) = firstPatches;
    joinPatches(arma::span(vecSize, vecSize + (Ny - blockSize) / blockOverlap)) = patchesRightEdge;
    joinPatches(arma::span(vecSize + (Ny - blockSize) / blockOverlap + 1,
                           vecSize + (Ny - blockSize) / blockOverlap + 1 + (Nx - blockSize) / blockOverlap)) = patchesBottomEdge;
    actualPatches = arma::sort(joinPatches.elem(arma::find_unique(joinPatches)));

    // Get new vector size
    newVecSize = actualPatches.n_elem;

    // Memory allocation
    U.resize(newVecSize);
    S.resize(newVecSize);
    V.resize(newVecSize);
    for (size_t it = 0; it < newVecSize; it++)
    {
      U[it] = arma::zeros<arma::mat>(blockSize * blockSize, T);
      S[it] = arma::zeros<arma::vec>(T);
      V[it] = arma::zeros<arma::mat>(T, T);
    }

    for (size_t it = 0; it < newVecSize; it++)
    {
      // Extract block
      for (size_t k = 0; k < T; k++)
      {
        int newy = patches(0, actualPatches(it), k);
        int newx = patches(1, actualPatches(it), k);
        block.col(k) = arma::vectorise(u(arma::span(newy, newy + blockSize - 1),
                                         arma::span(newx, newx + blockSize - 1), arma::span(k)));
      }

      // Do the SVD
      arma::svd_econ(Ublock, Sblock, Vblock, block);
      U[it] = Ublock;
      S[it] = Sblock;
      V[it] = Vblock;
    }
    return;
  };

  // Reconstruct block in the image sequence after thresholding
  arma::cube Reconstruct(const double lambda)
  {
    arma::cube v = arma::zeros<arma::cube>(Nx, Ny, T);
    arma::cube weights = arma::zeros<arma::cube>(Nx, Ny, T);
    arma::vec zvec = arma::zeros<arma::vec>(T);
    arma::vec wvec = arma::zeros<arma::vec>(T);

    arma::mat block, Ublock, Vblock;
    arma::vec Sblock, Snew;

    block.set_size(blockSize * blockSize, T);
    Ublock.set_size(blockSize * blockSize, T);
    Vblock.set_size(T, T);
    Sblock.set_size(T);
    Snew.set_size(T);

    for (size_t it = 0; it < newVecSize; it++)
    {
      Ublock = U[it];
      Sblock = S[it];
      Vblock = V[it];

      if (expWeighting)
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

      for (size_t k = 0; k < T; k++)
      {
        int newy = patches(0, actualPatches(it), k);
        int newx = patches(1, actualPatches(it), k);

        v(arma::span(newy, newy + blockSize - 1),
          arma::span(newx, newx + blockSize - 1),
          arma::span(k, k)) += arma::reshape(block.col(k), blockSize, blockSize);

        weights(arma::span(newy, newy + blockSize - 1),
                arma::span(newx, newx + blockSize - 1),
                arma::span(k, k)) += arma::ones<arma::mat>(blockSize, blockSize);
      }
    }

    // Apply weighting
    v /= weights;
    v.elem(find_nonfinite(v)).zeros();

    return v;
  };

private:
  arma::icube patches;
  uint32_t Nx, Ny, T, blockSize, blockOverlap, vecSize, newVecSize;
  bool expWeighting;

  arma::uvec actualPatches;

  std::vector<arma::mat> U, V;
  std::vector<arma::vec> S;
};

#endif
