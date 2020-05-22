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
    nxMbs = Nx - blockSize;
    nyMbs = Ny - blockSize;
    nxMbsDbo = nxMbs / blockOverlap;
    nyMbsDbo = nyMbs / blockOverlap;
    vecSize = (1 + nxMbsDbo) * (1 + nyMbsDbo);

    block.set_size(blockSize * blockSize, T);
    Ublock.set_size(blockSize * blockSize, T);
    Vblock.set_size(T, T);
    Sblock.set_size(T);
    Sthresh.set_size(T);
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
    // Fix block overlap parameter
    arma::uvec firstPatches(vecSize);
    uint32_t kiter = 0;
    for (size_t i = 0; i < 1 + nyMbs; i += blockOverlap)
    {
      for (size_t j = 0; j < 1 + nxMbs; j += blockOverlap)
      {
        firstPatches(kiter) = i * nyMbs + j;
        kiter++;
      }
    }

    // Code must include right and bottom edges
    // of the image sequence to ensure an
    // accurate PGURE reconstruction
    arma::uvec patchesBottomEdge(1 + nyMbsDbo);
    for (size_t i = 0; i < 1 + nyMbs; i += blockOverlap)
    {
      patchesBottomEdge(i / blockOverlap) = (nyMbs + 1) * i + nxMbs;
    }

    arma::uvec patchesRightEdge(1 + nxMbsDbo);
    for (size_t i = 0; i < 1 + nxMbs; i += blockOverlap)
    {
      patchesRightEdge(i / blockOverlap) = (nyMbs + 1) * nxMbs + i;
    }

    // Concatenate and find unique indices
    arma::uvec joinPatches(vecSize + 1 + nyMbsDbo + 1 + nxMbsDbo);
    joinPatches(arma::span(0, vecSize - 1)) = firstPatches;
    joinPatches(arma::span(vecSize, vecSize + nyMbsDbo)) = patchesRightEdge;
    joinPatches(arma::span(vecSize + nyMbsDbo + 1, vecSize + nyMbsDbo + 1 + nxMbsDbo)) = patchesBottomEdge;
    actualPatches = arma::sort(joinPatches.elem(arma::find_unique(joinPatches)));

    // Get new vector size
    newVecSize = actualPatches.n_elem;

    // Memory allocation
    U.resize(newVecSize);
    S.resize(newVecSize);
    V.resize(newVecSize);

    for (size_t it = 0; it < newVecSize; it++)
    {
      // Extract block
      for (size_t k = 0; k < T; k++)
      {
        int newY = patches(0, actualPatches(it), k);
        int newX = patches(1, actualPatches(it), k);
        block.col(k) = arma::vectorise(
            u(arma::span(newY, newY + blockSize - 1),
              arma::span(newX, newX + blockSize - 1),
              arma::span(k)));
      }

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
    arma::mat weightsInc = arma::ones<arma::mat>(blockSize, blockSize);
    arma::vec zvec = arma::zeros<arma::vec>(T);
    arma::vec wvec = arma::zeros<arma::vec>(T);

    for (size_t it = 0; it < newVecSize; it++)
    {
      Ublock = U[it];
      Sblock = S[it];
      Vblock = V[it];

      if (expWeighting) // Gaussian-weighted singular value thresholding
      {
        wvec = arma::abs(Sblock.max() * arma::exp(-0.5 * lambda * arma::square(Sblock)));
        pguresvt::softThreshold(Sthresh, Sblock, zvec, wvec);
      }
      else // Simple singular value thresholding
      {
        pguresvt::softThreshold(Sthresh, Sblock, zvec, lambda);
      }

      // Reconstruct from SVD
      block = Ublock * diagmat(Sthresh) * Vblock.t();

      for (size_t k = 0; k < T; k++)
      {
        int newY = patches(0, actualPatches(it), k);
        int newX = patches(1, actualPatches(it), k);

        v(arma::span(newY, newY + blockSize - 1),
          arma::span(newX, newX + blockSize - 1),
          arma::span(k, k)) += arma::reshape(block.col(k), blockSize, blockSize);

        weights(arma::span(newY, newY + blockSize - 1),
                arma::span(newX, newX + blockSize - 1),
                arma::span(k, k)) += weightsInc;
      }
    }

    // Apply weighting
    v /= weights;
    v.elem(find_nonfinite(v)).zeros();

    return v;
  };

private:
  arma::icube patches;
  uint32_t Nx, Ny, T;
  uint32_t blockSize, blockOverlap;
  bool expWeighting;

  uint32_t vecSize, newVecSize, nxMbs, nyMbs, nxMbsDbo, nyMbsDbo;

  arma::uvec actualPatches;
  arma::mat block, Ublock, Vblock;
  arma::vec Sblock, Sthresh;

  std::vector<arma::mat> U, V;
  std::vector<arma::vec> S;
};

#endif
