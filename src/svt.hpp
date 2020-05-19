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

#ifndef SVT_H
#define SVT_H

// C++ headers
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

// OpenMP library
#include <omp.h>

// Armadillo library
#include <armadillo>

class SVT
{
public:
  SVT() {}
  ~SVT() {}

  // Allocate memory for SVD step
  void Initialize(const arma::icube &sequencePatches, int w, int h, int l,
                  int blocksize, int blockoverlap)
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
    arma::mat Ublock, Vblock;
    arma::vec Sblock;

    Ublock.set_size(Bs * Bs, T);
    Sblock.set_size(T);
    Vblock.set_size(T, T);

    // Fix block overlap parameter
    arma::uvec firstpatches(vecSize);
    int kiter = 0;
    for (int i = 0; i < 1 + (Ny - Bs); i += Bo)
    {
      for (int j = 0; j < 1 + (Nx - Bs); j += Bo)
      {
        firstpatches(kiter) = i * (Ny - Bs) + j;
        kiter++;
      }
    }

    // Code must include right and bottom edges
    // of the image sequence to ensure an
    // accurate PGURE reconstruction
    arma::uvec patchesbottomedge(1 + (Ny - Bs) / Bo);
    for (int i = 0; i < 1 + (Ny - Bs); i += Bo)
    {
      patchesbottomedge(i / Bo) = (Ny - Bs + 1) * i + (Nx - Bs);
    }

    arma::uvec patchesrightedge(1 + (Nx - Bs) / Bo);
    for (int i = 0; i < 1 + (Nx - Bs); i += Bo)
    {
      patchesrightedge(i / Bo) = (Ny - Bs + 1) * (Nx - Bs) + i;
    }

    // Concatenate and find unique indices
    arma::uvec joinpatches(vecSize + 1 + (Ny - Bs) / Bo + 1 + (Nx - Bs) / Bo);
    joinpatches(arma::span(0, vecSize - 1)) = firstpatches;
    joinpatches(arma::span(vecSize, vecSize + (Ny - Bs) / Bo)) =
        patchesrightedge;
    joinpatches(arma::span(vecSize + (Ny - Bs) / Bo + 1,
                           vecSize + (Ny - Bs) / Bo + 1 + (Nx - Bs) / Bo)) =
        patchesbottomedge;
    actualpatches =
        arma::sort(joinpatches.elem(arma::find_unique(joinpatches)));

    // Get new vector size
    newVecSize = actualpatches.n_elem;

    // Memory allocation
    U.resize(newVecSize);
    S.resize(newVecSize);
    V.resize(newVecSize);
    for (int it = 0; it < newVecSize; it++)
    {
      U[it] = arma::zeros<arma::mat>(Bs * Bs, T);
      S[it] = arma::zeros<arma::vec>(T);
      V[it] = arma::zeros<arma::mat>(T, T);
    }

    //#pragma omp parallel for private(Ublock, Sblock, Vblock)
    for (int it = 0; it < newVecSize; it++)
    {
      arma::mat block(Bs * Bs, T);

      // Extract block
      for (int k = 0; k < T; k++)
      {
        int newy = patches(0, actualpatches(it), k);
        int newx = patches(1, actualpatches(it), k);
        block.col(k) =
            arma::vectorise(u(arma::span(newy, newy + Bs - 1),
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
  arma::cube Reconstruct(double lambda)
  {
    arma::cube v = arma::zeros<arma::cube>(Nx, Ny, T);
    arma::cube weights = arma::zeros<arma::cube>(Nx, Ny, T);

    arma::mat block, Ublock, Vblock;
    arma::vec Sblock;

    block.set_size(Bs * Bs, T);
    Ublock.set_size(Bs * Bs, T);
    Sblock.set_size(T);
    Vblock.set_size(T, T);

    /*
    #pragma omp parallel for shared(v, weights) \
               private(block, Ublock, Sblock, Vblock)
    */
    for (int it = 0; it < newVecSize; it++)
    {
      Ublock = U[it];
      Sblock = S[it];
      Vblock = V[it];

      // Basic singular value thresholding
      // arma::vec Snew = arma::sign(Sblock)
      //                      % arma::max(
      //                          arma::abs(Sblock) - lambda,
      //                          arma::zeros<arma::vec>(T));

      // Gaussian-weighted singular value thresholding
      arma::vec wvec = arma::abs(
          Sblock.max() * arma::exp(-1 * lambda * arma::square(Sblock) / 2));

      // Apply threshold
      arma::vec Snew =
          arma::sign(Sblock) %
          arma::max(arma::abs(Sblock) - wvec, arma::zeros<arma::vec>(T));

      // Reconstruct from SVD
      block = Ublock * diagmat(Snew) * Vblock.t();

      // Deal with block weights (TODO: currently all weights = 1)
      for (int k = 0; k < T; k++)
      {
        int newy = patches(0, actualpatches(it), k);
        int newx = patches(1, actualpatches(it), k);
        v(arma::span(newy, newy + Bs - 1), arma::span(newx, newx + Bs - 1),
          arma::span(k, k)) += arma::reshape(block.col(k), Bs, Bs);
        weights(arma::span(newy, newy + Bs - 1),
                arma::span(newx, newx + Bs - 1), arma::span(k, k)) +=
            arma::ones<arma::mat>(Bs, Bs);
      }
    }

    // Include the weighting
    v /= weights;
    v.elem(find_nonfinite(v)).zeros();
    return v;
  }

private:
  int Nx, Ny, T, Bs, Bo, vecSize, newVecSize;
  arma::icube patches;
  arma::uvec actualpatches;

  // Collate U, S, V
  std::vector<arma::mat> U, V;
  std::vector<arma::vec> S;
};

#endif
