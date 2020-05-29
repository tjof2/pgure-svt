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

#ifndef HOTPIXEL_HPP
#define HOTPIXEL_HPP

#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <armadillo>

#include "parallel.hpp"
#include "utils.hpp"

template <typename T>
void HotPixelFilter(arma::Cube<T> &sequence, const double threshold, const int n_jobs)
{
  uint32_t Nx = sequence.n_rows;
  uint32_t Ny = sequence.n_cols;
  uint32_t Nt = sequence.n_slices;

  double mad_scale = 1.0 / 0.6745;

  auto &&func = [&](uint32_t i) {
    arma::vec8 medianWindow(arma::fill::zeros);

    double median = arma::median(arma::median(sequence.slice(i)));
    double mad = arma::median(arma::median((arma::abs(sequence.slice(i) - median)))) * mad_scale;
    arma::uvec outliers = arma::find(arma::abs(sequence.slice(i) - median) > threshold * mad);

    for (size_t j = 0; j < outliers.n_elem; j++)
    {
      medianWindow.zeros();
      arma::uvec sub = arma::ind2sub(arma::size(Nx, Ny), outliers(j));

      if ((sub(0) > 0) && (sub(0) < Nx - 1) && (sub(1) > 0) && (sub(1) < Ny - 1))
      {
        medianWindow(0) = sequence(sub(0) - 1, sub(1) - 1, i);
        medianWindow(1) = sequence(sub(0) - 1, sub(1), i);
        medianWindow(2) = sequence(sub(0) - 1, sub(1) + 1, i);
        medianWindow(3) = sequence(sub(0), sub(1) - 1, i);
        medianWindow(4) = sequence(sub(0), sub(1) + 1, i);
        medianWindow(5) = sequence(sub(0) + 1, sub(1) - 1, i);
        medianWindow(6) = sequence(sub(0) + 1, sub(1), i);
        medianWindow(7) = sequence(sub(0) + 1, sub(1) + 1, i);

        medianWindow = arma::sort(medianWindow);
        sequence(sub(0), sub(1), i) = static_cast<T>(0.5 * (medianWindow(3) + medianWindow(4)));
      }
      else // Edge outliers are replaced by the median of the frame
      {
        sequence(sub(0), sub(1), i) = static_cast<T>(median);
      }
    }
  };

  parallel(func, static_cast<uint32_t>(0), static_cast<uint32_t>(Nt), n_jobs); // Apply over the images

  return;
}

#endif
