/***************************************************************************

    Copyright (C) 2015-2020 Tom Furnival

    This file is part of PGURE-SVT.

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
#include <vector>
#include <armadillo>

void HotPixelFilter(arma::cube &sequence, const double threshold)
{
  int Nx = sequence.n_rows;
  int Ny = sequence.n_cols;
  int T = sequence.n_slices;

  arma::vec medianwindow(8, arma::fill::zeros);

  for (int i = 0; i < T; i++)
  {
    double median = arma::median(arma::median(sequence.slice(i)));
    double medianAbsDev =
        arma::median(arma::median(arma::abs(sequence.slice(i) - median))) /
        0.6745;
    arma::uvec outliers = arma::find(arma::abs(sequence.slice(i) - median) >
                                     threshold * medianAbsDev);

    for (size_t j = 0; j < outliers.n_elem; j++)
    {
      medianwindow.zeros();
      arma::uvec sub = arma::ind2sub(arma::size(Nx, Ny), outliers(j));

      if ((int)sub(0) > 0 && (int)sub(0) < Nx - 1 && (int)sub(1) > 0 &&
          (int)sub(1) < Ny - 1)
      {
        medianwindow(0) = sequence(sub(0) - 1, sub(1) - 1, i);
        medianwindow(1) = sequence(sub(0) - 1, sub(1), i);
        medianwindow(2) = sequence(sub(0) - 1, sub(1) + 1, i);
        medianwindow(3) = sequence(sub(0), sub(1) - 1, i);
        medianwindow(4) = sequence(sub(0), sub(1) + 1, i);
        medianwindow(5) = sequence(sub(0) + 1, sub(1) - 1, i);
        medianwindow(6) = sequence(sub(0) + 1, sub(1), i);
        medianwindow(7) = sequence(sub(0) + 1, sub(1) + 1, i);

        medianwindow = arma::sort(medianwindow);
        sequence(sub(0), sub(1), i) = (medianwindow(3) + medianwindow(4)) / 2;
      }
      else
      {
        // Edge pixels are currently replaced by the median of the frame
        // (as they are not usually very important)
        sequence(sub(0), sub(1), i) = median;
      }
    }
  }
  return;
}

#endif
