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

#ifndef ARPS_HPP
#define ARPS_HPP

#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <armadillo>

namespace pguresvt
{
  template <typename T>
  class MotionEstimator
  {
  public:
    MotionEstimator(const arma::Cube<T> &A,
                    const uint32_t blockSize,
                    const uint32_t timeIter,
                    const uint32_t timeWindow,
                    const uint32_t motionWindow,
                    const uint32_t nImages) : A(A),
                                              blockSize(blockSize),
                                              timeIter(timeIter),
                                              timeWindow(timeWindow),
                                              motionWindow(motionWindow),
                                              nImages(nImages)
    {
      Nx = A.n_rows;
      Ny = A.n_cols;
      Nt = A.n_slices;

      nxMbs = Nx - blockSize;
      nyMbs = Ny - blockSize;
      OoBlockSizeSq = 1.0 / (blockSize * blockSize);
      vecSize = (1 + nxMbs) * (1 + nyMbs);

      patches = arma::zeros<arma::icube>(2, vecSize, 2 * timeWindow + 1);
      motions = arma::zeros<arma::icube>(2, vecSize, 2 * timeWindow);
    };

    ~MotionEstimator(){};

    void Estimate()
    {
      int negInc;

      if (timeIter < timeWindow)
      {
        uint32_t loopEnd = Nt - timeIter - 1;

        for (size_t i = 0; i < vecSize; i++) // Populate reference frame coordinates
        {
          patches(0, i, timeIter) = i % (1 + nyMbs);
          patches(1, i, timeIter) = i / (1 + nxMbs);
        }

        for (size_t i = 0; i < loopEnd; i++) // Perform motion estimation forwards
        {
          ARPSMotionEstimation(i, timeIter + i, timeIter + i + 1, timeIter + i);
        }

        for (size_t i = 0; i < timeIter; i++) // Perform motion estimation backwards
        {
          negInc = -1 * (i + 1);
          ARPSMotionEstimation(negInc, timeIter + negInc + 1, timeIter + negInc, timeIter + negInc + 1);
        }
      }
      else if (timeIter >= (nImages - timeWindow))
      {
        uint32_t endFrame = timeIter - (nImages - Nt);
        uint32_t loopEnd = 2 * timeWindow - endFrame;

        for (size_t i = 0; i < vecSize; i++) // Populate reference frame coordinates
        {
          patches(0, i, endFrame) = i % (1 + nyMbs);
          patches(1, i, endFrame) = i / (1 + nxMbs);
        }

        for (size_t i = 0; i < loopEnd; i++) // Perform motion estimation forwards
        {
          ARPSMotionEstimation(i, endFrame + i, endFrame + i + 1, endFrame + i);
        }

        for (size_t i = 0; i < endFrame; i++) // Perform motion estimation backwards
        {
          negInc = -1 * (i + 1);

          if (2 * timeWindow == endFrame)
          {
            ARPSMotionEstimation(negInc, endFrame + negInc + 1, endFrame + negInc, endFrame + negInc);
          }
          else
          {
            ARPSMotionEstimation(negInc, endFrame + negInc + 1, endFrame + negInc, endFrame + negInc + 1);
          }
        }
      }
      else
      {
        for (size_t i = 0; i < vecSize; i++) // Populate reference frame coordinates
        {
          patches(0, i, timeWindow) = i % (1 + nyMbs);
          patches(1, i, timeWindow) = i / (1 + nxMbs);
        }

        for (size_t i = 0; i < timeWindow; i++) // Perform motion estimation forwards
        {
          ARPSMotionEstimation(i, timeWindow + i, timeWindow + i + 1, timeWindow + i);
        }

        for (size_t i = 0; i < timeWindow; i++) // Perform motion estimation backwards
        {
          negInc = -1 * (i + 1);
          ARPSMotionEstimation(negInc, timeWindow + negInc + 1, timeWindow + negInc, timeWindow + negInc + 1);
        }
      }
      return;
    }

    arma::icube GetEstimate() { return patches; }

  private:
    arma::Cube<T> A;
    uint32_t blockSize, timeIter, timeWindow, motionWindow, nImages;

    arma::icube patches, motions;
    uint32_t Nx, Ny, Nt;
    uint32_t nxMbs, nyMbs, vecSize;
    double OoBlockSizeSq;

    // Adaptive Rood Pattern Search (ARPS) method
    void ARPSMotionEstimation(const int curFrame, const int iARPS1, const int iARPS2, const int iARPS3)
    {
      double norm = 0;
      double costsScale = 1E8;
      arma::vec costs = arma::ones<arma::vec>(6) * costsScale;
      arma::umat checkMat = arma::zeros<arma::umat>(2 * motionWindow + 1, 2 * motionWindow + 1);
      arma::imat LDSP = arma::zeros<arma::imat>(6, 2);
      arma::imat SDSP = arma::zeros<arma::imat>(5, 2);

      for (size_t it = 0; it < vecSize; it++)
      {
        costs.fill(costsScale);
        checkMat.zeros();
        LDSP.zeros();
        SDSP.zeros();

        SDSP(0, 0) = 0;
        SDSP(0, 1) = -1;
        SDSP(1, 0) = -1;
        SDSP(1, 1) = 0;
        SDSP(2, 0) = 0;
        SDSP(2, 1) = 0;
        SDSP(3, 0) = 1;
        SDSP(3, 1) = 0;
        SDSP(4, 0) = 0;
        SDSP(4, 1) = 1;
        LDSP.rows(arma::span(0, 4)) = SDSP;

        int i = it % (1 + nxMbs);
        int j = it / (1 + nyMbs);

        int x = j;
        int y = i;

        arma::Cube<T> refBlock = A(arma::span(i, i + blockSize - 1),
                                   arma::span(j, j + blockSize - 1),
                                   arma::span(iARPS1));

        arma::Cube<T> newBlock = A(arma::span(i, i + blockSize - 1),
                                   arma::span(j, j + blockSize - 1),
                                   arma::span(iARPS2));

        norm = arma::norm(refBlock.slice(0) - newBlock.slice(0), "fro");
        costs(2) = norm * norm * OoBlockSizeSq;

        checkMat(motionWindow, motionWindow) = 1;

        uint32_t maxIdx;
        int stepSize;

        if (j == 0)
        {
          stepSize = 2;
          maxIdx = 5;
        }
        else
        {
          int yTmp = std::abs(motions(0, it, iARPS3));
          int xTmp = std::abs(motions(1, it, iARPS3));
          stepSize = (xTmp <= yTmp) ? yTmp : xTmp;
          if ((yTmp == 0 && xTmp == stepSize) || (xTmp == 0 && yTmp == stepSize))
          {
            maxIdx = 5;
          }
          else
          {
            maxIdx = 6;
            LDSP(5, 0) = motions(1, it, iARPS3);
            LDSP(5, 1) = motions(0, it, iARPS3);
          }
        }

        LDSP(0, 0) = 0;
        LDSP(0, 1) = -1 * stepSize;
        LDSP(1, 0) = -1 * stepSize;
        LDSP(1, 1) = 0;
        LDSP(2, 0) = 0;
        LDSP(2, 1) = 0;
        LDSP(3, 0) = stepSize;
        LDSP(3, 1) = 0;
        LDSP(4, 0) = 0;
        LDSP(4, 1) = stepSize;

        // Currently not used, but motion estimation can be
        // predictive if this value is larger than 0
        double pMotion = 0.0;

        // Do the LDSP
        bool skipIt = false;

        for (size_t k = 0; k < maxIdx; k++)
        {
          int refBlkVer = y + LDSP(k, 1);
          int refBlkHor = x + LDSP(k, 0);

          skipIt = ((k == 2) ||
                    (stepSize == 0) ||
                    (refBlkHor < 0) ||
                    (refBlkVer < 0) ||
                    (refBlkHor + blockSize - 1) >= Ny ||
                    (refBlkVer + blockSize - 1) >= Nx);

          if (!skipIt) // Only evaluate if none of the above is true
          {
            arma::Cube<T> powBlock = A(arma::span(refBlkVer, refBlkVer + blockSize - 1),
                                       arma::span(refBlkHor, refBlkHor + blockSize - 1),
                                       arma::span(iARPS2));
            if (curFrame == 0)
            {
              norm = arma::norm(refBlock.slice(0) - powBlock.slice(0), "fro");
              costs(k) = norm * norm * OoBlockSizeSq;
            }
            else if (curFrame < 0)
            {
              norm = arma::norm(refBlock.slice(0) - powBlock.slice(0), "fro");
              costs(k) = norm * norm * OoBlockSizeSq;

              if (pMotion > 0.0)
              {
                arma::ivec predPos = arma::vectorise(
                    patches(arma::span(), arma::span(it), arma::span(iARPS1)) -
                    motions(arma::span(), arma::span(it), arma::span(iARPS3)));
                costs(k) += pMotion * std::sqrt(std::pow(predPos(0) - refBlkVer, 2) +
                                                std::pow(predPos(1) - refBlkHor, 2));
              }
            }
            else if (curFrame > 0)
            {
              norm = arma::norm(refBlock.slice(0) - powBlock.slice(0), "fro");
              costs(k) = norm * norm * OoBlockSizeSq;

              if (pMotion > 0.0)
              {
                arma::ivec predPos = arma::vectorise(
                    patches(arma::span(), arma::span(it), arma::span(iARPS1)) +
                    motions(arma::span(), arma::span(it), arma::span(iARPS3)));

                norm = arma::norm(refBlock.slice(0) - powBlock.slice(0), "fro");
                costs(k) += pMotion * std::sqrt(std::pow(predPos(0) - refBlkVer, 2) +
                                                std::pow(predPos(1) - refBlkHor, 2));
              }
            }

            checkMat(LDSP(k, 1) + motionWindow, LDSP(k, 0) + motionWindow) = 1;
          }
        }

        arma::uvec point = arma::find(costs == costs.min());
        x += LDSP(point(0), 0);
        y += LDSP(point(0), 1);
        double cost = costs.min();
        costs.fill(costsScale);
        costs(2) = cost;

        // Do the SDSP
        bool doneFlag = false;

        do
        {
          bool skipIt = false;

          for (int k = 0; k < 5; k++)
          {
            int refBlkVer = y + SDSP(k, 1);
            int refBlkHor = x + SDSP(k, 0);

            skipIt = ((k == 2) ||
                      (refBlkHor < 0) ||
                      (refBlkVer < 0) ||
                      (refBlkHor + blockSize - 1) >= Ny ||
                      (refBlkVer + blockSize - 1) >= Nx ||
                      (refBlkHor < (int)(j - motionWindow)) ||
                      (refBlkHor > (int)(j + motionWindow)) ||
                      (refBlkVer < (int)(i - motionWindow)) ||
                      (refBlkVer > (int)(i + motionWindow)) ||
                      (checkMat(y - i + SDSP(k, 1) + motionWindow,
                                x - j + SDSP(k, 0) + motionWindow) == 1));

            if (!skipIt) // Only evaluate if none of the above is true
            {
              arma::Cube<T> powBlock = A(arma::span(refBlkVer, refBlkVer + blockSize - 1),
                                         arma::span(refBlkHor, refBlkHor + blockSize - 1),
                                         arma::span(iARPS2));
              if (curFrame == 0)
              {
                norm = arma::norm(refBlock.slice(0) - powBlock.slice(0), "fro");
                costs(k) = norm * norm * OoBlockSizeSq;
              }
              else if (curFrame < 0)
              {
                norm = arma::norm(refBlock.slice(0) - powBlock.slice(0), "fro");
                costs(k) = norm * norm * OoBlockSizeSq;

                if (pMotion > 0.0)
                {
                  arma::ivec predPos = arma::vectorise(
                      patches(arma::span(), arma::span(it), arma::span(iARPS1)) -
                      motions(arma::span(), arma::span(it), arma::span(iARPS3)));

                  costs(k) += pMotion * std::sqrt(std::pow(predPos(0) - refBlkVer, 2) +
                                                  std::pow(predPos(1) - refBlkHor, 2));
                }
              }
              else if (curFrame > 0)
              {
                norm = arma::norm(refBlock.slice(0) - powBlock.slice(0), "fro");
                costs(k) = norm * norm * OoBlockSizeSq;

                if (pMotion > 0.0)
                {
                  arma::ivec predPos = arma::vectorise(
                      patches(arma::span(), arma::span(it), arma::span(iARPS1)) +
                      motions(arma::span(), arma::span(it), arma::span(iARPS3)));

                  costs(k) += pMotion * std::sqrt(std::pow(predPos(0) - refBlkVer, 2) +
                                                  std::pow(predPos(1) - refBlkHor, 2));
                }
              }

              checkMat(y - i + SDSP(k, 1) + motionWindow, x - j + SDSP(k, 0) + motionWindow) = 1;
            }
          }

          point = arma::find(costs == costs.min());
          cost = costs.min();

          if (point(0) == 2)
          {
            doneFlag = true;
          }
          else
          {
            x += SDSP(point(0), 0);
            y += SDSP(point(0), 1);
            costs.fill(costsScale);
            costs(2) = cost;
          }
        } while (!doneFlag);

        motions(0, it, iARPS3) = y - i;
        motions(1, it, iARPS3) = x - j;
        patches(0, it, iARPS2) = y;
        patches(1, it, iARPS2) = x;
      }
      return;
    }
  };

} // namespace pguresvt

#endif
