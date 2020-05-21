/***************************************************************************

    PGURE-SVT Denoising

    Author: Tom Furnival
    Email:  tjof2@cam.ac.uk

    Copyright (C) 2015-2020 Tom Furnival

    This program uses Singular Value Thresholding (SVT) [1], combined
    with an unbiased risk estimator (PGURE) to denoise a video sequence
    of microscopy images [2]. Noise parameters for a mixed Poisson-Gaussian
    noise model are automatically estimated during the denoising.

    References:
    [1] "Unbiased Risk Estimates for Singular Value Thresholding and
        Spectral Estimators", (2013), Candes, EJ et al.
        http://dx.doi.org/10.1109/TSP.2013.2270464

    [2] "An Unbiased Risk Estimator for Image Denoising in the Presence
        of Mixed Poisson–Gaussian Noise", (2014), Le Montagner, Y et al.
        http://dx.doi.org/10.1109/TIP.2014.2300821

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

#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <cstdarg>
#include <stdexcept>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <vector>
#include <armadillo>

namespace libtiff
{
#include "tiffio.h"
}

extern "C"
{
#include "medfilter.h"
}

#include "arps.hpp"
#include "hotpixel.hpp"
#include "params.hpp"
#include "noise.hpp"
#include "pgure.hpp"
#include "parallel.hpp"
#include "utils.hpp"

// Main program
int main(int argc, char **argv)
{

  pguresvt::print(std::cout,
                  "PGURE-SVT Denoising\n",
                  "Author: Tom Furnival\n",
                  "Email:  tjof2@cam.ac.uk\n");

  if (argc != 2)
  {
    pguresvt::print(std::cout, "  Usage: ./PGURE-SVT paramfile");
    return -1;
  }

  std::map<std::string, std::string> opts;
  std::ifstream paramFile(argv[1], std::ios::in);

  ParseParameters(paramFile, opts);

  if (opts.count("filename") == 0 ||
      opts.count("start_image") == 0 ||
      opts.count("end_image") == 0) // Check all required parameters are specified
  {
    pguresvt::print(std::cerr, "**WARNING** Required parameters not specified\n",
                    "            You must specify filename, start and end frame");
    return -1;
  }

  // Extract parameters
  std::string filename = opts.at("filename");
  int lastindex = filename.find_last_of(".");
  std::string filestem = filename.substr(0, lastindex);

  uint32_t startImage = std::stoi(opts.at("start_image"));
  uint32_t endImage = std::stoi(opts.at("end_image"));
  uint32_t nImages = endImage - startImage + 1;

  // Move onto optional parameters
  uint32_t Bs = (opts.count("patch_size") == 1) ? std::stoi(opts.at("patch_size")) : 4;
  uint32_t T = (opts.count("trajectory_length") == 1) ? std::stoi(opts.at("trajectory_length")) : 15;
  T = (Bs * Bs < T) ? (Bs * Bs) - 1 : T;

  // Noise parameters (initialized at -1 unless user-defined)
  double alpha = (opts.count("alpha") == 1) ? std::stod(opts.at("alpha")) : -1.;
  double mu = (opts.count("mu") == 1) ? std::stod(opts.at("mu")) : -1.;
  double sigma = (opts.count("sigma") == 1) ? std::stod(opts.at("sigma")) : -1.;

  // SVT threshold (initialized at -1 unless user-defined)
  bool pgureOpt = (opts.count("optimize_pgure") == 1) ? pguresvt::strToBool(opts.at("optimize_pgure")) : true;
  double lambda;
  if (!pgureOpt)
  {
    if (opts.count("lambda") == 1)
    {
      lambda = std::stod(opts.at("lambda"));
    }
    else
    {
      pguresvt::print(std::cerr,
                      "**WARNING** PGURE optimization is turned OFF but no lambda ",
                      "specified in parameter file");
      return -1;
    }
  }

  uint32_t motionWindow = (opts.count("motion_neighbourhood") == 1) ? std::stoi(opts.at("motion_neighbourhood")) : 7;
  uint32_t medianSize = (opts.count("median_filter") == 1) ? std::stoi(opts.at("median_filter")) : 5;
  uint32_t Bo = (opts.count("patch_overlap") == 1) ? std::stoi(opts.at("patch_overlap")) : 1;
  uint32_t noiseMethod = (opts.count("noise_method") == 1) ? std::stoi(opts.at("noise_method")) : 4;

  double hotPixelThreshold = (opts.count("hot_pixel") == 1) ? std::stoi(opts.at("hot_pixel")) : 10;
  int randomSeed = (opts.count("random_seed") == 1) ? std::stoi(opts.at("random_seed")) : -1;
  bool expWeighting = (opts.count("exponential_weighting") == 1) ? pguresvt::strToBool(opts.at("exponential_weighting")) : true;

  double tol = 1E-7;
  if (opts.count("tolerance") == 1) // PGURE tolerance parsing
  {
    std::istringstream osTol(opts.at("tolerance"));
    double tol;
    osTol >> tol;
  }

  // Overall program timer now we've loaded the parameters
  auto t0Start = std::chrono::high_resolution_clock::now();

  std::string inFilename = filestem + ".tif";
  if (!std::ifstream(inFilename.c_str())) // Check file exists
  {
    pguresvt::print(std::cerr, "**WARNING** File ", inFilename, " not found");
    return -1;
  }

  // Load TIFF stack
  uint32_t tiffWidth, tiffHeight;
  uint16_t tiffDepth;
  libtiff::TIFF *MultiPageTiff = libtiff::TIFFOpen(inFilename.c_str(), "r");
  libtiff::TIFFGetField(MultiPageTiff, TIFFTAG_IMAGEWIDTH, &tiffWidth);
  libtiff::TIFFGetField(MultiPageTiff, TIFFTAG_IMAGELENGTH, &tiffHeight);
  libtiff::TIFFGetField(MultiPageTiff, TIFFTAG_BITSPERSAMPLE, &tiffDepth);

  if (tiffWidth != tiffHeight) // Only work with square images
  {
    pguresvt::print(std::cerr, "**WARNING** Frame dimensions are not square, got ", tiffWidth, "x", tiffHeight);
    return -1;
  }

  if (tiffDepth != 8 && tiffDepth != 16) // Only work with 8-bit or 16-bit images
  {
    pguresvt::print(std::cerr, "**WARNING** Images must be 8-bit or 16-bit, got ", tiffDepth, "-bit");
    return -1;
  }

  arma::cube inputSequence(tiffHeight, tiffWidth, 0);
  arma::cube filteredSequence(tiffHeight, tiffWidth, 0);

  if (MultiPageTiff) // Import the image sequence
  {
    uint32_t dircount = 0;
    uint32_t imgcount = 0;

    int memsize = 512 * 1024;  // L2 cache size
    int filtsize = medianSize; // Median filter size in pixels

    do
    {
      if (dircount >= (startImage - 1) && dircount <= (endImage - 1))
      {
        inputSequence.resize(tiffHeight, tiffWidth, imgcount + 1);
        filteredSequence.resize(tiffHeight, tiffWidth, imgcount + 1);

        uint16_t *Buffer = new uint16_t[tiffWidth * tiffHeight];
        uint16_t *FilteredBuffer = new uint16_t[tiffWidth * tiffHeight];

        for (size_t tiffRow = 0; tiffRow < tiffHeight; tiffRow++)
        {
          libtiff::TIFFReadScanline(MultiPageTiff, &Buffer[tiffRow * tiffWidth], tiffRow, 0);
        }

        arma::Mat<uint16_t> TiffSlice(Buffer, tiffHeight, tiffWidth);
        inplace_trans(TiffSlice);
        inputSequence.slice(imgcount) = arma::conv_to<arma::mat>::from(TiffSlice);

        // Apply median filter (constant-time) to the 8-bit image
        ConstantTimeMedianFilter(Buffer, FilteredBuffer, tiffWidth, tiffHeight,
                                 tiffWidth, tiffWidth, filtsize, 1, memsize);

        arma::Mat<uint16_t> FilteredTiffSlice(FilteredBuffer, tiffHeight, tiffWidth);
        inplace_trans(FilteredTiffSlice);
        filteredSequence.slice(imgcount) = arma::conv_to<arma::mat>::from(FilteredTiffSlice);
        imgcount++;
      }
      dircount++;
    } while (libtiff::TIFFReadDirectory(MultiPageTiff));
    libtiff::TIFFClose(MultiPageTiff);
  }

  // Is number of frames compatible?
  if (nImages > inputSequence.n_slices)
  {
    pguresvt::print(std::cerr, "**WARNING** Sequence only has ",
                    inputSequence.n_slices, " frames");
    return -1;
  }

  // TIFF import timer
  auto t0End = std::chrono::high_resolution_clock::now();
  auto t0Elapsed = std::chrono::duration_cast<std::chrono::microseconds>(t0End - t0Start);
  pguresvt::print_fixed(5, "TIFF import took ", t0Elapsed.count() * 1E-6, " seconds");

  auto t1Start = std::chrono::high_resolution_clock::now();

  // Copy image sequence and sizes
  arma::cube cleanSequence = inputSequence;
  cleanSequence.zeros();

  // Get dimensions
  uint32_t Nx = tiffHeight;
  uint32_t Ny = tiffWidth;

  // Initial outlier detection (for hot pixels)
  // using median absolute deviation
  HotPixelFilter(inputSequence, hotPixelThreshold);

  // Loop over time windows
  uint32_t frameWindow = std::floor(T / 2);

  auto &&func = [&, lambda_ = lambda](uint32_t timeIter) {
    auto lambda = lambda_;
    // Extract the subset of the image sequence
    arma::cube u(Nx, Ny, T), uFilter(Nx, Ny, T), v(Nx, Ny, T);
    if (timeIter < frameWindow)
    {
      u = inputSequence.slices(0, 2 * frameWindow);
      uFilter = filteredSequence.slices(0, 2 * frameWindow);
    }
    else if (timeIter >= (nImages - frameWindow))
    {
      u = inputSequence.slices(nImages - 2 * frameWindow - 1, nImages - 1);
      uFilter = filteredSequence.slices(nImages - 2 * frameWindow - 1, nImages - 1);
    }
    else
    {
      u = inputSequence.slices(timeIter - frameWindow, timeIter + frameWindow);
      uFilter = filteredSequence.slices(timeIter - frameWindow, timeIter + frameWindow);
    }

    // Basic sequence normalization
    double inputmax = u.max();
    u /= inputmax;
    uFilter /= uFilter.max();

    // Perform noise estimation
    if (pgureOpt)
    {
      NoiseEstimator *noise = new NoiseEstimator;
      noise->Estimate(u, alpha, mu, sigma, 8, noiseMethod, 0);
      delete noise;
    }

    // Perform motion estimation
    MotionEstimator *motion = new MotionEstimator;
    motion->Estimate(uFilter, timeIter, frameWindow, nImages, Bs, motionWindow);
    arma::icube sequencePatches = motion->GetEstimate();
    delete motion;

    // Perform PGURE optimization
    PGURE *optimizer = new PGURE(u, sequencePatches, alpha, sigma, mu, Bs, Bo, randomSeed, expWeighting);
    // Determine optimum threshold value (max 1000 evaluations)
    if (pgureOpt)
    {
      lambda = (timeIter == 0) ? arma::accu(u) / (Nx * Ny * T) : lambda;
      lambda = optimizer->Optimize(tol, lambda, u.max(), 1E3);
      v = optimizer->Reconstruct(lambda);
    }
    else
    {
      v = optimizer->Reconstruct(lambda);
    }
    delete optimizer;

    // Rescale back to original range
    v *= inputmax;

    // Place frames back into sequence
    if (timeIter < frameWindow)
    {
      cleanSequence.slice(timeIter) = v.slice(timeIter);
    }
    else if (timeIter >= (nImages - frameWindow))
    {
      cleanSequence.slice(timeIter) = v.slice(timeIter - (nImages - T));
    }
    else
    {
      cleanSequence.slice(timeIter) = v.slice(frameWindow);
    }
  };
  parallel(func, static_cast<uint32_t>(nImages));

  // PGURE-SVT timer
  auto t1End = std::chrono::high_resolution_clock::now();
  auto t1Elapsed = std::chrono::duration_cast<std::chrono::microseconds>(t1End - t1Start);
  pguresvt::print_fixed(5, "PGURE-SVT took ", t1Elapsed.count() * 1E-6, " seconds");

  auto t2Start = std::chrono::high_resolution_clock::now();

  // Normalize to [0,65535] range
  cleanSequence = 65535 * (cleanSequence - cleanSequence.min()) / (cleanSequence.max() - cleanSequence.min());

  arma::Cube<uint16_t> outTiff(tiffWidth, tiffHeight, nImages);
  outTiff = arma::conv_to<arma::Cube<uint16_t>>::from(cleanSequence);

  // Get the filename
  std::string outFilename = filestem + "-CLEANED.tif";

  // Set the output file headers
  libtiff::TIFF *MultiPageTiffOut = libtiff::TIFFOpen(outFilename.c_str(), "w");
  libtiff::TIFFSetField(MultiPageTiffOut, TIFFTAG_IMAGEWIDTH, tiffWidth);
  libtiff::TIFFSetField(MultiPageTiffOut, TIFFTAG_IMAGELENGTH, tiffHeight);
  libtiff::TIFFSetField(MultiPageTiffOut, TIFFTAG_BITSPERSAMPLE, 16);

  // Write the file
  if (!MultiPageTiffOut)
  {
    pguresvt::print(std::cerr, "**WARNING** File ", outFilename, " could not be written");
    return -1;
  }

  for (size_t tOut = 0; tOut < nImages; tOut++)
  {
    libtiff::TIFFSetField(MultiPageTiffOut, TIFFTAG_IMAGEWIDTH, tiffWidth);
    libtiff::TIFFSetField(MultiPageTiffOut, TIFFTAG_IMAGELENGTH, tiffHeight);
    libtiff::TIFFSetField(MultiPageTiffOut, TIFFTAG_BITSPERSAMPLE, 16);
    libtiff::TIFFSetField(MultiPageTiffOut, TIFFTAG_SAMPLESPERPIXEL, 1);
    libtiff::TIFFSetField(MultiPageTiffOut, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    libtiff::TIFFSetField(MultiPageTiffOut, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
    libtiff::TIFFSetField(MultiPageTiffOut, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
    libtiff::TIFFSetField(MultiPageTiffOut, TIFFTAG_SUBFILETYPE, FILETYPE_PAGE);
    libtiff::TIFFSetField(MultiPageTiffOut, TIFFTAG_PAGENUMBER, tOut, nImages);

    for (size_t tiffRow = 0; tiffRow < tiffHeight; tiffRow++)
    {
      arma::Mat<uint16_t> outSlice = outTiff.slice(tOut);
      inplace_trans(outSlice);
      uint16_t *OutBuffer = outSlice.memptr();
      libtiff::TIFFWriteScanline(MultiPageTiffOut, &OutBuffer[tiffRow * tiffWidth], tiffRow, 0);
    }
    libtiff::TIFFWriteDirectory(MultiPageTiffOut);
  }
  libtiff::TIFFClose(MultiPageTiffOut);

  // Export TIFF timer
  auto t2End = std::chrono::high_resolution_clock::now();
  auto t2Elapsed = std::chrono::duration_cast<std::chrono::microseconds>(t2End - t2Start);
  pguresvt::print_fixed(5, "TIFF export took ", t2Elapsed.count() * 1E-6, " seconds\n");

  return 0;
}
