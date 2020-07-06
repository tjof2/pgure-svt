/**********************
# Author: Tom Furnival
# License: GPLv3
***********************/

#include <fstream>
#include <string>

namespace libtiff
{
#include "tiffio.h"
}

#include "hotpixel.hpp"
#include "utils.hpp"
#include "pguresvt.hpp"

int main(int argc, char **argv)
{
  std::chrono::high_resolution_clock::time_point t0, t1;

  pguresvt::Print(std::cout,
                  "PGURE-SVT Denoising\n",
                  "Author: Tom Furnival\n",
                  "Email:  tjof2@cam.ac.uk\n");

  if (argc != 2) // Print usage if parameter file not present as argument
  {
    pguresvt::Print(std::cout, "  Usage: ./PGURE-SVT paramfile");
    return -1;
  }

  std::map<std::string, std::string> opts;
  std::ifstream paramFile(argv[1], std::ios::in);

  pguresvt::ParseParameters(paramFile, opts);

  if (opts.count("filename") == 0 ||
      opts.count("start_frame") == 0 ||
      opts.count("end_frame") == 0) // Check all required parameters are specified
  {
    pguresvt::Print(std::cerr, "**ERROR**\n", "Required parameters not specified\n",
                    "You must specify 'filename', 'start_frame' and 'end_frame'\n");
    return -1;
  }

  // Extract parameters
  std::string filename = opts.at("filename");
  std::string filestem = filename.substr(0, filename.find_last_of("."));

  uint32_t startImage = std::stoi(opts.at("start_frame"));
  uint32_t endImage = std::stoi(opts.at("end_frame"));
  uint32_t nImages = endImage - startImage + 1;

  uint32_t blockSize = (opts.count("patch_size") == 1) ? std::stoi(opts.at("patch_size")) : 4;
  uint32_t trajLength = (opts.count("trajectory_length") == 1) ? std::stoi(opts.at("trajectory_length")) : 15;
  uint32_t motionWindow = (opts.count("motion_neighbourhood") == 1) ? std::stoi(opts.at("motion_neighbourhood")) : 7;
  uint32_t medianSize = (opts.count("median_filter") == 1) ? std::stoi(opts.at("median_filter")) : 5;
  uint32_t blockOverlap = (opts.count("patch_overlap") == 1) ? std::stoi(opts.at("patch_overlap")) : 1;
  uint32_t noiseMethod = (opts.count("noise_method") == 1) ? std::stoi(opts.at("noise_method")) : 4;
  uint32_t maxIter = (opts.count("max_iter") == 1) ? std::stoi(opts.at("max_iter")) : 1000;
  int nJobs = (opts.count("n_jobs") == 1) ? std::stoi(opts.at("n_jobs")) : -1;

  // Noise parameters (initialized at -1 unless user-defined)
  double alpha = (opts.count("noise_alpha") == 1) ? std::stod(opts.at("noise_alpha")) : -1.;
  double mu = (opts.count("noise_mu") == 1) ? std::stod(opts.at("noise_mu")) : -1.;
  double sigma = (opts.count("noise_sigma") == 1) ? std::stod(opts.at("noise_sigma")) : -1.;

  double hotPixelThreshold = (opts.count("hot_pixel") == 1) ? std::stoi(opts.at("hot_pixel")) : -1.0;
  int randomSeed = (opts.count("random_seed") == 1) ? std::stoi(opts.at("random_seed")) : -1;
  bool expWeighting = (opts.count("exponential_weighting") == 1) ? pguresvt::StrToBool(opts.at("exponential_weighting")) : true;
  bool normalizeImg = (opts.count("normalize") == 1) ? pguresvt::StrToBool(opts.at("normalize")) : false;
  bool motionEstimation = (opts.count("motion_estimation") == 1) ? pguresvt::StrToBool(opts.at("motion_estimation")) : true;

  // SVT threshold (initialized at -1 unless user-defined)
  bool optPGURE = (opts.count("optimize_pgure") == 1) ? pguresvt::StrToBool(opts.at("optimize_pgure")) : true;
  double lambda = 0.0;
  if (!optPGURE)
  {
    if (opts.count("lambda") == 1)
    {
      lambda = std::stod(opts.at("lambda"));
    }
    else
    {
      pguresvt::Print(std::cerr,
                      "**ERROR**\nPGURE optimization is turned OFF but ",
                      "no lambda specified in parameter file\n");
      return -1;
    }
  }

  double tol = 1E-7;
  if (opts.count("tolerance") == 1) // PGURE tolerance parsing
  {
    std::istringstream osTol(opts.at("tolerance"));
    double tol;
    osTol >> tol;
  }

  t0 = std::chrono::high_resolution_clock::now(); // Time TIFF import

  std::string inFilename = filestem + ".tif";
  if (!std::ifstream(inFilename.c_str())) // Check file exists
  {
    pguresvt::Print(std::cerr, "**ERROR**\nFile ", inFilename, " not found\n");
    return -1;
  }

  uint32_t tiffWidth, tiffHeight;
  uint16_t tiffDepth;
  libtiff::TIFF *MultiPageTiff = libtiff::TIFFOpen(inFilename.c_str(), "r");
  libtiff::TIFFGetField(MultiPageTiff, TIFFTAG_IMAGEWIDTH, &tiffWidth);
  libtiff::TIFFGetField(MultiPageTiff, TIFFTAG_IMAGELENGTH, &tiffHeight);
  libtiff::TIFFGetField(MultiPageTiff, TIFFTAG_BITSPERSAMPLE, &tiffDepth);

  if (tiffWidth != tiffHeight) // Only work with square images
  {
    pguresvt::Print(std::cerr, "**ERROR**\nFrame dimensions are not square, got ", tiffWidth, "x", tiffHeight, "\n");
    return -1;
  }

  if (tiffDepth != 8 && tiffDepth != 16) // Only work with 8-bit or 16-bit images
  {
    pguresvt::Print(std::cerr, "**ERROR**\nImages must be 8-bit or 16-bit, got ", tiffDepth, "-bit depth \n");
    return -1;
  }

  arma::Cube<uint16_t> inputSeq(tiffHeight, tiffWidth, 0);

  if (MultiPageTiff) // Import the image sequence
  {
    uint32_t dircount = 0;
    uint32_t imgcount = 0;

    do
    {
      if ((dircount >= (startImage - 1)) && (dircount < endImage))
      {
        inputSeq.resize(tiffHeight, tiffWidth, imgcount + 1);

        uint16_t *Buffer = new uint16_t[tiffWidth * tiffHeight];

        for (size_t tiffRow = 0; tiffRow < tiffHeight; tiffRow++)
        {
          libtiff::TIFFReadScanline(MultiPageTiff, &Buffer[tiffRow * tiffWidth], tiffRow, 0);
        }

        arma::Mat<uint16_t> curSlice(Buffer, tiffHeight, tiffWidth);
        inplace_trans(curSlice);
        inputSeq.slice(imgcount) = curSlice;

        imgcount++;
        delete[] Buffer;
      }
      dircount++;

    } while (libtiff::TIFFReadDirectory(MultiPageTiff));
    libtiff::TIFFClose(MultiPageTiff);
  }

  if (nImages > inputSeq.n_slices) // Is number of frames compatible?
  {
    pguresvt::Print(std::cerr, "**ERROR**\n Sequence only has ", inputSeq.n_slices, " frames, expected ", nImages, "\n");
    return -1;
  }

  t1 = std::chrono::high_resolution_clock::now();
  pguresvt::PrintFixed(4, "TIFF import:    ", std::setw(10), pguresvt::ElapsedSeconds(t0, t1), " seconds");

  if (hotPixelThreshold >= 0.0) // Initial outlier detection for hot pixels
  {
    t0 = std::chrono::high_resolution_clock::now();

    pguresvt::HotPixelFilter(inputSeq, hotPixelThreshold, nJobs);

    t1 = std::chrono::high_resolution_clock::now();
    pguresvt::PrintFixed(4, "Outlier filter: ", std::setw(10), pguresvt::ElapsedSeconds(t0, t1), " seconds");
  }

  t0 = std::chrono::high_resolution_clock::now(); // PGURE-SVT timer

  arma::cube cleanSeq;
  arma::mat res;
  uint32_t result;

  result = PGURESVT(cleanSeq, res, inputSeq,
                    trajLength, blockSize, blockOverlap, motionWindow,
                    medianSize, noiseMethod, maxIter, nJobs, randomSeed,
                    optPGURE, expWeighting, motionEstimation,
                    lambda, alpha, mu, sigma, tol);

  t1 = std::chrono::high_resolution_clock::now();
  pguresvt::PrintFixed(4, "PGURE-SVT:      ", std::setw(10), pguresvt::ElapsedSeconds(t0, t1), " seconds");

  t0 = std::chrono::high_resolution_clock::now();

  arma::Cube<uint16_t> outTiff(tiffWidth, tiffHeight, nImages);

  if (normalizeImg) // Normalize to [0,65535] range
  {
    cleanSeq = 65535 * (cleanSeq - cleanSeq.min()) / (cleanSeq.max() - cleanSeq.min());
  }
  outTiff = arma::conv_to<arma::Cube<uint16_t>>::from(cleanSeq);

  std::string outFilename = filestem + "-CLEANED.tif";                           // Get the output filename
  libtiff::TIFF *MultiPageTiffOut = libtiff::TIFFOpen(outFilename.c_str(), "w"); // Set the output file headers
  libtiff::TIFFSetField(MultiPageTiffOut, TIFFTAG_IMAGEWIDTH, tiffWidth);
  libtiff::TIFFSetField(MultiPageTiffOut, TIFFTAG_IMAGELENGTH, tiffHeight);
  libtiff::TIFFSetField(MultiPageTiffOut, TIFFTAG_BITSPERSAMPLE, 16);

  if (!MultiPageTiffOut) // Try to write the file
  {
    pguresvt::Print(std::cerr, "**ERROR**\nFile ", outFilename, " could not be written\n");
    return -1;
  }

  for (size_t tOut = 0; tOut < nImages; tOut++) // Now write the data
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

  t1 = std::chrono::high_resolution_clock::now();
  pguresvt::PrintFixed(4, "TIFF export:    ", std::setw(10), pguresvt::ElapsedSeconds(t0, t1), " seconds\n");
  pguresvt::Print(std::cerr, "Output file:    ", outFilename, "\n");

  return result;
}
