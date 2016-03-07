/***************************************************************************

	PGURE-SVT Denoising

	Author:	Tom Furnival	
	Email:	tjof2@cam.ac.uk

	Copyright (C) 2015 Tom Furnival

	This program uses Singular Value Thresholding (SVT) [1], combined 
	with an unbiased risk estimator (PGURE) to denoise a video sequence 
	of microscopy images [2]. Noise parameters for a mixed Poisson-Gaussian
	noise model are automatically estimated during the denoising.

	References:
	[1] 	"Unbiased Risk Estimates for Singular Value Thresholding and
			Spectral Estimators", (2013), Candes, EJ et al.
			http://dx.doi.org/10.1109/TSP.2013.2270464  

	[2]		"An Unbiased Risk Estimator for Image Denoising in the Presence 
			of Mixed Poisson–Gaussian Noise", (2014), Le Montagner, Y et al.
			http://dx.doi.org/10.1109/TIP.2014.2300821 

***************************************************************************/

// C++ headers
#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip> 
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <stdarg.h>
#include <stdexcept>
#include <stdlib.h>
#include <string>
#include <vector>

// OpenMP library
#include <omp.h>

// Armadillo library
#include <armadillo>

// NLopt library
#include <nlopt.h>

// LibTIFF
namespace libtiff {
    #include "tiffio.h"
}

// Constant-time median filter
extern "C" {
	#include "medfilter.h"
}


// Own headers
#include "arps.hpp"
#include "params.hpp"
#include "tools.hpp"
#include "noise.hpp"
#include "pgure.hpp"
#include "svt.hpp"

/**************************************************************************/

// Main program
int main(int argc, char** argv) {

	// Overall program timer
	auto overallstart = std::chrono::steady_clock::now();
			
	// Print program header
	std::cout<<std::endl;
	std::cout<<"PGURE-SVT Denoising"<<std::endl;
	std::cout<<"Author: Tom Furnival"<<std::endl;
	std::cout<<"Email:  tjof2@cam.ac.uk"<<std::endl<<std::endl;
	std::cout<<"Version 0.1 - December 2015"<<std::endl<<std::endl;
	
	/////////////////////////////		
	//						   //
	//    PARAMETER IMPORT     //
	//						   //
	/////////////////////////////

	// Read in the parameter file name
    if( argc != 2) {
		std::cout<<"  Usage: ./PGURE-SVT paramfile"<<std::endl;
		return -1;
    }  
    std::map<std::string, std::string> programOptions;    
	std::ifstream paramFile(argv[1], std::ios::in);
	
	// Parse the parameter file
	ParseParameters(paramFile, programOptions);
	
	// Check all required parameters are specified
	if(programOptions.count("filename") == 0 || programOptions.count("start_image") == 0 || programOptions.count("end_image") == 0) {
		std::cout<<"**WARNING** Required parameters not specified"<<std::endl;
		std::cout<<"            You must specify filename, start and end frame"<<std::endl;
		return -1;
	}
	
	// Extract parameters
	// File path
	std::string filename = programOptions.at("filename");
	int lastindex = filename.find_last_of("."); 
	std::string filestem = filename.substr(0, lastindex);
	
	// Frames to process
	int startimg = std::stoi(programOptions.at("start_image"));
	int endimg = std::stoi(programOptions.at("end_image"));
	int num_images = endimg - startimg + 1;
	
	// Move onto optional parameters	
	// Patch size and trajectory length
	// Check sizes to ensure SVD is done right way round
	int Bs = (programOptions.count("patch_size") == 1) ? std::stoi(programOptions.at("patch_size")) : 4;
	//int Overlap = (programOptions.count("patch_overlap") == 1) ? std::stoi(programOptions.at("patch_overlap")) : 1;
	int T = (programOptions.count("trajectory_length") == 1) ? std::stoi(programOptions.at("trajectory_length")) : 15;
	T = (Bs*Bs<T) ? (Bs*Bs)-1 : T;
	std::string casoratisize = std::to_string(Bs*Bs) + "x" + std::to_string(T);
		
	// Motion estimation
	bool MotionEst = (programOptions.count("motion_estimation") == 1) ? strToBool(programOptions.at("motion_estimation")) : true;

	// Noise parameters initialized at -1 unless user-defined
	double alpha = (programOptions.count("alpha") == 1) ? std::stod(programOptions.at("alpha")) : -1.;
	double mu = (programOptions.count("mu") == 1) ? std::stod(programOptions.at("mu")) : -1.;
	double sigma = (programOptions.count("sigma") == 1) ? std::stod(programOptions.at("sigma")) : -1.;

	// SVT thresholds and noise parameters initialized at -1 unless user-defined
	bool pgureOpt = (programOptions.count("pgure") == 1) ? strToBool(programOptions.at("pgure")) : true;
	double lambda;
	if(!pgureOpt) {
		if(programOptions.count("lambda") == 1) {
			lambda = std::stod(programOptions.at("lambda"));
		}
		else {
			std::cout<<"**WARNING** PGURE optimization is turned OFF but no lambda specified in parameter file"<<std::endl;
			return -1;
		}
	}

	// Move onto advanced parameters
	// Verbosity
	bool Verbose = (programOptions.count("verbose") == 1) ? strToBool(programOptions.at("verbose")) : false;
		
	// Motion neigbourhood size
	int MotionP = (programOptions.count("motion_neighbourhood") == 1) ? std::stoi(programOptions.at("motion_neighbourhood")) : 7;
	
	// Size of median filter
	int MedianSize = (programOptions.count("median_filter") == 1) ? std::stoi(programOptions.at("median_filter")) : 5;
		
	// PGURE tolerance
	double tol = 1E-7;
	if(programOptions.count("tolerance") == 1) {
		std::istringstream osTol(programOptions.at("tolerance"));
		double tol;
		osTol >> tol;
	}
	
	// Print parameters and information
	std::cout<<std::setw(40)<<std::string(40,'-')<<std::endl;
	std::cout<<std::left<<std::setw(30)<<"Parameters"<<std::setw(10)<<"Value"<<std::endl;
	std::cout<<std::setw(40)<<std::string(40,'-')<<std::endl;
	std::cout<<std::left<<std::setw(30)<<"Number of frames"<<std::setw(10)<<num_images<<std::endl;
	std::cout<<std::left<<std::setw(30)<<"Image patch size"<<std::setw(10)<<Bs<<std::endl;
	std::cout<<std::left<<std::setw(30)<<"Trajectory length"<<std::setw(10)<<T<<std::endl;	
	std::cout<<std::left<<std::setw(30)<<"Casorati dimensions"<<std::setw(10)<<casoratisize<<std::endl;
	std::cout<<std::endl;
	std::cout<<std::left<<std::setw(30)<<"Automatic noise estimation"<<std::setw(10);
	if(alpha < 0. || mu < 0. || sigma < 0.) {
		std::cout<<"ON"<<std::endl;
	}
	else {
		std::cout<<"OFF"<<std::endl;
	}	
	std::cout<<std::left<<std::setw(30)<<"PGURE optimization"<<std::setw(10);
	if(pgureOpt) {
		std::cout<<"ON"<<std::endl;
	}
	else {
		std::cout<<"OFF"<<std::endl;
	}
	std::cout<<std::left<<std::setw(30)<<"ARPS motion estimation"<<std::setw(10);
	if(MotionEst) {
		std::cout<<"ON"<<std::endl;
	}
	else {
		std::cout<<"OFF"<<std::endl;
	}
	std::cout<<std::left<<std::setw(30)<<"Verbose output"<<std::setw(10);
	if(Verbose) {
		std::cout<<"ON"<<std::endl;
	}
	else {
		std::cout<<"OFF"<<std::endl;
	}
	std::cout<<std::setw(40)<<std::string(40,'-')<<std::endl<<std::endl;
	
	/////////////////////////////		
	//						   //
	//     SEQUENCE IMPORT     //
	//						   //
	/////////////////////////////

	// Check file exists
	std::string infilename = filestem + ".tif";	
	if(!std::ifstream(infilename.c_str())) {
		std::cout<<"**WARNING** File "<<infilename<<" not found"<<std::endl;
		return -1;
	}
	
	// Load TIFF stack
	int tiffWidth, tiffHeight;
	unsigned short tiffDepth;
	libtiff::TIFF *MultiPageTiff = libtiff::TIFFOpen(infilename.c_str(), "r");
	libtiff::TIFFGetField(MultiPageTiff, TIFFTAG_IMAGEWIDTH, &tiffWidth);
	libtiff::TIFFGetField(MultiPageTiff, TIFFTAG_IMAGELENGTH, &tiffHeight);
	libtiff::TIFFGetField(MultiPageTiff, TIFFTAG_BITSPERSAMPLE, &tiffDepth); 
	
	// Only work with square images
	if(tiffWidth != tiffHeight) {
		std::cout<<"**WARNING** Frame dimensions are not square"<<std::endl;
		return -1;
	}
	// Only work with 8-bit or 16-bit images
	if(tiffDepth != 8 && tiffDepth != 16) {
		std::cout<<"**WARNING** Images must be 8-bit or 16-bit"<<std::endl;
		return -1;
	}
	
	// Import the image sequence	
	arma::cube inputsequence(tiffHeight,tiffWidth,0);
	arma::cube filteredsequence(tiffHeight,tiffWidth,0);
	if(MultiPageTiff) {
		int dircount = 0;
		int imgcount = 0;
		do {
			if(dircount >= (startimg-1) && dircount <= (endimg-1)) {		
				inputsequence.resize(tiffHeight, tiffWidth, imgcount+1);
				filteredsequence.resize(tiffHeight, tiffWidth, imgcount+1);
			
				unsigned short *Buffer = new unsigned short[tiffWidth*tiffHeight];
				unsigned short *FilteredBuffer = new unsigned short[tiffWidth*tiffHeight];
			
				for(int tiffRow = 0; tiffRow < tiffHeight; tiffRow++) {
		       		libtiff::TIFFReadScanline(MultiPageTiff, &Buffer[tiffRow*tiffWidth], tiffRow, 0);
		    	}  
		    	        	
		    	arma::Mat<unsigned short> TiffSlice( Buffer, tiffHeight, tiffWidth);
		    	inplace_trans(TiffSlice);
		    	inputsequence.slice(imgcount) = arma::conv_to<arma::mat>::from(TiffSlice);
		    	
		    	// Apply median filter (constant-time) to the 8-bit image        	
		    	int memsize = 512 * 1024;	// L2 cache size
		    	int filtsize = MedianSize;	// Median filter size in pixels
				ConstantTimeMedianFilter(Buffer, FilteredBuffer, tiffWidth, tiffHeight, tiffWidth, tiffWidth, filtsize, 1, memsize);
		    	arma::Mat<unsigned short> FilteredTiffSlice( FilteredBuffer, tiffHeight, tiffWidth);
		    	inplace_trans(FilteredTiffSlice);
		    	filteredsequence.slice(imgcount) = arma::conv_to<arma::mat>::from(FilteredTiffSlice);       			
       			imgcount++;
       		}
    		dircount++;
		} while(libtiff::TIFFReadDirectory(MultiPageTiff));
		libtiff::TIFFClose(MultiPageTiff);
	}	
	// Is number of frames compatible?
	if(num_images > (int)inputsequence.n_slices) {
		std::cout<<"**WARNING** Sequence only has "<<inputsequence.n_slices<<" frames"<<std::endl;
		return -1;
	}	

	// Copy image sequence and sizes
	arma::cube noisysequence = inputsequence;
	arma::cube cleansequence = inputsequence;
	cleansequence.zeros();
	
	/////////////////////////////		
	//						   //
	//    MEMORY ALLOCATION    //
	//						   //
	/////////////////////////////
	
	// Get dimensions
	int Nx = tiffHeight;
	int Ny = tiffWidth;

	// Pre-allocate memory for large arrays of Casorati SVDs
	int aDim, bDim;
	aDim = Bs*Bs;
	bDim = T;
	int vecSize = (Nx-Bs+1)*(Ny-Bs+1);		
	std::vector<arma::mat> Uall, Vall, U1all, V1all, U2pall, V2pall, U2mall, V2mall;
	std::vector<arma::vec> Sall, S1all, S2pall, S2mall;	
	Uall.resize(vecSize);
	Vall.resize(vecSize);
	Sall.resize(vecSize);
	U1all.resize(vecSize);
	V1all.resize(vecSize);
	S1all.resize(vecSize);
	U2pall.resize(vecSize);
	V2pall.resize(vecSize);
	S2pall.resize(vecSize);
	U2mall.resize(vecSize);
	V2mall.resize(vecSize);
	S2mall.resize(vecSize);
	for(int it = 0; it < vecSize; it++) {
		Uall[it] = arma::zeros<arma::mat>(aDim,bDim);
		Sall[it] = arma::zeros<arma::vec>(bDim);
		Vall[it] = arma::zeros<arma::mat>(bDim,bDim);
		U1all[it] = arma::zeros<arma::mat>(aDim,bDim);
		S1all[it] = arma::zeros<arma::vec>(bDim);
		V1all[it] = arma::zeros<arma::mat>(bDim,bDim);
		U2pall[it] = arma::zeros<arma::mat>(aDim,bDim);
		S2pall[it] = arma::zeros<arma::vec>(bDim);
		V2pall[it] = arma::zeros<arma::mat>(bDim,bDim);
		U2mall[it] = arma::zeros<arma::mat>(aDim,bDim);
		S2mall[it] = arma::zeros<arma::vec>(bDim);
		V2mall[it] = arma::zeros<arma::mat>(bDim,bDim);
	}	
	
	/////////////////////////////		
	//						   //
	//     START THE LOOP      //
	//						   //
	/////////////////////////////
	
	// Print table headings
	int ww = 10;
	std::cout<<std::endl;
	std::cout<<std::right<<std::setw(5*ww+5)<<std::string(5*ww+5,'-')<<std::endl;
	std::cout<<std::setw(5)<<"Frame"<<std::setw(ww)<<"Gain"<<std::setw(ww)<<"Offset"<<std::setw(ww)<<"Sigma"<<std::setw(ww)<<"Lambda"<<std::setw(ww)<<"Time (s)"<<std::endl;
	std::cout<<std::setw(5*ww+5)<<std::string(5*ww+5,'-')<<std::endl;

	// Initialise some more matrices
	int framewindow = std::floor(T/2);	
	arma::icube sequencePatches = arma::zeros<arma::icube>(2,vecSize,2*framewindow+1);
	arma::icube sequenceMotions = arma::zeros<arma::icube>(2,vecSize,2*framewindow);		

	// Loop over time windows	
	for(int timeiter = 0; timeiter < num_images; timeiter++) {
	
		// Time the loop iteration
		auto startLoopTimer = std::chrono::steady_clock::now();	
		
		// Verbose timer
		auto verboseStart = std::chrono::steady_clock::now();	
		
		// Extract the subset of the image sequence
		arma::cube u(Nx, Ny, T), ufilter(Nx, Ny, T), v(Nx, Ny, T);		
		if(timeiter < framewindow) {
			u = noisysequence.slices(0,2*framewindow);
			ufilter = filteredsequence.slices(0,2*framewindow);	
		}
		else if(timeiter >= (num_images - framewindow)) {
			u = noisysequence.slices(num_images-2*framewindow-1,num_images-1);
			ufilter = filteredsequence.slices(num_images-2*framewindow-1,num_images-1);
		}
		else {
			u = noisysequence.slices(timeiter - framewindow, timeiter + framewindow);
			ufilter = filteredsequence.slices(timeiter - framewindow, timeiter + framewindow);
		}
						
		// Basic sequence normalization
		double inputmax = u.max();
		u /= inputmax;
		ufilter /= ufilter.max();	
		
		// Finish timing
		auto verboseEnd = std::chrono::steady_clock::now();
		auto verboseElapsed = std::chrono::duration_cast<std::chrono::microseconds>(verboseEnd - verboseStart);
		if(Verbose) {
			std::cout<<"          Preparation: "<<std::fixed<<std::setprecision(5)<<(verboseElapsed.count()/1E6)<<" s"<<std::endl;	
		}
					
		/////////////////////////////
		//						   //
		//    NOISE ESTIMATION     //
		//						   //
		/////////////////////////////
		
		// Verbose timer
		verboseStart = std::chrono::steady_clock::now();
		
		// Option to return Quadtree 
		// (e.g. for export)
		arma::cube quadtree(Nx,Ny,T);		
	
		// Perform noise estimation
		EstimateNoiseParams(&u, &quadtree, &alpha, &mu, &sigma, 8);
		
		// Finish timing
		verboseEnd = std::chrono::steady_clock::now();
		verboseElapsed = std::chrono::duration_cast<std::chrono::microseconds>(verboseEnd - verboseStart);
		if(Verbose) {
			std::cout<<"     Noise estimation: "<<std::fixed<<std::setprecision(5)<<(verboseElapsed.count()/1E6)<<" s"<<std::endl;	
		}

		/////////////////////////////
		//						   //
		//    MOTION ESTIMATION    //
		//						   //
		/////////////////////////////
		
		// Verbose timer
		verboseStart = std::chrono::steady_clock::now();
		
		// Perform motion estimation	
		if(timeiter < framewindow) {		
			// Populate reference frame coordinates
			for(int i = 0; i < vecSize; i++) {
				sequencePatches(0,i,timeiter) = i % (Ny-Bs+1);
				sequencePatches(1,i,timeiter) = i / (Nx-Bs+1);		
			}		
			
			if(MotionEst) {
				// Perform motion estimation			
				// Go forwards		
				for(int i = 0; i < T-timeiter-1; i++) {		
					ARPSMotionEstimation(&(ufilter.slice(timeiter+i)), &(ufilter.slice(timeiter+i+1)), &(sequencePatches.slice(timeiter+i)), &(sequencePatches.slice(timeiter+i+1)), &(sequenceMotions.slice(timeiter+i)), i, Bs, MotionP);
				}
				// Go backwards
				for(int i = -1; i >= -timeiter; i--) {		
					ARPSMotionEstimation(&(ufilter.slice(timeiter+i+1)), &(ufilter.slice(timeiter+i)), &(sequencePatches.slice(timeiter+i+1)), &(sequencePatches.slice(timeiter+i)), &(sequenceMotions.slice(timeiter+i+1)), i, Bs, MotionP);
				}
			}
			else {
				// Don't perform motion estimation	
				for(int i = 0; i < T; i++){
					sequencePatches.slice(i) = sequencePatches.slice(timeiter);
				}
			}			
		}
		else if(timeiter >= (num_images - framewindow)) {
			int endseqFrame = timeiter - (num_images - T);
			// Populate reference frame coordinates
			for(int i = 0; i < vecSize; i++) {
				sequencePatches(0,i,endseqFrame) = i % (Ny-Bs+1);
				sequencePatches(1,i,endseqFrame) = i / (Nx-Bs+1);		
			}
			
			if(MotionEst) {
				// Perform motion estimation			
				// Go forwards
				for(int i = 0; i < 2*framewindow - endseqFrame; i++) {
					ARPSMotionEstimation(&(ufilter.slice(endseqFrame+i)), &(ufilter.slice(endseqFrame+i+1)), &(sequencePatches.slice(endseqFrame+i)), &(sequencePatches.slice(endseqFrame+i+1)), &(sequenceMotions.slice(endseqFrame+i)), i, Bs, MotionP);
				}
				// Go backwards
				for(int i = -1; i >= -endseqFrame; i--) {
					if(2*framewindow == endseqFrame) {
						ARPSMotionEstimation(&(ufilter.slice(endseqFrame+i+1)), &(ufilter.slice(endseqFrame+i)), &(sequencePatches.slice(endseqFrame+i+1)), &(sequencePatches.slice(endseqFrame+i)), &(sequenceMotions.slice(endseqFrame+i)), i, Bs, MotionP);
					}
					else {
						ARPSMotionEstimation(&(ufilter.slice(endseqFrame+i+1)), &(ufilter.slice(endseqFrame+i)), &(sequencePatches.slice(endseqFrame+i+1)), &(sequencePatches.slice(endseqFrame+i)), &(sequenceMotions.slice(endseqFrame+i+1)), i, Bs, MotionP);
					}
				}
			}
			else {
				// Don't perform motion estimation	
				for(int i = 0; i < T; i++){
					sequencePatches.slice(i) = sequencePatches.slice(endseqFrame);
				}
			}
		}
		else {
			// Populate reference frame coordinates
			for(int i = 0; i < vecSize; i++) {
				sequencePatches(0,i,framewindow) = i % (Ny-Bs+1);
				sequencePatches(1,i,framewindow) = i / (Nx-Bs+1);		
			}	
			
			if(MotionEst) {
				// Perform motion estimation			
				// Go forwards	
				for(int i = 0; i < framewindow; i++) {		
					ARPSMotionEstimation(&(ufilter.slice(framewindow+i)), &(ufilter.slice(framewindow+i+1)), &(sequencePatches.slice(framewindow+i)), &(sequencePatches.slice(framewindow+i+1)), &(sequenceMotions.slice(framewindow+i)), i, Bs, MotionP);
				}
				// Go backwards
				for(int i = -1; i >= -framewindow; i--) {		
					ARPSMotionEstimation(&(ufilter.slice(framewindow+i+1)), &(ufilter.slice(framewindow+i)), &(sequencePatches.slice(framewindow+i+1)), &(sequencePatches.slice(framewindow+i)), &(sequenceMotions.slice(framewindow+i+1)), i, Bs, MotionP);
				}
			}
			else {
				// Don't perform motion estimation	
				for(int i = 0; i < T; i++){
					sequencePatches.slice(i) = sequencePatches.slice(framewindow);
				}
			}
		}	
		
		// Finish timing
		verboseEnd = std::chrono::steady_clock::now();
		verboseElapsed = std::chrono::duration_cast<std::chrono::microseconds>(verboseEnd - verboseStart);
		if(Verbose) {
			std::cout<<"    Motion estimation: "<<std::fixed<<std::setprecision(5)<<(verboseElapsed.count()/1E6)<<" s"<<std::endl;	
		}			
			
		/////////////////////////////
		//						   //
		//   PGURE OPTIMIZATION    //
		//						   //
		/////////////////////////////
		
		// Reshape to (n^2 x T) Casorati matrix
		arma::mat G(Nx*Ny, T);
		CubeReshape(&u, &G);
						
		// Specify perturbations
		double eps1 = G.max() * 1E-4;
		double eps2 = G.max() * 1E-2;

		// Generate random samples for stochastic evaluation
		arma::cube delta1 = GenerateRandomPerturbation(1, Nx, Ny, T);
		arma::cube delta2 = GenerateRandomPerturbation(2, Nx, Ny, T);
		arma::cube u1 = u + (delta1 * eps1);
		arma::cube u2p = u + (delta2 * eps2);
		arma::cube u2m = u - (delta2 * eps2);

		// Verbose timer
		verboseStart = std::chrono::steady_clock::now();

		// Do the block SVDs
		SVDDecompose(&u, &Uall, &Sall, &Vall, &sequencePatches, Bs, Nx, Ny, T);
		SVDDecompose(&u1, &U1all, &S1all, &V1all, &sequencePatches, Bs, Nx, Ny, T);
		SVDDecompose(&u2p, &U2pall, &S2pall, &V2pall, &sequencePatches, Bs, Nx, Ny, T);
		SVDDecompose(&u2m, &U2mall, &S2mall, &V2mall, &sequencePatches, Bs, Nx, Ny, T);
		
		// Finish timing
		verboseEnd = std::chrono::steady_clock::now();
		verboseElapsed = std::chrono::duration_cast<std::chrono::microseconds>(verboseEnd - verboseStart);
		if(Verbose) {			
			std::cout<<"         Casorati SVD: "<<std::fixed<<std::setprecision(5)<<(verboseElapsed.count()/1E6)<<" s";	
			std::cout<<" ("<<std::fixed<<std::setprecision(0)<<(int)(Uall.size()/(verboseElapsed.count()/1E6))<<" SVDs / second)"<<std::endl;	
		}
		
		// Load parameters for optimization routine
		struct PGureSearchParameters PGUREparams;
		PGUREparams.Nx = Nx;
		PGUREparams.Ny = Ny;
		PGUREparams.Bs = Bs;
		PGUREparams.T = T;

		// Number of function evaluations
		PGUREparams.count = 0;

		// Mixed noise parameters
		PGUREparams.alpha = alpha;
		PGUREparams.sigma = sigma;
		PGUREparams.mu = mu;

		// Perturbation amplitudes
		PGUREparams.eps1 = eps1;
		PGUREparams.eps2 = eps2;

		// Measured signal as (n^2 x T) Casorati matrix				
		PGUREparams.G = &G;

		// Random samples for stochastic evaluation
		PGUREparams.delta1 = &delta1;
		PGUREparams.delta2 = &delta2;			

		// SVD results for SVT thresholding and reconstruction
		PGUREparams.U = &Uall;
		PGUREparams.S = &Sall;
		PGUREparams.V = &Vall;
		PGUREparams.U1 = &U1all;
		PGUREparams.S1 = &S1all;
		PGUREparams.V1 = &V1all;
		PGUREparams.U2p = &U2pall;
		PGUREparams.S2p = &S2pall;
		PGUREparams.V2p = &V2pall;
		PGUREparams.U2m = &U2mall;
		PGUREparams.S2m = &S2mall;
		PGUREparams.V2m = &V2mall;
		PGUREparams.sequencePatches = &sequencePatches;

		// Verbose timer
		verboseStart = std::chrono::steady_clock::now();

		// Determine optimum threshold value (max 1000 evaluations)		
		if(pgureOpt) {
			double start = (timeiter == 0) ? arma::mean(arma::mean(G)) : lambda;
			double riskval = 0.;
			PGUREOptimize(&lambda, &riskval, &PGUREparams, tol, start, G.max(), 1E4);
		}
		
		// Finish timing
		verboseEnd = std::chrono::steady_clock::now();
		verboseElapsed = std::chrono::duration_cast<std::chrono::microseconds>(verboseEnd - verboseStart);
		if(Verbose) {			
			std::cout<<"   PGURE Optimization: "<<std::fixed<<std::setprecision(5)<<(verboseElapsed.count()/1E6)<<" s";	
			std::cout<<" ("<<PGUREparams.count<<" iterations)"<<std::endl;	
		}
		
		/////////////////////////////
		//						   //
		// SEQUENCE RECONSTRUCTION //
		//						   //
		/////////////////////////////
		
		// Verbose timer
		verboseStart = std::chrono::steady_clock::now();

		// Reconstruct the blocks into the denoised sequence
		SVDReconstruct(lambda, &v, &Uall, &Sall, &Vall, &sequencePatches, Bs, Nx, Ny, T);
		
		// Rescale back to original range
		v *= inputmax;
		
		// Place frame back into sequence		
		if(timeiter < framewindow) {
			cleansequence.slice(timeiter) = v.slice(timeiter);
		}
		else if(timeiter >= (num_images - framewindow)) {
			int endseqFrame = timeiter - (num_images - T);
			cleansequence.slice(timeiter) = v.slice(endseqFrame);
		}
		else {
			cleansequence.slice(timeiter) = v.slice(framewindow);
		}
		
		// Finish timing
		verboseEnd = std::chrono::steady_clock::now();
		verboseElapsed = std::chrono::duration_cast<std::chrono::microseconds>(verboseEnd - verboseStart);
		if(Verbose) {
			std::cout<<"       Reconstruction: "<<std::fixed<<std::setprecision(5)<<(verboseElapsed.count()/1E6)<<" s"<<std::endl;	
		}
	
		// Finish timing
		auto endLoopTimer = std::chrono::steady_clock::now();
		auto elapsedLoopTimer = std::chrono::duration_cast<std::chrono::microseconds>(endLoopTimer - startLoopTimer);

		// Output a table row with noise estimates, lambda and timing
		std::ostringstream framestring;
		framestring << timeiter+1;
		
		if(Verbose) {
			std::cout<<std::endl;
		}
		
		std::cout<<std::fixed<<std::setw(5)<<framestring.str()<<std::setw(ww)<<std::setprecision(3)<<alpha<<std::setw(ww)<<std::setprecision(3)<<mu<<std::setw(ww)<<std::setprecision(3)<<sigma<<std::setw(ww)<<std::setprecision(3)<<lambda<<std::setw(ww)<<std::setprecision(3)<<(elapsedLoopTimer.count()/1E6)<<std::endl;
		
		if(Verbose) {
			std::cout<<std::endl;
		}
	
	}

	// Finish the table off
	std::cout<<std::setw(5*ww+5)<<std::string(5*ww+5,'-')<<std::endl<<std::endl;
	
	/////////////////////////////
	//						   //
	//      SAVE SEQUENCE      //
	//						   //
	/////////////////////////////	

	// Normalize to [0,65535] range 
	cleansequence = (cleansequence - cleansequence.min())/(cleansequence.max() - cleansequence.min());
	arma::Cube<unsigned short> outTiff(tiffWidth,tiffHeight,num_images);
	outTiff = arma::conv_to<arma::Cube<unsigned short>>::from(65535*cleansequence);	
	
	// Get the filename
	std::string outfilename = filestem + "-CLEANED.tif";
	
	// Set the output file headers
	libtiff::TIFF *MultiPageTiffOut = libtiff::TIFFOpen(outfilename.c_str(), "w");
	libtiff::TIFFSetField(MultiPageTiffOut, TIFFTAG_IMAGEWIDTH, tiffWidth);
	libtiff::TIFFSetField(MultiPageTiffOut, TIFFTAG_IMAGELENGTH, tiffHeight);
	libtiff::TIFFSetField(MultiPageTiffOut, TIFFTAG_BITSPERSAMPLE, 16);
	
	// Write the file
	if(!MultiPageTiffOut) {
		std::cout<<"**WARNING** File "<<outfilename<<" could not be written"<<std::endl;
		return -1;
	}
		
	for(int tOut = 0; tOut < num_images; tOut++) {
		libtiff::TIFFSetField(MultiPageTiffOut, TIFFTAG_IMAGEWIDTH, tiffWidth);
       	libtiff::TIFFSetField(MultiPageTiffOut, TIFFTAG_IMAGELENGTH, tiffHeight);
    	libtiff::TIFFSetField(MultiPageTiffOut, TIFFTAG_BITSPERSAMPLE, 16);
    	libtiff::TIFFSetField(MultiPageTiffOut, TIFFTAG_SAMPLESPERPIXEL, 1);
    	libtiff::TIFFSetField(MultiPageTiffOut, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    	libtiff::TIFFSetField(MultiPageTiffOut, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
        libtiff::TIFFSetField(MultiPageTiffOut, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);	
    	
		libtiff::TIFFSetField(MultiPageTiffOut, TIFFTAG_SUBFILETYPE, FILETYPE_PAGE);	
		libtiff::TIFFSetField(MultiPageTiffOut, TIFFTAG_PAGENUMBER, tOut, num_images);		
		for(int tiffRow = 0; tiffRow < tiffHeight; tiffRow++) {
			arma::Mat<unsigned short> outSlice = outTiff.slice(tOut);
	    	inplace_trans(outSlice);			
			unsigned short *OutBuffer = outSlice.memptr();			
    		libtiff::TIFFWriteScanline(MultiPageTiffOut, &OutBuffer[tiffRow*tiffWidth], tiffRow, 0);
	   	}
	   	libtiff::TIFFWriteDirectory(MultiPageTiffOut);
  	}
  	libtiff::TIFFClose(MultiPageTiffOut);
	
	/////////////////////////////
	//						   //
	//      REPORT RESULT      //
	//						   //
	/////////////////////////////
	
	// Overall program timer
	auto overallend = std::chrono::steady_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(overallend - overallstart);
	std::cout<<"Total time: "<<std::setprecision(5)<<(elapsed.count()/1E6)<<" seconds"<<std::endl<<std::endl;	

	return 0;
}



