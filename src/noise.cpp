/***************************************************************************

	Copyright (C) 2015 Tom Furnival
	
	File: noise.cpp
	
	Library of noise estimation functions:
		1. Estimate noise parameters based on method in [1]
		2. Quadtree segmentation of image based on method in [2]
	
	References:
	[1] 	"Patch-Based Nonlocal Functional for Denoising Fluorescence 
			Microscopy Image Sequences", (2010), Boulanger, J et al.
			http://dx.doi.org/10.1109/TMI.2009.2033991 

	[2] 	"Deconvolution of 3D Fluorescence Micrographs with Automatic 
			Risk Minimization", (2008), Ramani, S et al.
			http://dx.doi.org/10.1109/ISBI.2008.4541100
			 	
***************************************************************************/

#include "noise.hpp"

// F-test lookup tables
arma::uvec DegOFreePlus1 = {2, 4, 8, 16, 32, 64, 128, 256};
arma::vec Ftest0100 = {5.39077,1.97222,1.38391,1.17439,1.08347,1.04087,1.02023,1.01006};
arma::vec Ftest0050 = {9.27663,2.40345,1.51833,1.22924,1.10838,1.05276,1.02604,1.01293};
arma::vec Ftest0025 = {15.4392,2.86209,1.64602,1.27893,1.13046,1.06318,1.03110,1.01543};
arma::vec Ftest0010 = {29.4567,3.52219,1.80896,1.33932,1.15670,1.07543,1.03702,1.01834};

// Discrete Laplacian operator
arma::mat laplacian = {1./8., 1./8., 1./8., 1./8., -1., 1./8., 1./8., 1./8., 1./8.};

/////////////////////////////
//						   //
//    NOISE ESTIMATION     //
//						   //
/////////////////////////////

void WeightFunction(arma::vec *x, arma::vec *w, int wtype) {
	double p = 0.;	
	switch (wtype) {
		case 0:
			p = 0.75;
			for(size_t i = 0; i < (*x).n_elem; i++) {
				(*w)(i) = (std::abs((*x)(i)) < p) ? 1. :  p / std::abs((*x)(i));
			}		
			break;
		case 1:
		default:
			p = 3.5;
			for(size_t i = 0; i < (*x).n_elem; i++) {
				(*w)(i) = (std::abs((*x)(i)) > p) ? 0. : (p*p - (*x)(i)*(*x)(i))*(p*p - (*x)(i)*(*x)(i)) / (p*p*p*p);
			}	
			break;	
	}
	return;
}

double InterqDist(arma::vec *row) {
	arma::vec sorted = sort((*row));
	int N = sorted.n_elem;
	int m = std::floor((std::floor((N+1)/2) + 1)/2);
	double diq = sorted(N-m-1) - sorted(m-1);	
	return diq;
}

double RobustVarEstimate(arma::vec *row) {
	double sig = 1.4826 * arma::median(arma::abs((*row) - arma::median((*row))));
	return sig*sig;	
}

double RobustMeanEstimate(arma::vec *row, int wtype) {
	int I = 1E4;
	int N = (*row).n_elem;
	double e = 0.;
	double tol = 1E-6;
	double d = 1.;
	double m = 0.;
	double m0 = 1E12;
	double eps = 1E-12;
	double aux = 0.;
		
	arma::vec w(N), r(N);
	w.ones();
		
	for(int i = 0; i < I; i++) {
		r = w % (*row);
		m = arma::accu(r);
		aux = arma::accu(w);
		m = (std::abs(aux) < eps) ? m0 : m / aux;
		r = (*row) - m;
		e = arma::mean(arma::abs(r));
		if(std::abs(m0 - m) < tol || e < tol) {
			break;
		}
		m0 = m;
		d = InterqDist(&r) + eps;
		r *= 1./d;
		WeightFunction(&r, &w, wtype);
	}
	return m;
}

arma::vec WLSFit(arma::vec *x, arma::vec *y, int wtype) {
	int I = 1E4;
	int N = (*x).n_elem;
	double e = 0.;
	double tol = 1E-6;
	double d = 1.;
	double a0 = 1E12;
	double b0 = 1E12;
	double eps = 1E-12;
	double aux = 0.;
	
	double sw2, sw2x, sw2y;
	
	arma::vec params(2);
	
	arma::vec w(N), w2(N), w2x(N), w2y(N), xy(N), f(N), r(N);
	w.ones();
	
	for(int i = 0; i < I; i++){
		w2 = w % w;
		w2x = w2 % (*x);
		w2y = w2 % (*y);
		xy = (*x) % (*y);
		xy = w2 % xy;
		
		sw2 = arma::accu(w2);
		sw2x = arma::accu(w2x);
		sw2y = arma::accu(w2y);
		
		params(0) = sw2 * arma::accu(xy) - sw2x*sw2y;
		xy = (*x) % (*x);
		xy = w2 % xy;
		aux = sw2 * arma::accu(xy) - sw2x*sw2x;
				
		params(0) = (std::abs(aux) < eps) ? a0 : params(0) / aux;
		params(1) = sw2y - params(0)*sw2x;
		params(1) = (std::abs(aux) < eps) ? b0 : params(1) / sw2;
		
		f = (*x) * params(0);
		f += params(1);
		
		r = (*y) - f;
		e = arma::mean(arma::abs(r));
		
		if( (std::abs(a0 - params(0)) < tol && std::abs(b0 - params[1]) < tol) || e < tol ) {
			break;
		}
		a0 = params[0];
		b0 = params[1];
		
		d = InterqDist(&r) + eps;
		r *= 1./d;
		WeightFunction(&r, &w, wtype);		
	}

	return params;
}

double ComputeMode(arma::vec a) {
	int maxCount = 0;
	int count;
	double maxValue = 0.;
	double M = a.max();
	double dyn = 1. * a.n_elem;
	a = arma::round(a * dyn/M);
	
	for(size_t i = 0; i < a.n_elem; i++) {
		count = 0;
		for(size_t j = 0; j < a.n_elem; j++) {
			if(a(j) == a(i)) {
				count++;
			}
		}
		if(count > maxCount) {
			maxCount = count;
			maxValue = a(i);
		}
	}
	maxValue *= M/dyn;	
	return maxValue;
}

arma::vec RestrictArray(arma::vec *a, int Is, int Ie) {
	arma::vec b(Ie - Is + 1);
	for(int i = Is; i <= Ie; i++) {
		b(i-Is) = (*a)(i);
	}
	return b;
}

arma::mat ConvolveFIR(arma::mat *in, arma::mat *ker) {
	int N = (*in).n_cols;
	
	arma::mat out(N,N);
	out.zeros();

	for(int x=0; x<N; x++) {
		for(int y=0; y<N; y++) {
			int xp = ((x+1) == N) ? 1 : (x+1);
			int xm = ((x-1) < 0) ? (N-2) : (x-1);
			int yp = ((y+1) == N) ? 1 : (y+1);
			int ym = ((y-1) < 0) ? (N-2) : (y-1);			
			
			arma::mat neigh(3,3);
			neigh 	<< (*in)(xm,ym) << (*in)(x,ym) << (*in)(xp,ym) << arma::endr
					<< (*in)(xm,y) << (*in)(x,y) << (*in)(xp,y) << arma::endr
					<< (*in)(xm,yp) << (*in)(x,yp) << (*in)(xp,yp) << arma::endr;
					
			out(y,x) = arma::accu(neigh % (-1* (*ker)));		
		}	
	}

	return out;
}

void EstimateNoiseParams(arma::cube *input, arma::cube *quadtree, double *alpha, double *mu, double *sigma, int size) {
	
	double dSi = 0.;
		
	int Nx = (*input).n_cols;
	int Ny = (*input).n_rows;
	int T = (*input).n_slices;
	
	int wtype = 0; // 0 - "Huber", 1 - "BiSquare"

	// Reshape Laplacian kernel to 3x3
	laplacian.reshape(3,3);

	// Perform quadtree decomposition of frames
	// to generate patches for noise estimation
	int maxVsize = (Nx/size)*(Nx/size);
	
	arma::vec means = arma::ones<arma::vec>(T*maxVsize);
	arma::vec vars = arma::ones<arma::vec>(T*maxVsize);	
	means *= -1;
	vars *= -1;

	arma::vec minslice(T);
	
	// Perform quadtree decomposition of frames
	// to generate patches for noise estimation
#pragma omp parallel for
	for(int i=0; i<T; i++) {	
		std::vector<arma::umat> treeDelete;
		treeDelete.resize(2);
		treeDelete[0] = arma::zeros<arma::umat>(3,1);
		treeDelete[0](2,0) = Nx;
		treeDelete[1] = arma::zeros<arma::umat>(0,0);
		
		treeDelete = QuadTree(&((*input).slice(i)), size, treeDelete, 0);
		arma::umat tree = treeDelete[0];
		arma::umat dele = arma::unique(arma::sort(treeDelete[1]));
	
		// Shed parents from Quadtree
		for(int k=0; k<(int)dele.n_elem; k++) {
			tree.shed_col(dele(0,k));
			dele -= 1;
		}		
	
		// Extract patches for robust estimation and 
		// also build Quadtree
		arma::mat tempQuadtree(Nx,Ny);
		for(int n=0; n<(int)tree.n_cols; n++) {
			int x = tree(0,n);
			int y = tree(1,n);
			int s = tree(2,n);
		
			// Extract cube from input and reshape to column vector
			arma::cube tmpmm = (*input).subcube(arma::span(x,x+s-1), arma::span(y,y+s-1), arma::span(i,i));
			tmpmm.reshape(s*s,1,1);
			arma::vec col = tmpmm.tube(0,0,arma::size(s*s,1));	
						
			// Add robust patch mean and variance to array	
			// Get robust mean estimate
			double meanEst = RobustMeanEstimate(&col, wtype);
			
			// Convolve with Laplacian operator
			arma::mat patch(s*s,1);
			patch.col(0) = col;
			patch.reshape(s,s);		
			patch = ConvolveFIR(&patch, &laplacian);
			patch.reshape(s*s,1);
			col = patch.col(0);
		
			// Set robust mean estimate
			means(i*maxVsize + n) = meanEst;
		
			// Set robust variance estimate
			vars(i*maxVsize + n) = RobustVarEstimate(&col);
			
			// Create frame with mean-valued segment	
			tempQuadtree.submat(arma::span(x,x+s-1), arma::span(y,y+s-1)) = meanEst * arma::ones<arma::mat>(s,s);
		}
					
		// Rescale and build the quadtree frame for display
		tempQuadtree = (tempQuadtree-tempQuadtree.min())/(tempQuadtree.max()-tempQuadtree.min());
		(*quadtree).slice(i) = arma::log10(1+tempQuadtree);
	}
		
	// Delete empty values (stored as -1)
	means = means.elem( find(means >= 0.) );
	vars = vars.elem( find(vars >= 0.) );
			
	// Get sorted indices of means
	arma::uvec indices = arma::stable_sort_index(means);
	arma::vec rmeans = means.elem(indices);
	arma::vec rvars = vars.elem(indices);
	
	// Output mean vs variance for debugging
	//arma::mat rmeansvars(rmeans.n_elem,2);
	//rmeansvars.col(0) = rmeans;
	//rmeansvars.col(1) = rvars;
	//rmeansvars.save("report/ExampleMeansVsVars.csv", arma::csv_ascii);
		
	// Calculate alpha
	arma::vec alphaBeta = WLSFit(&rmeans, &rvars, wtype);
	
	// Don't override user-defined alpha
	*alpha = (*alpha >= 0.) ? *alpha : alphaBeta(0);
	
	// Calculate mu and sigma
	int method = 4;
	switch(method) {
		case 1:
			{	
				// (Method 1 - PureDenoise@EPFL)
				int L = std::floor(1. * (Nx*Ny/means.n_elem));
				*mu = (*mu >= 0.) ? *mu : ComputeMode(RestrictArray(&rmeans, 0, std::round(0.05*L)));
				dSi = ComputeMode(RestrictArray(&rvars, 0, std::round(0.05*L)));
				*sigma = (*sigma >= 0.) ? *sigma : std::sqrt(dSi);
				break;
			}
		case 2:
			{
				// (Method 2 - PureDenoise@EPFL [commented out there]))
				*mu = (*mu >= 0.) ? *mu : ComputeMode(rmeans);		
				dSi = ComputeMode(rvars);
				*sigma = (*sigma >= 0.) ? *sigma : std::sqrt(std::max(dSi, std::max(alphaBeta(1)+alphaBeta(0)*dSi, 0.)));
				break;
			}
		case 3:
			{
				// (Method 3 - tjof2@cam.ac.uk)
				// This assumes that the DC offset is the mode of the means
				*mu = (*mu >= 0.) ? *mu : ComputeMode(rmeans);	
				*sigma = (*sigma >= 0.) ? *sigma : std::sqrt(std::abs(alphaBeta(1)+alphaBeta(0)*(*mu)));
				break;
			}
		case 4:
		default:
			{
				// (Method 4 - tjof2@cam.ac.uk)
				// This assumes that the DC offset is the min of the means,
				// and thus trys to avoid filtering out actual information 
				// (e.g. an amorphous substrate) during the SVT step
				*mu = (*mu >= 0.) ? *mu : rmeans(0);
				*sigma = (*sigma >= 0.) ? *sigma : std::sqrt(std::abs(alphaBeta(1)+alphaBeta(0)*(*mu)));
				break;
			}
	}

	return;
}

/////////////////////////////
//						   //
//   QUADTREE FUNCTIONS    //
//						   //
/////////////////////////////

// Test to see if a node should be split
bool SplitBlockQ(arma::mat *img, int size) {
				
	// Check if it can be split first by comparing
	// to minimum allowed size of block
	int N = (*img).n_cols;
	if( N <= size ) {
		return false;
	}
	
	// Calculate pseudo-residuals for estimating
	// variance due to noise
	int l = 2*2 + 1;
	arma::mat resids(N,N);
	for(int x=0; x<N; x++) {
		for(int y=0; y<N; y++) {
			int xp = ((x+1) == N) ? 1 : (x+1);
			int xm = ((x-1) < 0) ? (N-2) : (x-1);
			int yp = ((y+1) == N) ? 1 : (y+1);
			int ym = ((y-1) < 0) ? (N-2) : (y-1);
			resids(y,x) = l*(*img)(y,x) - ((*img)(yp,x) + (*img)(ym,x) + (*img)(y,xm) + (*img)(y,xp));
		}
	}
	resids /= std::sqrt(l*l + l);
	
	// Perform F-test based on data variance vs. noise variance
	double stat;
	int R = N*N;
	arma::vec value;
	
	double accuZ = arma::accu(*img) / R;
	double Sz = arma::accu(arma::square(*img - accuZ))/(R-1);
	
	double accuR = arma::accu(resids) / R;
	double Se = arma::accu(arma::square(resids - accuR))/(R-1);

	stat = (Sz > Se) ? Sz/Se : Se/Sz;	
	
	// Look-up value for F-test
	value = Ftest0025.elem( arma::find(DegOFreePlus1 == N) );	

	if(stat > value(0)) {
		return true;
	}
	else {
		return false;
	}		
}

// Recursive quadtree function
std::vector<arma::umat> QuadTree(arma::mat *img, int size, std::vector<arma::umat> treeDelete, int part) {

	int i = treeDelete[0](0, part);
	int j = treeDelete[0](1, part);
	int s = treeDelete[0](2, part);
	
	// Test if block should be split
	arma::mat patch = (*img).submat(arma::span(i, i+s-1), arma::span(j, j+s-1));
	if(!SplitBlockQ(&patch, size)) {
		return treeDelete;
	}		
	
	// If test returns TRUE, split
	s /= 2;
	int n = treeDelete[0].n_cols - 1;
	
	arma::umat newtreeadd(3,4);
	newtreeadd	<< i << i+s << i << i+s << arma::endr
				<< j << j << j+s << j+s << arma::endr
				<< s << s << s << s << arma::endr;				
	arma::umat newtree = arma::join_horiz(treeDelete[0], newtreeadd);
	
	arma::umat newdeleteadd(1,1);
	newdeleteadd << part << arma::endr;	
	arma::umat newdelete = arma::join_horiz(treeDelete[1], newdeleteadd);
	
	std::vector<arma::umat> newtreeDelete;
	newtreeDelete.resize(2);
	newtreeDelete[0] = newtree;
	newtreeDelete[1] = newdelete;	
	
	int iter = n;
	do {
		newtreeDelete = QuadTree(img, size, newtreeDelete, iter);
		iter++;
	} while(iter < n+4);

	return newtreeDelete;
}










