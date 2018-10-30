#include <mex.h>
#include <matrix.h>
#include <iostream>
#include "3DMM.h"
using namespace std;
void mexFunction(int nlhs,mxArray *plhs[], int nrhs,const mxArray *prhs[])
{
	double* vertex;
	double* tri;
	double* texture;
	double* src_img;
	int nver;
	int ntri;
	int width;
	int height;
	int nChannels;
	double* img;
	double* tri_ind;

	vertex = mxGetPr(prhs[0]);
	tri = mxGetPr(prhs[1]);
	texture = mxGetPr(prhs[2]);
	src_img = mxGetPr(prhs[3]);
	
	nver = *mxGetPr(prhs[4]);
	ntri = *mxGetPr(prhs[5]);

	width = *mxGetPr(prhs[6]);
	height = *mxGetPr(prhs[7]);
	nChannels = *mxGetPr(prhs[8]);
	

	const mwSize dims[3]={height, width, nChannels};
    plhs[0] = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS,mxREAL);
	img = mxGetPr(plhs[0]);

	const mwSize dims1[3]={height, width, 1};
    plhs[1] = mxCreateNumericArray(3, dims1, mxDOUBLE_CLASS,mxREAL);
	tri_ind = mxGetPr(plhs[1]);
    
	MM3D().ZBuffer(vertex, tri, texture, nver, ntri, src_img, width, height, nChannels, img, tri_ind);
}