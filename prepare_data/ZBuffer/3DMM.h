#ifndef MM3D_H
#define MM3D_H

#include <math.h>
#include "3DMMGlobal.h"
#include <cv.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define INF 1E20

class MM3D
{
public:
	// 3DMM and Reference Frame Mapping
	void ZBuffer(double* vertex, double* tri, double* texture, int nver, int ntri, double* src_img, int width, int height, int nChannels, double* img, double* tri_ind);
	bool PointInTri(Mat* point, Mat* pt1, Mat* pt2, Mat* pt3);

	double xmin;
	double xmax;
	double ymin;
	double ymax;
	double zmin;
	double zmax;

	
};



#endif