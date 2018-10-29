#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

#include "renderdepth_op.h"
#include <stdio.h>

using namespace tensorflow;
using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;
// ---------------------------------------------------------
// DIRECT PORT OF CAFFE CODE WITH MINIMAL CHANGES
// ---------------------------------------------------------

//#define ROUND_OFF 50000
//
//#define WARPS_PER_BLOCK 1
//#define THREADS_PER_WARP 32
//
//const int CAFFE_CUDA_NUM_THREADS = 512;
//
//inline int CAFFE_GET_BLOCKS(const int N) {
//  return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
//}

#define BLOCKDIM_X (32) //32 is equal to a warp in cuda device,
#define BLOCKDIM_Y (16)


__global__ void RenderDepth_kernel_1_initialize(
const float * vertex, const float * tri,
float * depth, float * tri_ind,
int nver, int ntri,int width, int height
)
{
const int b = blockIdx.z; //for the batch dimension
const int i = blockIdx.x * blockDim.x + threadIdx.x;
const int j = blockIdx.y * blockDim.y + threadIdx.y;
const bool withinXbonds = i < width;
const bool withinYbonds = j < height;

//TODO: WENBO: first step is to initialize the depth and tri_inx with -999999 and -1
if(withinXbonds && withinYbonds ){
depth[b * height * width*1 + j * width * 1+ i * 1 + 0 ] = - 99999999999999;
tri_ind[b * height * width*1 + j * width * 1+ i * 1 + 0 ] = -1;
}

return;

}


__global__ void RenderDepth_kernel_2_map(
const float * vertex, const float * tri,
float * depth, float * tri_ind,
double * point1,double * point2, double * point3,double * h,
int nver, int ntri,int width, int height
)
{
const int b = blockIdx.z; //for the batch dimension
const int i = blockIdx.x * blockDim.x + threadIdx.x;
//const int j = blockIdx.y * blockDim.y + threadIdx.y;
const bool withinXbonds = i < ntri;
//const bool withinYbonds = j < height;

//TODO: WENBO: second step is to map the vertexes to x,y coordinates and depth
if(withinXbonds){
int p1 = int(tri[0*ntri + i]);
int p2 = int(tri[1*ntri + i]);
int p3 = int(tri[2*ntri + i]);

point1[b * ntri * 2 + i * 2 + 0 ] = vertex[b * 3 * nver + 0 * nver + p1];
point1[b * ntri * 2 + i * 2 + 1 ] = vertex[b * 3 * nver + 1 * nver  +p1];
point2[b * ntri * 2 + i * 2 + 0 ] = vertex[b * 3 * nver + 0 * nver + p2];
point2[b * ntri * 2 + i * 2 + 1 ] = vertex[b * 3 * nver + 1 * nver + p2];
point3[b * ntri * 2 + i * 2 + 0 ] = vertex[b * 3 * nver + 0 * nver + p3];
point3[b * ntri * 2 + i * 2 + 1 ] = vertex[b * 3 * nver + 1 * nver + p3];
double cent3d_z = (double) ((
    vertex[b* 3* nver + 2 * nver + p1] +
    vertex[b* 3* nver + 2 * nver + p2] +
    vertex[b* 3* nver + 2 * nver + p3]
)/3.0f);
h[b * ntri + i] = cent3d_z;

}

return;

}



__device__ bool GPUPointInTri(double * point, double * pt1, double * pt2, double * pt3)
{
	double pointx = point[0];
	double pointy = point[1];

	double pt1x = pt1[0];
	double pt1y = pt1[1];

	double pt2x = pt2[0];
	double pt2y = pt2[1];

	double pt3x = pt3[0];
	double pt3y = pt3[1];

	double v0x = pt3x - pt1x;
	double v0y = pt3y - pt1y;

	double v1x = pt2x - pt1x;
	double v1y = pt2y - pt1y;

	double v2x = pointx - pt1x;
	double v2y = pointy - pt1y;

	double dot00 = v0x * v0x + v0y * v0y;
	double dot01 = v0x * v1x + v0y * v1y;
	double dot02 = v0x * v2x + v0y * v2y;
	double dot11 = v1x * v1x + v1y * v1y;
	double dot12 = v1x * v2x + v1y * v2y;

	double inverDeno = 0;
	if((dot00 * dot11 - dot01 * dot01) == 0)
		inverDeno = 0;
	else
		inverDeno = 1 / (dot00 * dot11 - dot01 * dot01);

	double u = (dot11 * dot02 - dot01 * dot12) * inverDeno;

	if(u < 0 || u > 1)
		return 0;

	double v = (dot00 * dot12 - dot01 * dot02) * inverDeno;

	if(v < 0 || v > 1)
		return 0;

	return u + v <= 1;
}
__global__ void RenderDepth_kernel_3_generate(
const float * vertex, const float * tri,
float * depth, float * tri_ind,
double * point1,double * point2, double * point3,double * h,
int nver, int ntri,int width, int height
)
{
const int b = blockIdx.z; //for the batch dimension
const int i = blockIdx.x * blockDim.x + threadIdx.x;
//const int j = blockIdx.y * blockDim.y + threadIdx.y;
const bool withinXbonds = i < ntri;
//const bool withinYbonds = j < height;

int x; int y;
//TODO: WENBO: second step is to map the vertexes to x,y coordinates and depth
if(withinXbonds){
double point[2];     double pt1[2];    double pt2[2]; double pt3[2];

pt1[0] = point1[b * ntri * 2 + i * 2 + 0]; pt1[1] = point1[b * ntri * 2 + i * 2 + 1];
pt2[0] = point2[b * ntri * 2 + i * 2 + 0]; pt2[1] = point2[b * ntri * 2 + i * 2 + 1];
pt3[0] = point3[b * ntri * 2 + i * 2 + 0]; pt3[1] = point3[b * ntri * 2 + i * 2 + 1];
int x_min = (int)ceil((double) min(min(pt1[0], pt2[0]), pt3[0]));
int x_max = (int)floor((double)max(max(pt1[0], pt2[0]), pt3[0]));

int y_min = (int)ceil((double) min(min(pt1[1], pt2[1]), pt3[1]));
int y_max = (int)floor((double)max(max(pt1[1], pt2[1]), pt3[1]));

if(x_max < x_min || y_max < y_min || x_max > width-1 || x_min < 0 || y_max > height-1 || y_min < 0)
    return;

   for(x = x_min; x <= x_max; x++) {
        for (y = y_min; y <= y_max; y++) {
            point[0] = x;
            point[1] = y;
            if( depth[b * height * width * 1 +  y * width * 1 + x *1 + 0] < h[b*ntri + i] &&
                GPUPointInTri( point,  pt1,  pt2,  pt3))
            {
                depth  [b * height * width * 1 +  y * width * 1 + x *1 + 0] = h[b*ntri + i];
                tri_ind[b * height * width * 1 +  y * width * 1 + x *1 + 0] = i;//   [x * height + y] = i;
            }
        }
    }
}
return;

}
void RenderDepth(const GPUDevice& d,
                 typename TTypes<float, 3>::ConstTensor vertex,
                 typename TTypes<float, 2>::ConstTensor tri,
                 typename TTypes<float, 4>::Tensor depth,
                 typename TTypes<float, 4>::Tensor tri_ind,
                 RenderDepthState params)
 {
int batch  = params.batch;
int width  = params.width;
int height = params.height;
int nver   = params.nver;
int ntri   = params.ntri;

//int error = -1;
dim3 grid;
dim3 block;
printf("RenderDepth ==> nver: %d, \tntri: %d,\twidth: %d, \theight: %d,\t ",nver,ntri,width,height);

double * point1 ;
double * point2 ;
double * point3 ;
double * h ;

//TODO: Wenbo the point1/2/3 in CPU code are shared by all batch samples.
//TODO: However, in GPU code, since we parallely calculate all batches, the point1/2/3 cannot be shared.
cudaMalloc((void**) & point1, sizeof(double) * batch * 2 * ntri);
cudaMalloc((void**) & point2, sizeof(double) * batch * 2 * ntri);
cudaMalloc((void**) & point3, sizeof(double) * batch * 2 * ntri);
cudaMalloc((void**) & h ,  sizeof(double) * batch * ntri);
cudaError_t err = cudaGetLastError();

block = dim3(BLOCKDIM_X,BLOCKDIM_Y,1);
grid = dim3((width + BLOCKDIM_X - 1)/ BLOCKDIM_X, (height + BLOCKDIM_Y - 1)/BLOCKDIM_Y, batch);

printf("1_initilize ===> block_x: %d,\tblock_y: %d,\tgrid_x: %d,\tgrid_y:%d,\tgrid_z: %d",block.x,block.y,grid.x,grid.y,grid.z);
RenderDepth_kernel_1_initialize <<<grid,block, 0>>>(
vertex.data(), tri.data(),
depth.data(), tri_ind.data(),
nver ,ntri,width,height
);

err = cudaGetLastError();
if (err != cudaSuccess) {
    printf("gpuerror in renderdepth_1_initialize %s\n", cudaGetErrorString(err));
    //THError("aborting");
    return;
}


//map the tri indexes to coordinates and depth
block = dim3(BLOCKDIM_X,1,1);
grid = dim3( (ntri+BLOCKDIM_X-1)/BLOCKDIM_X,1, batch);
printf("2_map ===> block_x: %d,\tblock_y: %d,\tgrid_x: %d,\tgrid_y:%d,\tgrid_z: %d",block.x,block.y,grid.x,grid.y,grid.z);
RenderDepth_kernel_2_map <<<grid,block, 0>>>(
vertex.data(), tri.data(),
depth.data(), tri_ind.data(),
point1,point2,point3, h,
nver ,ntri,width,height
);


err = cudaGetLastError();
if (err != cudaSuccess) {
    printf("gpuerror in renderdepth_2_map %s\n", cudaGetErrorString(err));
    //THError("aborting");
    return;
}

// generate the integer mapped 2d positions
block = dim3(BLOCKDIM_X,1,1);
grid = dim3( (ntri+BLOCKDIM_X-1)/BLOCKDIM_X,1, batch);
printf("3_generate ===> block_x: %d,\tblock_y: %d,\tgrid_x: %d,\tgrid_y:%d,\tgrid_z: %d",block.x,block.y,grid.x,grid.y,grid.z);
RenderDepth_kernel_3_generate <<<grid,block, 0>>>(
vertex.data(), tri.data(),
depth.data(), tri_ind.data(),
point1,point2,point3, h,
nver ,ntri,width,height
);

err = cudaGetLastError();
if (err != cudaSuccess) {
    printf("gpuerror in renderdepth_3_generate %s\n", cudaGetErrorString(err));
    //THError("aborting");
    return;
}

cudaFree(point1);
cudaFree(point2);
cudaFree(point3);
cudaFree(h);

}



__global__ void RenderDepthGrad_kernel(
const float * depth_grad, const float * vertex, const float * tri,
const float * depth,  const float * tri_ind,  float * vertex_grad,
int nver, int ntri,int width, int height
)
{
const int b = blockIdx.z; //for the batch dimension
const int i = blockIdx.x * blockDim.x + threadIdx.x;
const int j = blockIdx.y * blockDim.y + threadIdx.y;
const bool withinXbonds = i < width;
const bool withinYbonds = j < height;

//TODO: WENBO: first step is to initialize the depth and tri_inx with -999999 and -1
if(withinXbonds && withinYbonds ){

float depth_grad_ = depth_grad[b * height * width * 1 + j * width * 1 + i * 1 + 0]; // obtain the pixel's gradients
int tri_ind_ = tri_ind[b* height * width * 1 + j* width * 1 + i * 1 + 0];// obtain the current pixel's triangular index.
int p1 = (int)(tri[0 * ntri + tri_ind_]);// obtain the current pixel's 0-th vertex's index
int p2 = (int)(tri[1 * ntri + tri_ind_]);//obtain the current pixel's 1-th vertex's index
int p3 = (int)(tri[2 * ntri + tri_ind_]);//obtain the current pixel's 2-th vertex's index
//double point1[2]= {(double)(vertex(b,0,p1)), (double)(vertex(b,1,p1))};//obtain the current pixel's 0-th vertex's x,y coordinates
//double point2[2]= {(double)(vertex(b,0,p2)), (double)(vertex(b,1,p2))};//obtain the current pixel's 1-th vertex's x,y coordinates
//double point3[2]= {(double)(vertex(b,0,p3)), (double)(vertex(b,2,p3))};//obtain the current pixel's 2-th vertex's x,y coordinates


// the backpropagated gradients are accumulated on the vertexes's z coordinate,
// there is no gradients on x or y coordinates, because x,y is only used to decide which are occluded.
//TODO: For CUDA parallel computing, there might be multiple values added to a same location in vertex_grad,
//TODO: Therefore, we need to use atomicAdd to avoid such collisions.
atomicAdd( & vertex_grad[b * 3 * nver + 2 * nver + p1] ,  depth_grad_ * 1.0f/3.0f);
atomicAdd( & vertex_grad[b * 3 * nver + 2 * nver + p2] ,  depth_grad_ * 1.0f/3.0f);
atomicAdd( & vertex_grad[b * 3 * nver + 2 * nver + p3] ,  depth_grad_ * 1.0f/3.0f);
}

return;

}

void RenderDepthGrad(const GPUDevice& d,
                     typename TTypes<float, 4>::ConstTensor depth_grad,
                     typename TTypes<float, 3>::ConstTensor vertex,
                     typename TTypes<float, 2>::ConstTensor tri,
                     typename TTypes<float, 4>::ConstTensor depth,
                     typename TTypes<float, 4>::ConstTensor tri_ind,
                     typename TTypes<float, 3>::Tensor vertex_grad,
                     RenderDepthState params)
 {
 int batch  = params.batch;
int width  = params.width;
int height = params.height;
int nver   = params.nver;
int ntri   = params.ntri;

//int error = -1;
dim3 grid;
dim3 block;
printf("RenderDepthGrad ===> nver: %d, \tntri: %d,\twidth: %d, \theight: %d,\t ",nver,ntri,width,height);


block = dim3(BLOCKDIM_X,BLOCKDIM_Y,1);
grid = dim3((width + BLOCKDIM_X - 1)/ BLOCKDIM_X, (height + BLOCKDIM_Y - 1)/BLOCKDIM_Y, batch);


printf("renderdepthgrad ===> block_x: %d,\tblock_y: %d,\tgrid_x: %d,\tgrid_y:%d,\tgrid_z: %d",block.x,block.y,grid.x,grid.y,grid.z);
RenderDepthGrad_kernel <<<grid,block, 0>>>(
depth_grad.data(), vertex.data(), tri.data(),
depth.data(), tri_ind.data(),
vertex_grad.data(),
nver ,ntri,width,height
);

cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    printf("gpuerror in renderdepthgrad %s\n", cudaGetErrorString(err));
    //THError("aborting");
    return ;
}

}

#endif  // GOOGLE_CUDA
