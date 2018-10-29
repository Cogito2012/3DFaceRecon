#define EIGEN_USE_THREADS

#include <memory>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"

#include "renderdepth_op.h"
#include <type_traits> // <== INCLUDE THIS STANDARD HEADER
#include <stdio.h>

//typedef Eigen::GpuDevice GPUDevice;
using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;


using namespace tensorflow;



bool PointInTri(double * point, double * pt1, double * pt2, double * pt3)
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
// CPU specialization of actual computation.
//TODO: Wenbo two Constant input and two calculated output
void RenderDepth(const CPUDevice& d,
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

	int i,j;
	int x,y;

	double* point1 = new double[2 * ntri];
	double* point2 = new double[2 * ntri];
	double* point3 = new double[2 * ntri];
	double* h = new double[ntri];
//	double* imgh = new double[width * height];
//	double* tritex = new double[ntri * nChannels];
int b;
//batch loop
for(b=0; b< batch;b ++){

for(j = 0; j < height; j ++)
for(i = 0; i < width; i++)
{
//		imgh[i] = -99999999999999;
    depth(b,j, i, 0) = - 99999999999999;
//		tri_ind[i] = -1;
    tri_ind(b,j,i, 0) = -1;
}

for(i = 0; i < ntri; i++)
{
    // 3 point index of triangle
    int p1 = int(tri(0,i)); //int(tri[3*i]);
    int p2 = int(tri(1,i)); //int(tri[3*i+1]);
    int p3 = int(tri(2,i)); //int(tri[3*i+2]);

    point1[2*i]   = vertex(b,0,p1); //[3*p1];
    point1[2*i+1] = vertex(b,1,p1);// vertex[3*p1+1];
    point2[2*i]   = vertex(b,0,p2); //[3*p2];
    point2[2*i+1] = vertex(b,1,p2); //[3*p2+1];
    point3[2*i]   = vertex(b,0,p3); //[3*p3];
    point3[2*i+1] = vertex(b,1,p3); //[3*p3+1];

    //TODO: Wenbo I think that the simplest average of three vertexes are not accurate enougph, that the network cannot be well trained.
    //TODO: It's better to use a interpolation algorithm as in the papers.
    double cent3d_z = (double)((vertex(b,2,p1) + vertex(b,2,p2) + vertex(b,2,p3))/3.0f);  // (vertex[3*p1+2] + vertex[3*p2+2] + vertex[3*p3+2]) / 3;

    h[i] = cent3d_z;

//		for(j = 0; j < nChannels; j++)
//		{
//			tritex[nChannels*i+j] = (texture[nChannels*p1+j] + texture[nChannels*p2+j] + texture[nChannels*p3+j]) / 3;
//		}
}

//	Mat point(2, 1, CV_64F);
//	Mat pt1(2, 1, CV_64F);
//	Mat pt2(2, 1, CV_64F);
//	Mat pt3(2, 1, CV_64F);
double point[2];     double pt1[2];    double pt2[2]; double pt3[2];

//init image
//	for(i = 0; i < width * height * nChannels; i++)
//	{
//		img[i] = src_img[i];
//	}

for(i = 0; i < ntri; i++)
{
    pt1[0] = point1[2*i]; pt1[1] = point1[2*i+1];
    pt2[0] = point2[2*i]; pt2[1] = point2[2*i+1]; //((double*)pt2.data)[0] = point2[2*i]; ((double*)pt2.data)[1] = point2[2*i+1];
    pt3[0] = point3[2*i]; pt3[1] = point3[2*i+1]; //((double*)pt3.data)[0] = point3[2*i]; ((double*)pt3.data)[1] = point3[2*i+1];

    int x_min = (int)ceil((double) min(min(pt1[0], pt2[0]), pt3[0]));
    int x_max = (int)floor((double)max(max(pt1[0], pt2[0]), pt3[0]));

    int y_min = (int)ceil((double) min(min(pt1[1], pt2[1]), pt3[1]));
    int y_max = (int)floor((double)max(max(pt1[1], pt2[1]), pt3[1]));

    if(x_max < x_min || y_max < y_min || x_max > width-1 || x_min < 0 || y_max > height-1 || y_min < 0)
        continue;

    for(x = x_min; x <= x_max; x++)
    {
        for (y = y_min; y <= y_max; y++)
        {
            point[0] = x;
            point[1] = y;
            if( depth(b, y,x,0) < h[i] && PointInTri( point,  pt1,  pt2,  pt3))
            {
                depth(b,y, x, 0) = h[i] ; //imgh[x * height + y] = h[i];
//					for(j = 0; j < nChannels; j++)
//					{
//						img[j * width * height + x * height + y] =  tritex[nChannels * i + j];
//					}
                tri_ind(b,y,x,0) = i;//   [x * height + y] = i;
            }
        }
    }
}


delete[] point1;
delete[] point2;
delete[] point3;
delete[] h;
//	delete[] imgh;
//	delete[] tritex;
}

}

//TODO: 5 Constant input and 1 calculated output
void RenderDepthGrad(const CPUDevice& d,
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

int i,j;
int x,y;

int b;
for(b=0; b < batch; b++){
for(j = 0; j < height; j ++){
for(i = 0; i < width; i ++){
//TODO: Wenbo focus on the current depth pixel

float depth_grad_ = depth_grad(b,j,i,0); // obtain the pixel's gradients
int tri_ind_ = tri_ind(b,j,i,0) ;// obtain the current pixel's triangular index.
int p1 = (int)(tri(0,tri_ind_));// obtain the current pixel's 0-th vertex's index
int p2 = (int)(tri(1,tri_ind_));//obtain the current pixel's 1-th vertex's index
int p3 = (int)(tri(2,tri_ind_));//obtain the current pixel's 2-th vertex's index
//double point1[2]= {(double)(vertex(b,0,p1)), (double)(vertex(b,1,p1))};//obtain the current pixel's 0-th vertex's x,y coordinates
//double point2[2]= {(double)(vertex(b,0,p2)), (double)(vertex(b,1,p2))};//obtain the current pixel's 1-th vertex's x,y coordinates
//double point3[2]= {(double)(vertex(b,0,p3)), (double)(vertex(b,2,p3))};//obtain the current pixel's 2-th vertex's x,y coordinates


// the backpropagated gradients are accumulated on the vertexes's z coordinate,
// there is no gradients on x or y coordinates, because x,y is only used to decide which are occluded.
vertex_grad(b,2,p1) += depth_grad_ * 1.0f/3.0f;
vertex_grad(b,2,p2) += depth_grad_ * 1.0f/3.0f;
vertex_grad(b,2,p3) += depth_grad_ * 1.0f/3.0f;
}
}
}

}


template <typename Device>
class RenderDepthOp : public OpKernel {
public:
  explicit RenderDepthOp(OpKernelConstruction* context)
  : OpKernel(context){}
//, attrs(context)

  void Compute(OpKernelContext* context) override {
    const Tensor& vertex = context->input(0);
    const Tensor& tri = context->input(1);
    const Tensor& image = context->input(2);

//    OP_REQUIRES(context, input_0.shape() == input_1.shape(),
//                errors::InvalidArgument("Input shapes have to be the same"));


    //TODO: vertex shape is 3D [Batch, 3, nver]
    //TODO: tri shape is 2D [3, ntri]
    typename TTypes<float, 3>::ConstTensor vertex_data = vertex.tensor<float, 3>();
    typename TTypes<float, 2>::ConstTensor tri_data = tri.tensor<float, 2>();
    typename TTypes<float, 4>::ConstTensor image_data = image.tensor<float, 4>();


    const int batch = image_data.dimension(0);
    const int in_channels = image_data.dimension(3); //channel_last
    ///TODO: Wenbo I need to print some values like batch, channels, height, width to verify correctness...
    ///TODO: But how to print ?
//    printf("")
    const int in_height = image_data.dimension(1);
    const int in_width = image_data.dimension(2);



    const int vertex_batch  = vertex_data.dimension(0);
    const int vertex_space_dim = vertex_data.dimension(1);//MUST equal to 3
    const int nver = vertex_data.dimension(2);
    OP_REQUIRES(context, vertex_batch==batch, errors::InvalidArgument("The vertex's batch is not the same as image batch"));
    OP_REQUIRES(context, vertex_space_dim==3, errors::InvalidArgument("The vertex is not Batch x 3 x nver"));

    const int tri_space_dim = tri_data.dimension(0); //MUST equal to 3
    const int ntri = tri_data.dimension(1);
    printf("RenderDepth ==> nver: %d, \tntri: %d,\twidth: %d, \theight: %d,\t ",nver,ntri, in_width,in_height);
    OP_REQUIRES(context, tri_space_dim == 3, errors::InvalidArgument("The tri is not 3 x ntri"));


    RenderDepthState st(batch, in_height, in_width, nver, ntri);

//    OP_REQUIRES(context, st.out_width * st.out_height > 0,
//                errors::InvalidArgument("Invalid renderdepth settings"));

    //TODO: Wendo allocate memory space for both depth and tri_ind
    Tensor* depth = NULL;
    Tensor* tri_ind = NULL;
    TensorShape depth_shape({batch, in_height,in_width, 1});
    TensorShape tri_ind_shape({batch, in_height,in_width, 1});
    OP_REQUIRES_OK(context, context->allocate_output(0, depth_shape,   &depth)); //TODO: Wenbo the first output
    OP_REQUIRES_OK(context, context->allocate_output(1, tri_ind_shape, &tri_ind));//TODO: Wenbo the second output

    //TODO: Wenbo obtain the data tensor
    typename TTypes<float, 4>::Tensor depth_data = depth->tensor<float, 4>();
    typename TTypes<float, 4>::Tensor tri_ind_data = tri_ind->tensor<float, 4>();

    //call the function on CPU or GPU
//    if(std::is_same<Device,CPUDevice>::value){
        RenderDepth(context->eigen_device<Device>(),
          vertex_data , tri_data,
          depth_data, tri_ind_data,
          st);
//    }else{
//
//    }

  }

//private:
//  RenderDepthAttrs attrs;
};

template <typename Device>
class RenderDepthOpGrad : public OpKernel {
public:
  explicit RenderDepthOpGrad(OpKernelConstruction* context)
  : OpKernel(context) {}
//, attrs(context)
  void Compute(OpKernelContext* context) override {
    const Tensor& depth_grad = context->input(0);
    const Tensor& vertex = context->input(1);
    const Tensor& tri = context->input(2);
    const Tensor& depth = context->input(3);
    const Tensor& tri_ind = context->input(4);
    const Tensor& image = context->input(5);

    typename TTypes<float, 4>::ConstTensor depth_grad_data = depth_grad.tensor<float, 4>();
    typename TTypes<float, 3>::ConstTensor vertex_data = vertex.tensor<float, 3>();
    typename TTypes<float, 2>::ConstTensor tri_data = tri.tensor<float, 2>();
    typename TTypes<float, 4>::ConstTensor depth_data = depth.tensor<float, 4>();
    typename TTypes<float, 4>::ConstTensor tri_ind_data= tri_ind.tensor<float, 4>();
    typename TTypes<float, 4>::ConstTensor image_data = image.tensor<float,4>();

    const int batch = image_data.dimension(0);
    const int in_channels = image_data.dimension(3); //channel_last
    ///TODO: Wenbo I need to print some values like batch, channels, height, width to verify correctness...
    ///TODO: But how to print ?
//    printf("")
    const int in_height = image_data.dimension(1);
    const int in_width = image_data.dimension(2);

    const int vertex_batch  = vertex_data.dimension(0);
    const int vertex_space_dim = vertex_data.dimension(1);//MUST equal to 3
    const int nver = vertex_data.dimension(2);
    OP_REQUIRES(context, vertex_batch==batch, errors::InvalidArgument("The vertex's batch is not the same as image batch"));
    OP_REQUIRES(context, vertex_space_dim==3, errors::InvalidArgument("The vertex is not Batch x 3 x nver"));

    const int tri_space_dim = tri_data.dimension(0); //MUST equal to 3
    const int ntri = tri_data.dimension(1);
    OP_REQUIRES(context, tri_space_dim == 3, errors::InvalidArgument("The tri is not 3 x ntri"));


    RenderDepthState st(batch, in_height, in_width, nver, ntri);

    Tensor* vertex_grad = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, vertex.shape(),
                                                     &vertex_grad));//TODO: Wenbo the first output
//    Tensor* tri_grad = NULL;
//    OP_REQUIRES_OK(context, context->allocate_output(1, tri.shape(),
//                                                     &tri_grad));//TODO: Wenbo the second output

    typename TTypes<float, 3>::Tensor vertex_grad_data = vertex_grad->tensor<float, 3>();
//    typename TTypes<float, 2>::Tensor output_grad_1_data = tri_grad->tensor<float, 2>();

    RenderDepthGrad(context->eigen_device<Device>(),
                    depth_grad_data, vertex_data,tri_data, depth_data,tri_ind_data,
                    vertex_grad_data,
                    st);
  }
//private:
//  RenderDepthAttrs attrs;
};

using shape_inference::DimensionHandle;

REGISTER_OP("RenderDepth")
  .Input("vertex: float")
  .Input("tri: float")
  .Input("image: float")
//  .Attr("nver: int = 10000")
//  .Attr("ntri: int = 20000")
//  .Attr("width: int = 100")
//  .Attr("height: int = 50")
  .Output("depth: float")
  .Output("tri_ind: float")
  .SetShapeFn([](shape_inference::InferenceContext* c) {
//    RenderDepthAttrs attrs;
//    c->GetAttr("nver", &attrs.nver);
//    c->GetAttr("ntri", &attrs.ntri);
//    c->GetAttr("width", &attrs.width);
//    c->GetAttr("height", &attrs.height);

    DimensionHandle batch = c->Dim(c->input(0), 0);
    DimensionHandle h = c->Dim(c->input(2), 1); //TODO: Channel_last
    DimensionHandle w = c->Dim(c->input(2), 2);


//out_height = ceil((float)(padded_height - border_size *2) / (float)stride_1);
//    int out_channels = neighborhood_grid_width * neighborhood_grid_width;

    //TODO: Wenbo set depth image to be 1 channel as well as the tri_ind Tensor
    c->set_output(0, c->MakeShape({batch, h, w, 1 }));
    c->set_output(1, c->MakeShape({batch, h, w, 1 }));
    return Status::OK();
  });

REGISTER_OP("RenderDepthGrad")
  .Input("depth_grad: float")
  .Input("vertex: float")
  .Input("tri: float")
  .Input("depth: float")
  .Input("tri_ind: float")
  .Input("image: float")
//  .Attr("nver: int = 10000")
//  .Attr("ntri: int = 20000")
//  .Attr("width: int = 100")
//  .Attr("height: int = 50")
  .Output("vertex_grad: float")
//  .Output("tri_grad: float")
  .SetShapeFn([](shape_inference::InferenceContext* c) {
    c->set_output(0, c->input(1)); //vertex's gradients
//    c->set_output(1, c->input(2)); //tri's gradients which is of no use.
    return Status::OK();
  }
  );

//TODO: Wenbo added CPU Support
REGISTER_KERNEL_BUILDER(Name("RenderDepth").Device(DEVICE_CPU), RenderDepthOp<CPUDevice>);
REGISTER_KERNEL_BUILDER(Name("RenderDepthGrad").Device(DEVICE_CPU), RenderDepthOpGrad<CPUDevice>);


#if GOOGLE_CUDA

   //TODO: Wenbo GPU device will use another RenderDepth() and RenderDepthGrad()
  /* Declare explicit instantiations in kernel_example.cu.cc. */
//  extern RenderDepth();
//  extern RenderDepthGrad();

REGISTER_KERNEL_BUILDER(Name("RenderDepth").Device(DEVICE_GPU), RenderDepthOp<GPUDevice>);
REGISTER_KERNEL_BUILDER(Name("RenderDepthGrad").Device(DEVICE_GPU), RenderDepthOpGrad<GPUDevice>);

#endif // GOOGLE_CUDA
