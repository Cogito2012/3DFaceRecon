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

#include "render_depth_op.h"
#include <stdlib.h>
#include <type_traits> // <== INCLUDE THIS STANDARD HEADER
#include <stdio.h>

//typedef Eigen::GpuDevice GPUDevice;
using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;


using namespace tensorflow;


void get_point_weight(double* weight, double* point, double * pt1, double * pt2, double * pt3)
{
    // vectors
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

    // dot products
    double dot00 = v0x * v0x + v0y * v0y;
    double dot01 = v0x * v1x + v0y * v1y;
    double dot02 = v0x * v2x + v0y * v2y;
    double dot11 = v1x * v1x + v1y * v1y;
    double dot12 = v1x * v2x + v1y * v2y;

    // barycentric coordinates
    double inverDeno = 0;
    if((dot00 * dot11 - dot01 * dot01) == 0)
        inverDeno = 0;
    else
        inverDeno = 1 / (dot00 * dot11 - dot01 * dot01);

    double u = (dot11 * dot02 - dot01 * dot12) * inverDeno;
    double v = (dot00 * dot12 - dot01 * dot02) * inverDeno;

    // weight
    weight[0] = 1 - u - v;
    weight[1] = v;
    weight[2] = u;
}

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

    return u + v < 1;
}
// CPU specialization of actual computation.
//TODO: Wentao two Constant input and two calculated output
const int MAX_NTRI = 10 * 1000 * 1000;
static double point1[2* MAX_NTRI];
static double point2[2* MAX_NTRI];
static double point3[2* MAX_NTRI];
static double h     [1* MAX_NTRI];
static double tritex[3* MAX_NTRI];
static double tri_normal[3* MAX_NTRI];
void RenderDepth(const CPUDevice& d,
                 typename TTypes<float, 3>::ConstTensor vertex,
                 typename TTypes<float, 2>::ConstTensor tri,
                 typename TTypes<float, 3>::ConstTensor texture,
                 typename TTypes<float, 4>::ConstTensor image,
                 typename TTypes<float, 4>::Tensor depth,
                 typename TTypes<float, 4>::Tensor texture_image,
                 typename TTypes<float, 4>::Tensor normal,
                 typename TTypes<float, 4>::Tensor tri_ind,
                 RenderDepthState params)
{
    int batch  = params.batch;
    int width  = params.width;
    int height = params.height;
    int nver   = params.nver;
    int ntri   = params.ntri;
    int texture_ch = params.texture_ch;
    int nChannels = texture_ch;

    //printf("Using CPU to calculate\n");
    //printf("RenderDepth ==> nver: %d, \tntri: %d,\twidth: %d, \theight: %d,\ttexture_ch: %d \n",
    //nver,ntri, width,height,texture_ch);
    int i,j;
    int x,y;
    //printf("still good in entering RenderDepth function\n");
//	double* point1 = (double * ) malloc(2* ntri * sizeof(double));// new double[2 * ntri];
//	double* point2 =  (double * ) malloc(2* ntri * sizeof(double)); //new double[2 * ntri];
//	double* point3 =  (double * ) malloc(2* ntri * sizeof(double)); //new double[2 * ntri];
//	double* h =  (double * ) malloc(ntri * sizeof(double)); //new double[ntri];
    if(ntri >= MAX_NTRI) {
        printf("***************************ERROR*****************\n");
        printf("Too many triangular %d >= %d\n",ntri,MAX_NTRI);
        printf("***************************ERROR*****************\n");
        return ;
    }
    if(nChannels * ntri >= 3* MAX_NTRI) {
        printf("***************************ERROR*****************\n");
        printf("Too many triangular %d >= %d or texture channel is large %d>%d\n",ntri,MAX_NTRI,nChannels,3);
        printf("***************************ERROR*****************\n");
        return ;
    }
//	printf("Are we still good after allocating point1/2/3 and h\n");
//	printf("%p, %p, %p, %p\n",(void*) point1,(void*) point2,(void*) point3,(void*)h);
//	double* imgh = new double[width * height];
//	double* tritex = new double[ntri * nChannels];
//    double* vertex_normal = new double[nver * 3];
    int b;
    //batch loop
    for(b=0; b< batch; b ++) {

        for(j = 0; j < height; j ++)
            for(i = 0; i < width; i++)
            {
//		imgh[i] = -99999999999999;
                depth(b,j, i, 0) = - 99999999999999;
//		tri_ind[i] = -1;
                normal(b,j,i,0)= 0;
                normal(b,j,i,1) = 0 ;
                normal(b,j,i,2) = 0;
                tri_ind(b,j,i, 0) = -1;
            }
//        // initialize vertices normal
//        for(i = 0; i < nver; i++)
//        {
//            vertex_normal[3*i + 0] = 0;
//            vertex_normal[3*i + 1] = 0;
//            vertex_normal[3*i + 2] = 0;
//        }

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

            //TODO: I think that the simplest average of three vertexes are not accurate enougph, that the network cannot be well trained.
            //TODO: It's better to use a interpolation algorithm as in the papers.
            double cent3d_z = (double)((vertex(b,2,p1) + vertex(b,2,p2) + vertex(b,2,p3))/3.0f);  // (vertex[3*p1+2] + vertex[3*p2+2] + vertex[3*p3+2]) / 3;

            h[i] = cent3d_z;

            for(j = 0; j < nChannels; j++)
            {
                tritex[nChannels * i + j ] = (texture(b,j,p1) + texture(b,j,p2) + texture(b,j,p3))/3.0f;
            }

            // compute the triangle normals: n = (p1 - p2) x (p1 - p3)
            double   p1_2_x = vertex(b,0,p1) - vertex(b,0,p2);
            double   p1_2_y = vertex(b,1,p1) - vertex(b,1,p2);
            double   p1_2_z = vertex(b,2,p1) - vertex(b,2,p2);
            double   p1_3_x = vertex(b,0,p1) - vertex(b,0,p3);
            double   p1_3_y = vertex(b,1,p1) - vertex(b,1,p3);
            double   p1_3_z = vertex(b,2,p1) - vertex(b,2,p3);

            tri_normal[3*i + 0 ] = p1_2_y * p1_3_z - p1_2_z * p1_3_y;
            tri_normal[3*i + 1 ] = p1_2_z * p1_3_x - p1_2_x * p1_3_z;
            tri_normal[3*i + 2 ] = p1_2_x * p1_3_y - p1_2_y * p1_3_x;

//            // get vertices normals
//            for(j = 0; j < 3; j++)
//            {
//                vertex_normal[3*p1 + j] = vertex_normal[3*p1 + j] + tri_normal[3* i + j];
//                vertex_normal[3*p2 + j] = vertex_normal[3*p2 + j] + tri_normal[3* i + j];
//                vertex_normal[3*p3 + j] = vertex_normal[3*p3 + j] + tri_normal[3* i + j];
//            }

        }

        double point[2];
        double pt1[2];
        double pt2[2];
        double pt3[2];
//        double weight[3];

        //init image
        for(j = 0; j < height; j ++)
            for(i = 0; i < width; i++)
            {
                texture_image(b,j,i,0) = 0;
                texture_image(b,j,i,1) = 0;
                texture_image(b,j,i,2) = 0;
            }

        for(i = 0; i < ntri; i++)
        {
            int p1 = int(tri(0,i)); //int(tri[3*i]);
            int p2 = int(tri(1,i)); //int(tri[3*i+1]);
            int p3 = int(tri(2,i)); //int(tri[3*i+2]);

            pt1[0] = point1[2*i];
            pt1[1] = point1[2*i+1];
            pt2[0] = point2[2*i];
            pt2[1] = point2[2*i+1]; //((double*)pt2.data)[0] = point2[2*i]; ((double*)pt2.data)[1] = point2[2*i+1];
            pt3[0] = point3[2*i];
            pt3[1] = point3[2*i+1]; //((double*)pt3.data)[0] = point3[2*i]; ((double*)pt3.data)[1] = point3[2*i+1];

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
//                    // get weights
//                    get_point_weight(weight, point,  pt1,  pt2,  pt3);
//                    double p_depth = weight[0] * vertex(b, 2, p1) + weight[1] * vertex(b, 2, p1) + weight[2] * vertex(b, 2, p1);
                    double p_depth = h[i];
                    if( depth(b, y,x,0) < p_depth && PointInTri( point,  pt1,  pt2,  pt3))
                    {
                        depth(b,y, x, 0) = p_depth ; //imgh[x * height + y] = h[i];
                        for(j = 0; j < nChannels; j++)
                        {
//                            texture_image(b, y, x, j) = weight[0] * texture(b, j, p1) + weight[1] *  texture(b, j, p2) + texture(b, j, p3);
                            texture_image(b, y, x, j) = tritex[nChannels * i + j];
                            //j * width * height + x * height + y] =  tritex[nChannels * i + j];
                        }
                        // get vertices normal
                        for(j = 0; j < 3; j++)
                        {
//                            normal(b, y, x, j) = (vertex_normal[3* p1 + j] + vertex_normal[3* p2 + j] +  vertex_normal[3* p3 + j])/3.0f;
                              normal(b, y, x, j) = tri_normal[3* i + j];
                        }
                        tri_ind(b,y,x,0) = i;//   [x * height + y] = i;
                    }
                }
            }


        }



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
    for(b=0; b < batch; b++) {
        for(j = 0; j < height; j ++) {
            for(i = 0; i < width; i ++) {
//TODO: Wentao focus on the current depth pixel

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
        : OpKernel(context) {}
//, attrs(context)

    void Compute(OpKernelContext* context) override {
        //printf("I am entering RenderDepthOp Compute\n");
        const Tensor& vertex = context->input(0);
        const Tensor& tri = context->input(1);
        const Tensor& texture = context->input(2);
        const Tensor& image = context->input(3);

//    OP_REQUIRES(context, input_0.shape() == input_1.shape(),
//                errors::InvalidArgument("Input shapes have to be the same"));


        //TODO: vertex shape is 3D [Batch, 3, nver]
        //TODO: tri shape is 2D [3, ntri]
        typename TTypes<float, 3>::ConstTensor vertex_data = vertex.tensor<float, 3>();
        typename TTypes<float, 2>::ConstTensor tri_data = tri.tensor<float, 2>();
        typename TTypes<float, 3>::ConstTensor texture_data = texture.tensor<float,3>();
        typename TTypes<float, 4>::ConstTensor image_data = image.tensor<float, 4>();


        const int batch = image_data.dimension(0);
        const int in_channels = image_data.dimension(3); //channel_last
        ///TODO: Wentao I need to print some values like batch, channels, height, width to verify correctness...
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

        const int texture_ch = texture_data.dimension(1);
        //printf("RenderDepth ==> nver: %d, \tntri: %d,\twidth: %d, \theight: %d,\ttexture_channel: %d \n ",
        //            nver,ntri, in_width,in_height,texture_ch);
        OP_REQUIRES(context, tri_space_dim == 3, errors::InvalidArgument("The tri is not 3 x ntri"));
        OP_REQUIRES(context, texture_ch == 3, errors::InvalidArgument("The texture channel must be equal to image channel namely 3"));

        struct RenderDepthState st;
        st.batch = batch ;
        st.height = in_height;
        st.width = in_width ;
        st.nver = nver;
        st.ntri = ntri;
        st.texture_ch = texture_ch;

//    OP_REQUIRES(context, st.out_width * st.out_height > 0,
//                errors::InvalidArgument("Invalid renderdepth settings"));

        //TODO: Wendo allocate memory space for both depth and tri_ind
        Tensor* depth = NULL;
        Tensor* texture_image = NULL;
        Tensor* normal = NULL;
        Tensor* tri_ind = NULL;

        TensorShape depth_shape({batch, in_height,in_width, 1});
        TensorShape texture_image_shape({batch, in_height, in_width, texture_ch});
        TensorShape normal_shape({batch, in_height, in_width, 3});
        TensorShape tri_ind_shape({batch, in_height,in_width, 1});

        OP_REQUIRES_OK(context, context->allocate_output(0, depth_shape,   &depth)); //TODO: Wentao the first output
        OP_REQUIRES_OK(context, context->allocate_output(1, texture_image_shape, &texture_image));
        OP_REQUIRES_OK(context, context->allocate_output(2, normal_shape, &normal));
        OP_REQUIRES_OK(context, context->allocate_output(3, tri_ind_shape, &tri_ind));//TODO: Wentao the second output
        // printf("Ok after creating output\n");
        //TODO: Wentao obtain the data tensor
        typename TTypes<float, 4>::Tensor depth_data = depth->tensor<float, 4>();
        typename TTypes<float, 4>::Tensor tri_ind_data = tri_ind->tensor<float, 4>();
        typename TTypes<float, 4>:: Tensor texture_image_data = texture_image->tensor<float,4>();
        typename TTypes<float, 4>:: Tensor normal_data = normal->tensor<float,4>();

        //call the function on CPU or GPU
        RenderDepth(context->eigen_device<Device>(),
                    vertex_data , tri_data,texture_data, image_data,
                    depth_data,texture_image_data, normal_data, tri_ind_data,
                    st);
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
//    printf("I'm entering RenderDepthOpGrad Compute on \n");
//    printf()
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
        ///TODO: Wentao I need to print some values like batch, channels, height, width to verify correctness...
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


        struct RenderDepthState st;
        st.batch = batch ;
        st.height = in_height;
        st.width = in_width ;
        st.nver = nver;
        st.ntri = ntri;


        Tensor* vertex_grad = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, vertex.shape(),
                       &vertex_grad));//TODO: Wentao the first output
//    Tensor* tri_grad = NULL;
//    OP_REQUIRES_OK(context, context->allocate_output(1, tri.shape(),
//                                                     &tri_grad));//TODO: Wentao the second output

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
.Input("texture: float")
.Input("image: float")
//  .Attr("nver: int = 10000")
//  .Attr("ntri: int = 20000")
//  .Attr("width: int = 100")
//  .Attr("height: int = 50")
.Output("depth: float")
.Output("texture_image: float")
.Output("normal: float")
.Output("tri_ind: float")
.SetShapeFn([](shape_inference::InferenceContext* c) {
//    RenderDepthAttrs attrs;
//    c->GetAttr("nver", &attrs.nver);
//    c->GetAttr("ntri", &attrs.ntri);
//    c->GetAttr("width", &attrs.width);
//    c->GetAttr("height", &attrs.height);
//    printf("I am here in set shape Fn\n");
    DimensionHandle batch = c->Dim(c->input(0), 0);
    DimensionHandle h = c->Dim(c->input(3), 1); //TODO: Channel_last
    DimensionHandle w = c->Dim(c->input(3), 2);
    DimensionHandle ch = c->Dim(c->input(2), 1);

//out_height = ceil((float)(padded_height - border_size *2) / (float)stride_1);
//    int out_channels = neighborhood_grid_width * neighborhood_grid_width;

    //TODO: Wentao set depth image to be 1 channel as well as the tri_ind Tensor
    c->set_output(0, c->MakeShape({batch, h, w, 1 }));
    c->set_output(1, c->MakeShape({batch, h, w, ch}));
    c->set_output(2, c->MakeShape({batch, h, w, 3 }));
    c->set_output(3, c->MakeShape({batch, h, w, 1 }));
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

//TODO: Wentao added CPU Support
REGISTER_KERNEL_BUILDER(Name("RenderDepth").Device(DEVICE_CPU), RenderDepthOp<CPUDevice>);
REGISTER_KERNEL_BUILDER(Name("RenderDepthGrad").Device(DEVICE_CPU), RenderDepthOpGrad<CPUDevice>);


#if GOOGLE_CUDA

//TODO: Wentao GPU device will use another RenderDepth() and RenderDepthGrad()
/* Declare explicit instantiations in kernel_example.cu.cc. */
//  extern RenderDepth();
//  extern RenderDepthGrad();

REGISTER_KERNEL_BUILDER(Name("RenderDepth").Device(DEVICE_GPU), RenderDepthOp<GPUDevice>);
REGISTER_KERNEL_BUILDER(Name("RenderDepthGrad").Device(DEVICE_GPU), RenderDepthOpGrad<GPUDevice>);

#endif // GOOGLE_CUDA
