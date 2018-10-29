#ifndef __RENDERDEPTH_H_
#define  __RENDERDEPTH_H_


#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"

using namespace tensorflow;
//typedef Eigen::GpuDevice GPUDevice;
using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

#define min(a,b) ((a)<(b)?(a):(b))
#define max(a,b) ((a)>(b)?(a):(b))



//struct RenderDepthAttrs {
//  RenderDepthAttrs(OpKernelConstruction* c) {
//    OP_REQUIRES_OK(c, c->GetAttr("nver", &nver));
//    OP_REQUIRES_OK(c, c->GetAttr("ntri", &ntri));
////    OP_REQUIRES_OK(c, c->GetAttr("width", &width));
////    OP_REQUIRES_OK(c, c->GetAttr("height", &height));
//
//    OP_REQUIRES(c, kernel_size % 2 != 0,
//                errors::InvalidArgument("kernel_size must be odd"));
//  }
//  RenderDepthAttrs() {}
//
//  int nver;
//  int ntri;
////  int width;
////  int height;
//};
//
struct RenderDepthState {
//  RenderDepthState(int batch, int in_height, int in_width , int nver, int ntri) {
//    TODO: Wenbo this struct is never used...
//    batch = batch;
//    nver = nver;
//    ntri  = ntri;
//    width = in_width;
//    height = in_height;
//
//  }
    int batch;
    int nver;
    int ntri;
    int width;
    int height;
    int texture_ch;
};
//TODO: Wenbo added CPU functor declaration
void RenderDepth(const CPUDevice& d,
                   typename TTypes<float, 3>::ConstTensor vertex,
                 typename TTypes<float, 2>::ConstTensor tri,
                 typename TTypes<float, 3>::ConstTensor texture,
                 typename TTypes<float, 4>::ConstTensor image,
                 typename TTypes<float, 4>::Tensor depth,
                 typename TTypes<float, 4>::Tensor texture_image,
                 typename TTypes<float, 4>::Tensor tri_ind,
                 RenderDepthState params);

void RenderDepthGrad(const CPUDevice& d,
                     typename TTypes<float, 4>::ConstTensor depth_grad,
                     typename TTypes<float, 3>::ConstTensor vertex,
                     typename TTypes<float, 2>::ConstTensor tri,
                     typename TTypes<float, 4>::ConstTensor depth,
                     typename TTypes<float, 4>::ConstTensor tri_ind,
                     typename TTypes<float, 3>::Tensor vertex_grad,
                     RenderDepthState params);

 //TODO: Wenbo added GPU functor declaration (valid when CUDA is avalable)
#if GOOGLE_CUDA

void RenderDepth(const GPUDevice & d,
                  typename TTypes<float, 3>::ConstTensor vertex,
                 typename TTypes<float, 2>::ConstTensor tri,
                 typename TTypes<float, 3>::ConstTensor texture,
                 typename TTypes<float, 4>::ConstTensor image,
                 typename TTypes<float, 4>::Tensor depth,
                 typename TTypes<float, 4>::Tensor texture_image,
                 typename TTypes<float, 4>::Tensor tri_ind,
                 RenderDepthState params);

void RenderDepthGrad(const GPUDevice& d,
                     typename TTypes<float, 4>::ConstTensor depth_grad,
                     typename TTypes<float, 3>::ConstTensor vertex,
                     typename TTypes<float, 2>::ConstTensor tri,
                     typename TTypes<float, 4>::ConstTensor depth,
                     typename TTypes<float, 4>::ConstTensor tri_ind,
                     typename TTypes<float, 3>::Tensor vertex_grad,
                     RenderDepthState params);

#endif // GOOGLE_CUDA







#endif
