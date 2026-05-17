#include "OPERATOR/SGD.hpp"

#include <cuda_fp16.h>
#include <cuda/cmath>
#include <cuda_runtime_api.h>

static __device__ __forceinline__ unsigned int helperKernelHalf2ToU32(__half2 x)
{
  union
  {
    unsigned int u;
    __half2 h;
  } v;
  v.h = x;
  return v.u;
}

static __device__ __forceinline__ __half2 helperKernelU32ToHalf2(unsigned int x)
{
  union
  {
    unsigned int u;
    __half2 h;
  } v;
  v.u = x;
  return v.h;
}

static __device__ __forceinline__ half2 helperKernelHalf2SgdUpdate(half2 weight, half2 grad, float lr)
{
  half2 negLr = __float2half2_rn(-lr);
  return __hfma2(grad, negLr, weight);
}

__global__ void sgdKernelUpdate(uint4 * __restrict__ WEIGHT, const uint4 * __restrict__ GRAD, float lr, int N8)
{
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  if(index >= N8) return;

  uint4 weight = WEIGHT[index];
  uint4 grad = GRAD[index];
  uint4 out;
  out.x = helperKernelHalf2ToU32(helperKernelHalf2SgdUpdate(helperKernelU32ToHalf2(weight.x), helperKernelU32ToHalf2(grad.x), lr));
  out.y = helperKernelHalf2ToU32(helperKernelHalf2SgdUpdate(helperKernelU32ToHalf2(weight.y), helperKernelU32ToHalf2(grad.y), lr));
  out.z = helperKernelHalf2ToU32(helperKernelHalf2SgdUpdate(helperKernelU32ToHalf2(weight.z), helperKernelU32ToHalf2(grad.z), lr));
  out.w = helperKernelHalf2ToU32(helperKernelHalf2SgdUpdate(helperKernelU32ToHalf2(weight.w), helperKernelU32ToHalf2(grad.w), lr));
  WEIGHT[index] = out;
}

OPERATOR::SGD::SGD(HANDLER::Workspace& workspace)
{
  this->workspace = &workspace;
}

OPERATOR::SGD::SGD(HANDLER::Workspace& workspace, VIEW::Math& weight, VIEW::Math& grad)
{
  this->workspace = &workspace;
  this->weight = &weight;
  this->grad = &grad;
}

OPERATOR::SGD::~SGD()
{}

VIEW::Math& OPERATOR::SGD::getWeight()
{
  return *this->weight;
}

VIEW::Math& OPERATOR::SGD::getGrad()
{
  return *this->grad;
}

void OPERATOR::SGD::setOperand(VIEW::Math& weight, VIEW::Math& grad)
{
  this->weight = &weight;
  this->grad = &grad;
}

void OPERATOR::SGD::update(float lr)
{
  int N = this->weight->getCount();
  int N8 = N / 8;
  int threads = N <= 256 ? 256 : N <= 512 ? 512 : 1024;
  int blocks = cuda::ceil_div(N8, threads);
  sgdKernelUpdate<<<blocks, threads, 0, this->workspace->getStream()>>>((uint4 *)this->weight->getGpuPtr(), (const uint4 *)this->grad->getGpuPtr(), lr, N8);
}
