#include "OPERATOR/Normalize.hpp"

#include <cuda_bf16.h>
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

static __device__ __forceinline__ __half2 helperKernelHalf2Norm(half2 x, float scalar)
{
  half2 norm_rcp = __float2half2_rn(1.0f / scalar);
  return __hmul2(x, norm_rcp);
}

__global__ void normKernelByScalar(uint4 * __restrict__ OUT, const uint4 * __restrict__ IN, float scalar)
{
  int index = threadIdx.x + blockDim.x * blockIdx.x;

  float s = scalar;
  uint4 in = IN[index];
  uint4 out;
  out.x = helperKernelHalf2ToU32(helperKernelHalf2Norm(helperKernelU32ToHalf2(in.x), s));
  out.y = helperKernelHalf2ToU32(helperKernelHalf2Norm(helperKernelU32ToHalf2(in.y), s));
  out.z = helperKernelHalf2ToU32(helperKernelHalf2Norm(helperKernelU32ToHalf2(in.z), s));
  out.w = helperKernelHalf2ToU32(helperKernelHalf2Norm(helperKernelU32ToHalf2(in.w), s));
  OUT[index] = out;
}

OPERATOR::Normalize::Normalize(HANDLER::Workspace& workspace)
{
  this->workspace = &workspace;
}
OPERATOR::Normalize::Normalize(HANDLER::Workspace& workspace, VIEW::Math& out, VIEW::Math& in)
{
  this->workspace = &workspace;
  this->out = &out;
  this->in = &in;
}
OPERATOR::Normalize::~Normalize()
{}

VIEW::Math& OPERATOR::Normalize::getInput()
{
  return *this->in;
}
VIEW::Math& OPERATOR::Normalize::getOutput()
{
  return *this->out;
}

void OPERATOR::Normalize::setOperand(VIEW::Math& out, VIEW::Math& in)
{
  this->in = &in;
  this->out = &out;
}

void OPERATOR::Normalize::normByScalar(float scalar)
{
  int N = this->in->getCount();
  int N8 = N / 8;
  int threads = N <= 256 ? 256 : N <= 512 ? 512 : 1024;
  int blocks = cuda::ceil_div(N8, threads);
  normKernelByScalar<<<blocks, threads, 0, this->workspace->getStream()>>>((uint4 *) this->out->getGpuPtr(), (uint4 *) this->in->getGpuPtr(), scalar);
}