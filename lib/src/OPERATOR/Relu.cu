#include "OPERATOR/Relu.hpp"

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

static __device__ __forceinline__ half2 helperKernelHalf2ReluBackward(half2 dOut, half2 in)
{
  __half zero = __float2half_rn(0.0f);

  __half loIn = __low2half(in);
  __half hiIn = __high2half(in);
  __half loOut = __low2half(dOut);
  __half hiOut = __high2half(dOut);

  __half lo = __half2float(loIn) > 0.0f ? loOut : zero;
  __half hi = __half2float(hiIn) > 0.0f ? hiOut : zero;

  return __halves2half2(lo, hi);
}

__global__ void reluKernelForward(uint4 * __restrict__ OUT, const uint4 * __restrict__ IN, int N8)
{
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  if(index >= N8) return;

  __half2 zero = __float2half2_rn(0.0f);

  uint4 in = IN[index];
  uint4 out;
  out.x = helperKernelHalf2ToU32(__hmax2(helperKernelU32ToHalf2(in.x), zero));
  out.y = helperKernelHalf2ToU32(__hmax2(helperKernelU32ToHalf2(in.y), zero));
  out.z = helperKernelHalf2ToU32(__hmax2(helperKernelU32ToHalf2(in.z), zero));
  out.w = helperKernelHalf2ToU32(__hmax2(helperKernelU32ToHalf2(in.w), zero));
  OUT[index] = out;
}

__global__ void reluKernelBackward(uint4 * __restrict__ DIN, const uint4 * __restrict__ DOUT, const uint4 * __restrict__ IN, int N8)
{
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  if(index >= N8) return;

  uint4 dOut = DOUT[index];
  uint4 in = IN[index];
  uint4 dIn;
  dIn.x = helperKernelHalf2ToU32(helperKernelHalf2ReluBackward(helperKernelU32ToHalf2(dOut.x), helperKernelU32ToHalf2(in.x)));
  dIn.y = helperKernelHalf2ToU32(helperKernelHalf2ReluBackward(helperKernelU32ToHalf2(dOut.y), helperKernelU32ToHalf2(in.y)));
  dIn.z = helperKernelHalf2ToU32(helperKernelHalf2ReluBackward(helperKernelU32ToHalf2(dOut.z), helperKernelU32ToHalf2(in.z)));
  dIn.w = helperKernelHalf2ToU32(helperKernelHalf2ReluBackward(helperKernelU32ToHalf2(dOut.w), helperKernelU32ToHalf2(in.w)));
  DIN[index] = dIn;
}

__global__ void reluKernelForwardTail(__half * __restrict__ OUT, const __half * __restrict__ IN, int N)
{
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  if(index >= N) return;

  __half zero = __float2half_rn(0.0f);
  OUT[index] = __hmax(IN[index], zero);
}

__global__ void reluKernelBackwardTail(__half * __restrict__ DIN, const __half * __restrict__ DOUT, const __half * __restrict__ IN, int N)
{
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  if(index >= N) return;

  DIN[index] = __half2float(IN[index]) > 0.0f ? DOUT[index] : __float2half_rn(0.0f);
}

OPERATOR::Relu::Relu(HANDLER::Workspace& workspace)
{
  this->workspace = &workspace;
}

OPERATOR::Relu::Relu(HANDLER::Workspace& workspace, VIEW::Math& out, VIEW::Math& in)
{
  this->workspace = &workspace;
  this->out = &out;
  this->in = &in;
}

OPERATOR::Relu::~Relu()
{}

VIEW::Math& OPERATOR::Relu::getInput()
{
  return *this->in;
}

VIEW::Math& OPERATOR::Relu::getOutput()
{
  return *this->out;
}

VIEW::Math& OPERATOR::Relu::getGradInput()
{
  return *this->dIn;
}

VIEW::Math& OPERATOR::Relu::getGradOutput()
{
  return *this->dOut;
}

void OPERATOR::Relu::setOperand(VIEW::Math& out, VIEW::Math& in)
{
  this->out = &out;
  this->in = &in;
}

void OPERATOR::Relu::setGradOperand(VIEW::Math& dIn, VIEW::Math& dOut)
{
  this->dIn = &dIn;
  this->dOut = &dOut;
}

void OPERATOR::Relu::forward()
{
  int N = this->in->getCount();
  int N8 = N / 8;
  int threads = N <= 256 ? 256 : N <= 512 ? 512 : 1024;

  if(N8 > 0)
  {
    int blocks = cuda::ceil_div(N8, threads);
    reluKernelForward<<<blocks, threads, 0, this->workspace->getStream()>>>((uint4 *)this->out->getGpuPtr(), (const uint4 *)this->in->getGpuPtr(), N8);
  }

  int tailStart = N8 * 8;
  if(tailStart < N)
  {
    int tailN = N - tailStart;
    int blocks = cuda::ceil_div(tailN, threads);
    reluKernelForwardTail<<<blocks, threads, 0, this->workspace->getStream()>>>((__half *)this->out->getGpuPtr() + tailStart, (const __half *)this->in->getGpuPtr() + tailStart, tailN);
  }
}

void OPERATOR::Relu::backward()
{
  int N = this->in->getCount();
  int N8 = N / 8;
  int threads = N <= 256 ? 256 : N <= 512 ? 512 : 1024;

  if(N8 > 0)
  {
    int blocks = cuda::ceil_div(N8, threads);
    reluKernelBackward<<<blocks, threads, 0, this->workspace->getStream()>>>((uint4 *)this->dIn->getGpuPtr(), (const uint4 *)this->dOut->getGpuPtr(), (const uint4 *)this->in->getGpuPtr(), N8);
  }

  int tailStart = N8 * 8;
  if(tailStart < N)
  {
    int tailN = N - tailStart;
    int blocks = cuda::ceil_div(tailN, threads);
    reluKernelBackwardTail<<<blocks, threads, 0, this->workspace->getStream()>>>((__half *)this->dIn->getGpuPtr() + tailStart, (const __half *)this->dOut->getGpuPtr() + tailStart, (const __half *)this->in->getGpuPtr() + tailStart, tailN);
  }
}
