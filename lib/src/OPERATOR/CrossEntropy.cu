#include "OPERATOR/CrossEntropy.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <math.h>

__global__ void crossEntropyKernelClear(float *loss)
{
  *loss = 0.0f;
}

__global__ void crossEntropyKernelForwardBackward16(__half *dProb, float *loss, const __half *prob, const uint8_t *target, int rows, int targetOffset, float invBatchSize)
{
  __shared__ float lossSmem[1024];

  int row = threadIdx.x;
  float rowLoss = 0.0f;

  if(row < rows)
  {
    uint8_t targetClass = target[targetOffset + row];
    int rowOffset = row * 16;

    const uint4 *probRow = (const uint4 *)(prob + rowOffset);
    uint4 *dProbRow = (uint4 *)(dProb + rowOffset);

    uint4 chunk0 = probRow[0];
    uint4 chunk1 = probRow[1];
    __half *p0 = (__half *)&chunk0;
    __half *p1 = (__half *)&chunk1;
    __half outGrad0[8];
    __half outGrad1[8];

    #pragma unroll
    for(int i = 0; i < 8; i++)
    {
      float p = __half2float(p0[i]);
      float t = i == targetClass ? 1.0f : 0.0f;
      outGrad0[i] = __float2half_rn((p - t) * invBatchSize);

      if(t == 1.0f)
        rowLoss = -logf(p + 1e-7f);
    }

    #pragma unroll
    for(int i = 0; i < 8; i++)
    {
      int classIndex = i + 8;
      float p = __half2float(p1[i]);
      float t = classIndex == targetClass ? 1.0f : 0.0f;
      outGrad1[i] = __float2half_rn((p - t) * invBatchSize);

      if(t == 1.0f)
        rowLoss = -logf(p + 1e-7f);
    }

    dProbRow[0] = *(uint4 *)&outGrad0;
    dProbRow[1] = *(uint4 *)&outGrad1;
  }

  lossSmem[row] = rowLoss;
  __syncthreads();

  for(int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
  {
    if(row < stride)
      lossSmem[row] += lossSmem[row + stride];
    __syncthreads();
  }

  if(row == 0)
    *loss = lossSmem[0] * invBatchSize;
}

__global__ void crossEntropyKernelForwardBackwardGeneric(__half *dProb, float *loss, const __half *prob, const uint8_t *target, int rows, int cols, int targetOffset, float invBatchSize)
{
  int row = blockIdx.y;
  int col = blockIdx.x * 8;
  if(row >= rows || col >= cols) return;

  uint8_t targetClass = target[targetOffset + row];
  int rowOffset = row * cols;
  uint4 chunk = *((const uint4 *)(prob + rowOffset + col));
  __half *pHalf = (__half *)&chunk;
  __half outGrad[8];

  #pragma unroll
  for(int i = 0; i < 8; i++)
  {
    int classIndex = col + i;
    float p = __half2float(pHalf[i]);
    float t = classIndex == targetClass ? 1.0f : 0.0f;
    outGrad[i] = __float2half_rn((p - t) * invBatchSize);

    if(t == 1.0f)
      atomicAdd(loss, -logf(p + 1e-7f) * invBatchSize);
  }
  *((uint4 *)(dProb + rowOffset + col)) = *(uint4 *)&outGrad;
}

static int helperNextPow2(int x)
{
  int out = 1;
  while(out < x) out <<= 1;
  return out;
}

OPERATOR::CrossEntropy::CrossEntropy(HANDLER::Workspace& workspace)
{
  this->workspace = &workspace;
}

OPERATOR::CrossEntropy::CrossEntropy(HANDLER::Workspace& workspace, VIEW::Math& dProb, VIEW::Math& loss, VIEW::Math& prob, VIEW::Math& target)
{
  this->workspace = &workspace;
  this->dProb = &dProb;
  this->loss = &loss;
  this->prob = &prob;
  this->target = &target;
}

OPERATOR::CrossEntropy::~CrossEntropy()
{}

VIEW::Math& OPERATOR::CrossEntropy::getProb()
{
  return *this->prob;
}

VIEW::Math& OPERATOR::CrossEntropy::getTarget()
{
  return *this->target;
}

VIEW::Math& OPERATOR::CrossEntropy::getLoss()
{
  return *this->loss;
}

VIEW::Math& OPERATOR::CrossEntropy::getGradProb()
{
  return *this->dProb;
}

void OPERATOR::CrossEntropy::setOperand(VIEW::Math& dProb, VIEW::Math& loss, VIEW::Math& prob, VIEW::Math& target)
{
  this->dProb = &dProb;
  this->loss = &loss;
  this->prob = &prob;
  this->target = &target;
}

void OPERATOR::CrossEntropy::setTargetBatch(int batchSize, int offset)
{
  this->targetBatchSize = batchSize;
  this->targetOffset = offset;
}

void OPERATOR::CrossEntropy::forwardBackward()
{
  VIEW::Shape& layout = this->prob->getLayout();
  int probRows = layout.getRank() > 1 ? layout.getDim(0) : 1;
  int rows = this->targetBatchSize > 0 ? this->targetBatchSize : probRows;
  int cols = layout.getRank() > 1 ? layout.getDim(1) : layout.getDim(0);
  float invBatchSize = 1.0f / (float)rows;

  crossEntropyKernelClear<<<1, 1, 0, this->workspace->getStream()>>>((float *)this->loss->getGpuPtr());

  if(cols == 16 && rows <= 1024)
  {
    int threads = helperNextPow2(rows);
    crossEntropyKernelForwardBackward16<<<1, threads, 0, this->workspace->getStream()>>>(
      (__half *)this->dProb->getGpuPtr(),
      (float *)this->loss->getGpuPtr(),
      (const __half *)this->prob->getGpuPtr(),
      (const uint8_t *)this->target->getGpuPtr(),
      rows,
      this->targetOffset,
      invBatchSize);
    return;
  }

  dim3 blocks(cols / 8, rows);
  crossEntropyKernelForwardBackwardGeneric<<<blocks, 1, 0, this->workspace->getStream()>>>(
    (__half *)this->dProb->getGpuPtr(),
    (float *)this->loss->getGpuPtr(),
    (const __half *)this->prob->getGpuPtr(),
    (const uint8_t *)this->target->getGpuPtr(),
    rows,
    cols,
    this->targetOffset,
    invBatchSize);
}
