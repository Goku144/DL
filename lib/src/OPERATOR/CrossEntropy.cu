#include "OPERATOR/CrossEntropy.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <math.h>

__global__ void crossEntropyKernelClear(float *loss)
{
  *loss = 0.0f;
}

__global__ void crossEntropyKernelForwardBackward(__half *dProb, float *loss, const __half *prob, const uint8_t *target, int rows, int cols, float invBatchSize)
{
  int row = blockIdx.y;
  int col = blockIdx.x * 8;
  if(row >= rows || col >= cols) return;

  uint8_t targetClass = target[row];
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

void OPERATOR::CrossEntropy::forwardBackward()
{
  VIEW::Shape& layout = this->prob->getLayout();
  int rows = layout.getRank() > 1 ? layout.getDim(0) : 1;
  int cols = layout.getRank() > 1 ? layout.getDim(1) : layout.getDim(0);
  float invBatchSize = 1.0f / (float)rows;
  dim3 blocks(cols / 8, rows);

  crossEntropyKernelClear<<<1, 1, 0, this->workspace->getStream()>>>((float *)this->loss->getGpuPtr());
  crossEntropyKernelForwardBackward<<<blocks, 1, 0, this->workspace->getStream()>>>(
    (__half *)this->dProb->getGpuPtr(),
    (float *)this->loss->getGpuPtr(),
    (const __half *)this->prob->getGpuPtr(),
    (const uint8_t *)this->target->getGpuPtr(),
    rows,
    cols,
    invBatchSize);
}
