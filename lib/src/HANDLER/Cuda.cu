#include "HANDLER/Cuda.hpp"
#include "VIEW/Math.hpp"

#include <stdio.h>
#include <stdlib.h>

#if (CUDA_CPU == 1)
#include <cuda_runtime_api.h>
#endif

HANDLER::Cuda::Cuda(size_t capacity)
{
  if(cudaMalloc(&this->data, capacity) != cudaSuccess) CORE::logFatal("Cuda Faild to allocate MEMORY");
  if(cudaMallocHost(&this->data, capacity) != cudaSuccess)
    CORE::logFatal("Cuda Faild to allocate GPU MEMORY");
  this->capacity = capacity;
}

HANDLER::Cuda::~Cuda()
{
  if(cudaFree(this->data) != cudaSuccess)
    CORE::logWarn("Cuda Faild to free GPU MEMORY");
}

size_t HANDLER::Cuda::getOffset()
{
  return this->offset;
}

size_t HANDLER::Cuda::getCapacity()
{
  return this->capacity;
}

void HANDLER::Cuda::allocate(size_t& offset, size_t capacity)
{
  capacity = CORE::ALIGNE(capacity, CORE::ALIGNE_TO_256);

  if(capacity > this->capacity - this->offset)
  {
    CORE::logWarn("Cuda Arena Handler Out Of Memory");
    return;
  }

  offset = this->offset;
  this->offset += capacity;
}

void HANDLER::Cuda::reset()
{
  this->offset = 0;
}