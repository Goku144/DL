#include "HANDLER/Cuda.hpp"
#include "HANDLER/IO.hpp"
#include "VIEW/Math.hpp"

#include <stdio.h>
#include <stdlib.h>

#if (CUDA_CPU == 1)
#include <cuda_runtime_api.h>
#endif

HANDLER::Cuda::Cuda(size_t capacity, const char* file, int line)
{
  if(cudaMalloc(&this->data, capacity) != cudaSuccess) 
    CORE::logFatal(file, line, "Cuda Faild to allocate MEMORY");
  this->capacity = capacity;
}

HANDLER::Cuda::~Cuda()
{
  if(cudaFree(this->data) != cudaSuccess)
    CORE::logWarn(__FILE__, __LINE__,"Cuda Faild to free GPU MEMORY");
}

void *HANDLER::Cuda::getData()
{
  return this->data;
}

size_t HANDLER::Cuda::getOffset()
{
  return this->offset;
}

size_t HANDLER::Cuda::getCapacity()
{
  return this->capacity;
}


CORE::errIO HANDLER::Cuda::allocate(void **gpuPtr, size_t& offset, size_t capacity)
{
  if(gpuPtr == NULL) return CORE::ioErrNull;

  capacity = CORE::ALIGNE(capacity, CORE::ALIGNE_TO_256);

  if(capacity > this->capacity - this->offset) 
    return CORE::ioErrOutOfBound;
    
  *gpuPtr = (uint8_t *) this->data + offset;
  offset = this->offset;
  this->offset += capacity;
  return CORE::ioSuccess;
}

void HANDLER::Cuda::reset()
{
  this->offset = 0;
}