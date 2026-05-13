#include "HANDLER/Cpu.hpp"
#include "VIEW/Math.hpp"

#include <stdio.h>
#include <stdlib.h>

#if (CUDA_CPU == 1)
#include <cuda_runtime_api.h>
#endif

HANDLER::Cpu::Cpu(size_t capacity)
{
#if (CUDA_CPU == 0)
  this->data = aligned_alloc(CORE::ALIGNE_TO_256, capacity);
  if(this->data == NULL) CORE::logFatal("Cpu Faild to allocate MEMORY");
#else
  if(cudaMallocHost(&this->data, capacity) != cudaSuccess)
    CORE::logFatal("Cpu Faild to allocate pinned MEMORY");
#endif
  this->capacity = capacity;
}

HANDLER::Cpu::~Cpu()
{
#if (CUDA_CPU == 0)
  free(this->data);
#else
  if(cudaFree(this->data) != cudaSuccess)
    CORE::logWarn("Cpu Faild to free pinned MEMORY");
#endif
}

size_t HANDLER::Cpu::getOffset()
{
  return this->offset;
}

size_t HANDLER::Cpu::getCapacity()
{
  return this->capacity;
}

void HANDLER::Cpu::allocate(size_t& offset, size_t capacity)
{
  capacity = CORE::ALIGNE(capacity, CORE::ALIGNE_TO_256);

  if(capacity > this->capacity - this->offset)
  {
    CORE::logWarn("Cpu Arena Handler Out Of Memory");
    return;
  }

  offset = this->offset;
  this->offset += capacity;
}

void HANDLER::Cpu::reset()
{
  this->offset = 0;
}