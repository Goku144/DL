#include "HANDLER/Cpu.hpp"
#include "HANDLER/IO.hpp"
#include "VIEW/Math.hpp"

#include <stdio.h>
#include <stdlib.h>

#if (CUDA_CPU == 1)
#include <cuda_runtime_api.h>
#endif

HANDLER::Cpu::Cpu(size_t capacity, const char* file, int line)
{
#if (CUDA_CPU == 0)
  this->data = aligned_alloc(CORE::ALIGNE_TO_256, capacity);
  if(this->data == NULL) CORE::logFatal(file, line, "Cpu Faild to allocate MEMORY");
#else
  if(cudaMallocHost(&this->data, capacity) != cudaSuccess)
    CORE::logFatal(file, line, "Cpu Faild to allocate pinned MEMORY");
#endif
  this->capacity = capacity;
}

HANDLER::Cpu::~Cpu()
{
#if (CUDA_CPU == 0)
  free(this->data);
#else
  if(cudaFree(this->data) != cudaSuccess)
    CORE::logWarn(__FILE__, __LINE__, "Cpu Faild to free pinned MEMORY");
#endif
}

void *HANDLER::Cpu::getData()
{
  return this->data;
}

size_t HANDLER::Cpu::getOffset()
{
  return this->offset;
}

size_t HANDLER::Cpu::getCapacity()
{
  return this->capacity;
}

CORE::errIO HANDLER::Cpu::allocate(void **cpuPtr, size_t& offset, size_t capacity)
{
  if(cpuPtr == NULL) return CORE::ioErrNull;

  capacity = CORE::ALIGNE(capacity, CORE::ALIGNE_TO_256);

  if(capacity > this->capacity - this->offset)
    return CORE::ioErrOutOfBound;

  *cpuPtr = (uint8_t *) this->data + offset;
  offset = this->offset;
  this->offset += capacity;
  return CORE::ioSuccess;
}

void HANDLER::Cpu::reset()
{
  this->offset = 0;
}