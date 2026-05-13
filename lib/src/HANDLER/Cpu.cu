#include "CORE/State.hpp"
#include "HANDLER/Cpu.hpp"

#include <stdio.h>
#include <stdlib.h>

#if (CUDA_CPU == 1)
#include <cuda_runtime_api.h>
#endif

HANDLER::Cpu::Cpu(size_t capacity)
{
#if (CUDA_CPU == 0)
  this->data = aligned_alloc(ALIGNE_TO, capacity);
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

