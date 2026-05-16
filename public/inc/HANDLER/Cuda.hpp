#if !defined(HANDLER_CUDA_HPP)
#define HANDLER_CUDA_HPP

#include "CORE/State.hpp"

namespace HANDLER
{


class __align__(CORE::ALIGNE_TO_256) Cuda
{
private:
  void *data = NULL;
  size_t offset = 0;
  size_t capacity = 0;

public:
  Cuda(size_t capacity = CORE::MEMORY_1_GB, const char* file = __FILE__, int line = __LINE__);
  ~Cuda();

  void *getData();

  size_t getOffset();

  size_t getCapacity();

  CORE::errIO allocate(void **gpuPtr, size_t& offset, size_t capacity);

  void reset();
};
  
}

#endif /* HANDLER_CUDA_HPP */