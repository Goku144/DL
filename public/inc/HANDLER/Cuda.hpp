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
  Cuda(size_t capacity = CORE::MEMORY_1_GB);
  ~Cuda();

  size_t getOffset();

  size_t getCapacity();

  void allocate(size_t& offset, size_t capacity);

  void reset();

  void copyToDevice();
};
  
}

#endif /* HANDLER_CUDA_HPP */
