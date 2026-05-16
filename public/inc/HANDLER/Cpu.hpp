#if !defined(HANDLER_CPU_HPP)
#define HANDLER_CPU_HPP

#include "CORE/State.hpp"

namespace HANDLER
{


class __align__(CORE::ALIGNE_TO_256) Cpu
{
private:
  void *data = NULL;
  size_t offset = 0;
  size_t capacity = 0;

public:
  Cpu(size_t capacity = CORE::MEMORY_1_GB, const char* file = __FILE__, int line = __LINE__);
  ~Cpu();

  void *getData();

  size_t getOffset();

  size_t getCapacity();

  CORE::errIO allocate(void **cpuPtr, size_t& offset, size_t capacity);

  void reset();
};

}

#endif /* HANDLER_CPU_HPP */
