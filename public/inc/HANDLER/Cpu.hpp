#if !defined(HANDLE_CPU_HPP)
#define HANDLE_CPU_HPP

#include <stdint.h>

namespace HANDLER
{

#define CUDA_CPU 0
#define ALIGNE_TO 256

#define ALIGNE_256(x) (x + uintptr_t(ALIGNE_TO - 1)) & ~uintptr_t(ALIGNE_TO - 1)
#define DEFAULT_MEMORY_SIZE 1 * 1024 * 1024 * 1024

class __align__(ALIGNE_TO) Cpu
{
private:
  void *data = NULL;
  size_t offset = 0;
  size_t capacity = 0;

public:
  Cpu(size_t capacity = DEFAULT_MEMORY_SIZE);
  ~Cpu();

  size_t getOffset();

  size_t getCapacity();

  void allocate(size_t& offset);

  void reset();

  void copy(HANDLER::Cpu& dst, const HANDLER::Cpu src);
  
  CORE::State writeFile();

  CORE::State readFile();

  CORE::State readText();

#if (CUDA_CPU == 1)
  void copyToHost();
#endif
};
  
}

#endif /* HANDLE_CPU_HPP */
