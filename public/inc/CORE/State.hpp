#if !defined(CORE_STATE_HPP)
#define CORE_STATE_HPP

#include <stdint.h>

namespace CORE
{
  #define CUDA_CPU 0

  enum Aligne
  {
    ALIGNE_TO_32 = 32,
    ALIGNE_TO_64 = 64,
    ALIGNE_TO_128 = 128,
    ALIGNE_TO_256 = 256,
  };

  enum MemorySize
  {
    MEMORY_32_MB = 32 * 1024 * 1024,
    MEMORY_64_MB = 2 * MEMORY_32_MB,
    MEMORY_128_MB = 2 * MEMORY_64_MB,
    MEMORY_256_MB = 2 * MEMORY_128_MB,
    MEMORY_512_MB = 2 * MEMORY_256_MB,
    MEMORY_1_GB = 2 * MEMORY_512_MB,
  };

    enum State
  {
    INFO = 0,
    WARN,
    FATAL,
  };

  #define logInfo(x, ...) printState(CORE::INFO, __FILE__, __LINE__, x, ##__VA_ARGS__);
  #define logWarn(x, ...) printState(CORE::WARN, __FILE__, __LINE__, x, ##__VA_ARGS__);
  #define logFatal(x, ...) printState(CORE::FATAL, __FILE__, __LINE__, x, ##__VA_ARGS__);
  
  void printState(State level, const char *file, int line, const char *fmt, ...);

  inline size_t aligne(size_t& x, Aligne aligneTo);

  #define ALIGNE(x, aligneTo) aligne(x, aligneTo)
}

#endif /* CORE_STATE_HPP */
