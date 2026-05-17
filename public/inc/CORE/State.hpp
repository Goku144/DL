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

  enum errIO
  {
    ioSuccess = 0x00,
    ioErrCopyToHost = 0x01 << 1,
    ioErrCopyToDevice = 0x01 << 2,
    ioErrOutOfMemory = 0x01 << 7,
    ioErrOutOfBound = 0x01 << 8,
    ioErrNull = 0x01 << 9,
    ioErrInvalidValue = 0x01 << 10,
    ioErrInvalidState = 0x01 << 11,
  };

  enum errFile
  {
    fileSuccess = 0x00,
    fileErrRead = 0x01 << 1,
    fileErrWrite = 0x01 << 2,
    fileErrReadCsv = 0x01 << 3,
    fileErrReadImg = 0x01 << 4,
    fileErrOpen = 0x01 << 5,
    fileErrClose = 0x01 << 6,
    fileErrNull = 0x01 << 7,
    fileErrInvalidState = 0x01 << 8,
    fileErrIO = 0x01 << 9,
  };

  enum errWorkspace
  {
    workspaceSuccess = 0x00,
    workspaceErrCudnnCreate = 0x01 << 1,
    workspaceErrCudnnDestroy = 0x01 << 2,
    workspaceErrCublasLtCreate = 0x01 << 3,
    workspaceErrCublasLtDestroy = 0x01 << 4,
    workspaceErrScratchAlloc = 0x01 << 5,
    workspaceErrScratchFree = 0x01 << 6,
    workspaceErrScratchOutOfBound = 0x01 << 7,
    workspaceErrNull = 0x01 << 8,
  };

  enum State
  {
    INFO = 0,
    WARN,
    FATAL,
  };

  #define logInfo(file, line, x, ...) printState(CORE::INFO, file, line, x, ##__VA_ARGS__);
  #define logWarn(file, line, x, ...) printState(CORE::WARN, file, line, x, ##__VA_ARGS__);
  #define logFatal(file, line, x, ...) printState(CORE::FATAL, file, line, x, ##__VA_ARGS__);
  
  void printState(State level, const char *file, int line, const char *fmt, ...);

  inline size_t aligne(size_t x, CORE::Aligne aligneTo) 
  {return (x + uintptr_t(aligneTo - 1)) & ~uintptr_t(aligneTo - 1);}

  #define ALIGNE(x, aligneTo) aligne(x, aligneTo)
}

#endif /* CORE_STATE_HPP */
