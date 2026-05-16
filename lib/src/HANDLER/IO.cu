#include "HANDLER/IO.hpp"

#include <string>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>

static CORE::errIO checkCpuHandling(HANDLER::Cpu *cpuHandler, VIEW::Math &math)
{
  if(cpuHandler == NULL) return CORE::ioErrNull;
  uint8_t *cpuPtr = (uint8_t *) cpuHandler->getData() + math.getCpuOffset();
  if(cpuPtr != (uint8_t *) math.getCpuPtr()) return CORE::ioErrInvalidState;
  return CORE::ioSuccess;
}

static CORE::errIO checkGpuHandling(HANDLER::Cuda *gpuHandler, VIEW::Math &math)
{
  if(gpuHandler == NULL) return CORE::ioErrNull;
  uint8_t *gpuPtr = (uint8_t *) gpuHandler->getData() + math.getGpuOffset();
  if(gpuPtr != (uint8_t *) math.getGpuPtr()) return CORE::ioErrInvalidState;
  return CORE::ioSuccess;
}

static bool isAGreaterThenB(VIEW::Math& a, VIEW::Math& b)
{
  return a.getBytes() > b.getBytes();
}

static const char *getIOErrorMessage(CORE::errIO err)
{
  if(err == CORE::ioSuccess) return "IO success";
  if(err == CORE::ioErrCopyToHost) return "IO faild to copy from device to host or from host to host";
  if(err == CORE::ioErrCopyToDevice) return "IO faild to copy from host to device or from device to device";
  if(err == CORE::ioErrReadBytes) return "IO failed to read bytes";
  if(err == CORE::ioErrWriteBytes) return "IO failed to write bytes";
  if(err == CORE::ioErrCreadCsv) return "IO failed to read csv";
  if(err == CORE::ioErrReadImg) return "IO failed to read image";
  if(err == CORE::ioErrOutOfMemory) return "IO out of memory";
  if(err == CORE::ioErrOutOfBound) return "IO out of bound";
  if(err == CORE::ioErrNull) return "IO null pointer";
  if(err == CORE::ioErrInvalidValue) return "IO invalid value";
  if(err == CORE::ioErrInvalidState) return "IO invalid state";
  if(err == CORE::ioErrOpen) return "IO failed to open";
  if(err == CORE::ioErrClose) return "IO failed to close";
  return "IO unknown error";
}

static void printMathValue(void *data, VIEW::DType dtype, size_t index)
{
  if(dtype == VIEW::CHAR)
  {
    printf("%5c", ((char *)data)[index]);
    return;
  }

  if(dtype == VIEW::F16)
  {
    printf("%5u", ((uint16_t *)data)[index]);
    return;
  }

  if(dtype == VIEW::FLOAT)
  {
    printf("%5.2f", ((float *)data)[index]);
    return;
  }
}

static void printMathIndent(int depth)
{
  for(int index = 0; index < depth; index++)
    printf("  ");
}

static void printMathRow(void *data, VIEW::DType dtype, size_t offset, int width, int depth)
{
  printMathIndent(depth);
  printf("[");
  for(int index = 0; index < width; index++)
  {
    printf(" ");
    printMathValue(data, dtype, offset + index);
  }
  printf(" ]\n");
}

static void printMathShape(VIEW::Shape &layout)
{
  printf("shape(");
  for(int index = 0; index < layout.getRank(); index++)
  {
    if(index > 0)
      printf(", ");
    printf("%d", layout.getDim(index));
  }
  printf(")\n");
}

static void printMathBlock(void *data, VIEW::DType dtype, VIEW::Shape &layout, int axis, size_t offset, int depth)
{
  if(axis == layout.getRank() - 1)
  {
    printMathRow(data, dtype, offset, layout.getDim(axis), depth);
    return;
  }

  printMathIndent(depth);
  printf("[\n");
  for(int index = 0; index < layout.getDim(axis); index++)
  {
    size_t nextOffset = offset + index * layout.getStride(axis);
    printMathBlock(data, dtype, layout, axis + 1, nextOffset, depth + 1);

    if(axis < layout.getRank() - 2 && index + 1 < layout.getDim(axis))
      printf("\n");
  }
  printMathIndent(depth);
  printf("]\n");
}

HANDLER::IO::IO(HANDLER::Cpu& handleCpu, HANDLER::Cuda& handleGpu)
{
  this->handleCpu = &handleCpu;
  this->handleGpu = &handleGpu;
}

HANDLER::IO::IO(HANDLER::Cpu& handleCpu)
{
  this->handleCpu = &handleCpu;
}

HANDLER::IO::IO(HANDLER::Cuda& handleGpu)
{
  this->handleGpu = &handleGpu;
}

HANDLER::IO::~IO()
{}

CORE::errIO HANDLER::IO::getErr()
{
  CORE::errIO err = this->err;
  this->err = CORE::ioSuccess;
  return err;
}

CORE::errIO HANDLER::IO::peekErr() const
{
  return this->err;
}

void HANDLER::IO::clearErr()
{
  this->err = CORE::ioSuccess;
}

void HANDLER::IO::setHandler(HANDLER::Cpu& handleCpu, HANDLER::Cuda& handleGpu)
{
  this->handleCpu = &handleCpu;
  this->handleGpu = &handleGpu;
}


void HANDLER::IO::setHandler(HANDLER::Cpu& handleCpu)
{
  this->handleCpu = &handleCpu;
}

void HANDLER::IO::setHandler(HANDLER::Cuda& HandleGpu)
{
  this->handleGpu = &HandleGpu;
}

void HANDLER::IO::bindCpu(VIEW::Math& math)
{
  VIEW::Shape layout = math.getLayout();
  math.setCount(layout.getDim(0) * layout.getStride(0));
  math.setBytes(CORE::ALIGNE(math.getCount() * layout.getDType(), CORE::ALIGNE_TO_256));
  void *cpuPtr;
  size_t offset = 0;
  if((this->err = handleCpu->allocate(&cpuPtr, offset, math.getBytes())) != CORE::ioSuccess) return;
  math.setCpuPtr(cpuPtr);
}

void HANDLER::IO::bindGpu(VIEW::Math& math)
{
  VIEW::Shape layout = math.getLayout();
  math.setCount(layout.getDim(0) * layout.getStride(0));
  math.setBytes(CORE::ALIGNE(math.getCount() * layout.getDType(), CORE::ALIGNE_TO_256));
  void *gpuPtr;
  size_t offset = 0;
  if((this->err = handleGpu->allocate(&gpuPtr, offset, math.getBytes())) != CORE::ioSuccess) return;
  math.setGpuPtr(gpuPtr);
}

void HANDLER::IO::bind(VIEW::Math& math)
{
  this->bindCpu(math);
  this->bindGpu(math);
}

void unbind(VIEW::Math& math)
{
  math.setBytes(0);
  math.setCount(0);
  math.setCpuOffset(0);
  math.setGpuOffset(0);
  math.setCpuPtr(NULL);
  math.setGpuPtr(NULL);
  math.getLayout().setShape((int[]){0,0,0,0});
}

void HANDLER::IO::copyHostToHost(VIEW::Math& dstMath, VIEW::Math& srcMath)
{
  if((this->err = checkCpuHandling(this->handleCpu, srcMath)) != CORE::ioSuccess) return;
  if((this->err = checkCpuHandling(this->handleCpu, dstMath)) != CORE::ioSuccess) return;
  if(!isAGreaterThenB(dstMath, srcMath)) 
  {
    this->err = CORE::ioErrOutOfBound;
    return;
  }
#if CUDA_CPU == 1
  if(cudaMemcpy(dstMath.getCpuPtr(), srcMath.getCpuPtr(), srcMath.getBytes(), cudaMemcpyHostToHost) != cudaSuccess)
  {
    this->err = CORE::ioErrCopyToHost;
    return;
  }
#else
  memcpy(dstMath.getCpuPtr(), srcMath.getCpuPtr(), srcMath.getBytes());
#endif
  dstMath.setLayout(srcMath.getLayout());
}

void HANDLER::IO::copyHostToHost(VIEW::Math& dstMath, void *src, size_t n, VIEW::DType dtype)
{
  if(src == NULL)
  {
    this->err = CORE::ioErrNull;
    return;
  }
  if((this->err = checkCpuHandling(this->handleCpu, dstMath)) != CORE::ioSuccess) return;
  if(dstMath.getBytes() < n) 
  {
    this->err = CORE::ioErrOutOfBound;
    return;
  }
#if CUDA_CPU == 1
  if(cudaMemcpy(dstMath.getCpuPtr(), src, n * dtype, cudaMemcpyHostToHost) != cudaSuccess)
  {
    this->err = CORE::ioErrCopyToHost;
    return;
  }
#else
  memcpy(dstMath.getCpuPtr(), src, n * dtype);
#endif
  int dims[VIEW::MAX_RANK] = {(int)n, 0, 0, 0};
  dstMath.getLayout().setShape(dims, 1, dtype);
}

void HANDLER::IO::copyHostToHost(void *dst, VIEW::Math& srcMath, size_t n, VIEW::DType dtype)
{
  if(dst == NULL)
  {
    this->err = CORE::ioErrNull;
    return;
  }
  if((this->err = checkCpuHandling(this->handleCpu, srcMath)) != CORE::ioSuccess) return;
  if(srcMath.getBytes() < n) 
  {
    this->err = CORE::ioErrOutOfBound;
    return;
  }
#if CUDA_CPU == 1
  if(cudaMemcpy(dst, srcMath.getCpuPtr(), n * dtype, cudaMemcpyHostToHost) != cudaSuccess)
  {
    this->err = CORE::ioErrCopyToHost;
    return;
  }
#else
  memcpy(dst, srcMath.getCpuPtr(), n * dtype);
#endif
}

void HANDLER::IO::copyHostToDevice(VIEW::Math& math)
{
  if((this->err = checkCpuHandling(this->handleCpu, math)) != CORE::ioSuccess) return;
  if((this->err = checkGpuHandling(this->handleGpu, math)) != CORE::ioSuccess) return;
  if(cudaMemcpy(math.getGpuPtr(), math.getCpuPtr(), math.getBytes(), cudaMemcpyHostToDevice) != cudaSuccess)
  {
    this->err = CORE::ioErrCopyToDevice;
    return;
  }
}

void HANDLER::IO::copyHostToDevice(VIEW::Math& dstMath, VIEW::Math& srcMath)
{
  if((this->err = checkCpuHandling(this->handleCpu, srcMath)) != CORE::ioSuccess) return;
  if((this->err = checkGpuHandling(this->handleGpu, dstMath)) != CORE::ioSuccess) return;
  if(!isAGreaterThenB(dstMath, srcMath)) 
  {
    this->err = CORE::ioErrOutOfBound;
    return;
  }
  if(cudaMemcpy(dstMath.getGpuPtr(), srcMath.getCpuPtr(), srcMath.getBytes(), cudaMemcpyHostToDevice) != cudaSuccess)
  {
    this->err = CORE::ioErrCopyToHost;
    return;
  }
  dstMath.setLayout(srcMath.getLayout());
}

void HANDLER::IO::copyHostToDevice(VIEW::Math& dstMath, void *src, size_t n, VIEW::DType dtype)
{
  if(src == NULL)
  {
    this->err = CORE::ioErrNull;
    return;
  }
  if((this->err = checkGpuHandling(this->handleGpu, dstMath)) != CORE::ioSuccess) return;
  if(dstMath.getBytes() < n) 
  {
    this->err = CORE::ioErrOutOfBound;
    return;
  }
  if(cudaMemcpy(dstMath.getGpuPtr(), src, n * dtype, cudaMemcpyHostToDevice) != cudaSuccess)
  {
    this->err = CORE::ioErrCopyToHost;
    return;
  }
  int dims[VIEW::MAX_RANK] = {(int)n, 0, 0, 0};
  dstMath.getLayout().setShape(dims, 1, dtype);
}

void HANDLER::IO::copyHostToDevice(void *dst, VIEW::Math& srcMath, size_t n, VIEW::DType dtype)
{
  if(dst == NULL)
  {
    this->err = CORE::ioErrNull;
    return;
  }
  if((this->err = checkCpuHandling(this->handleCpu, srcMath)) != CORE::ioSuccess) return;
  if(srcMath.getBytes() < n) 
  {
    this->err = CORE::ioErrOutOfBound;
    return;
  }
  if(cudaMemcpy(dst, srcMath.getCpuPtr(), n * dtype, cudaMemcpyHostToDevice) != cudaSuccess)
  {
    this->err = CORE::ioErrCopyToHost;
    return;
  }
}

void HANDLER::IO::copyDeviceToHost(VIEW::Math& math)
{
  if((this->err = checkCpuHandling(this->handleCpu, math)) != CORE::ioSuccess) return;
  if((this->err = checkGpuHandling(this->handleGpu, math)) != CORE::ioSuccess) return;
  if(cudaMemcpy(math.getCpuPtr(), math.getGpuPtr(), math.getBytes(), cudaMemcpyDeviceToHost) != cudaSuccess)
  {
    this->err = CORE::ioErrCopyToHost;
    return;
  }
}

void HANDLER::IO::copyDeviceToHost(VIEW::Math& dstMath, VIEW::Math& srcMath)
{
  if((this->err = checkGpuHandling(this->handleGpu, srcMath)) != CORE::ioSuccess) return;
  if((this->err = checkCpuHandling(this->handleCpu, dstMath)) != CORE::ioSuccess) return;
  if(!isAGreaterThenB(dstMath, srcMath)) 
  {
    this->err = CORE::ioErrOutOfBound;
    return;
  }
  if(cudaMemcpy(dstMath.getCpuPtr(), srcMath.getGpuPtr(), srcMath.getBytes(), cudaMemcpyDeviceToHost) != cudaSuccess)
  {
    this->err = CORE::ioErrCopyToHost;
    return;
  }
  dstMath.setLayout(srcMath.getLayout());
}

void HANDLER::IO::copyDeviceToHost(VIEW::Math& dstMath, void *src, size_t n, VIEW::DType dtype)
{
  if(src == NULL)
  {
    this->err = CORE::ioErrNull;
    return;
  }
  if((this->err = checkCpuHandling(this->handleCpu, dstMath)) != CORE::ioSuccess) return;
  if(dstMath.getBytes() < n) 
  {
    this->err = CORE::ioErrOutOfBound;
    return;
  }
  if(cudaMemcpy(dstMath.getCpuPtr(), src, n * dtype, cudaMemcpyDeviceToHost) != cudaSuccess)
  {
    this->err = CORE::ioErrCopyToHost;
    return;
  }
  int dims[VIEW::MAX_RANK] = {(int)n, 0, 0, 0};
  dstMath.getLayout().setShape(dims, 1, dtype);
}

void HANDLER::IO::copyDeviceToHost(void *dst, VIEW::Math& srcMath, size_t n, VIEW::DType dtype)
{
  if(dst == NULL)
  {
    this->err = CORE::ioErrNull;
    return;
  }
  if((this->err = checkGpuHandling(this->handleGpu, srcMath)) != CORE::ioSuccess) return;
  if(srcMath.getBytes() < n) 
  {
    this->err = CORE::ioErrOutOfBound;
    return;
  }
  if(cudaMemcpy(dst, srcMath.getGpuPtr(), n * dtype, cudaMemcpyDeviceToHost) != cudaSuccess)
  {
    this->err = CORE::ioErrCopyToHost;
    return;
  }
}

void HANDLER::IO::copyDeviceToDevice(VIEW::Math& dstMath, VIEW::Math& srcMath)
{
  if((this->err = checkGpuHandling(this->handleGpu, srcMath)) != CORE::ioSuccess) return;
  if((this->err = checkGpuHandling(this->handleGpu, dstMath)) != CORE::ioSuccess) return;
  if(!isAGreaterThenB(dstMath, srcMath)) 
  {
    this->err = CORE::ioErrOutOfBound;
    return;
  }
  if(cudaMemcpy(dstMath.getGpuPtr(), srcMath.getGpuPtr(), srcMath.getBytes(), cudaMemcpyDeviceToDevice) != cudaSuccess)
  {
    this->err = CORE::ioErrCopyToHost;
    return;
  }
  dstMath.setLayout(srcMath.getLayout());
}

void HANDLER::IO::copyDeviceToDevice(VIEW::Math& dstMath, void *src, size_t n, VIEW::DType dtype)
{
  if(src == NULL)
  {
    this->err = CORE::ioErrNull;
    return;
  }
  if((this->err = checkGpuHandling(this->handleGpu, dstMath)) != CORE::ioSuccess) return;
  if(dstMath.getBytes() < n) 
  {
    this->err = CORE::ioErrOutOfBound;
    return;
  }
  if(cudaMemcpy(dstMath.getGpuPtr(), src, n * dtype, cudaMemcpyDeviceToDevice) != cudaSuccess)
  {
    this->err = CORE::ioErrCopyToHost;
    return;
  }
  int dims[VIEW::MAX_RANK] = {(int)n, 0, 0, 0};
  dstMath.getLayout().setShape(dims, 1, dtype);
}

void HANDLER::IO::copyDeviceToDevice(void *dst, VIEW::Math& srcMath, size_t n, VIEW::DType dtype)
{
  if(dst == NULL)
  {
    this->err = CORE::ioErrNull;
    return;
  }
  if((this->err = checkGpuHandling(this->handleGpu, srcMath)) != CORE::ioSuccess) return;
  if(srcMath.getBytes() < n) 
  {
    this->err = CORE::ioErrOutOfBound;
    return;
  }
  if(cudaMemcpy(dst, srcMath.getGpuPtr(), n * dtype, cudaMemcpyDeviceToDevice) != cudaSuccess)
  {
    this->err = CORE::ioErrCopyToHost;
    return;
  }
}

void HANDLER::IO::printData(VIEW::Math& math, const char* file, int line) const
{
  VIEW::Shape layout = math.getLayout();

  CORE::logInfo(file, line, "Math (data):");

  void *ptr = (void *) ((uint8_t *)this->handleCpu->getData() + math.getCpuOffset());
  VIEW::DType dtype = layout.getDType();
  int rank = layout.getRank();

  if(rank <= 0)
  {
    printf("[]\n");
    return;
  }

  printMathShape(layout);
  printMathBlock(ptr, dtype, layout, 0, 0, 0);
}

void HANDLER::IO::info(CORE::State level, const char *file, int line) const
{
  if(this->err == CORE::ioSuccess) return;

  if(level == CORE::FATAL)
  {
    CORE::logFatal(file, line, "%s", getIOErrorMessage(this->err));
    return;
  }

  CORE::logWarn(file, line, "%s", getIOErrorMessage(this->err));
}