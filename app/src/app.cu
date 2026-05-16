#include "HANDLER/IO.hpp"
#include "VIEW/Math.hpp"

#include <cuda_runtime_api.h>
#include <stdint.h>
#include <stdio.h>

static const size_t TEST_COUNT = 8;

static void logSection(HANDLER::IO& io, const char *name)
{
  printf("\n--- %s ---\n", name);
  io.info(CORE::WARN, __FILE__, __LINE__);
  io.clearErr();
}

static int expectIO(HANDLER::IO& io, const char *name)
{
  if(io.peekErr() == CORE::ioSuccess) return 0;

  printf("[ FAIL ] %s\n", name);
  io.info(CORE::WARN, __FILE__, __LINE__);
  io.clearErr();
  return 1;
}

static int expectTrue(const char *name, bool condition)
{
  if(condition)
  {
    printf("[ PASS ] %s\n", name);
    return 0;
  }

  printf("[ FAIL ] %s\n", name);
  return 1;
}

static int expectFloatArray(const char *name, const float *actual, const float *expected, size_t count)
{
  for(size_t index = 0; index < count; index++)
  {
    if(actual[index] != expected[index])
    {
      printf("[ FAIL ] %s at %zu: got %.2f expected %.2f\n", name, index, actual[index], expected[index]);
      return 1;
    }
  }

  printf("[ PASS ] %s\n", name);
  return 0;
}

static int expectU16Array(const char *name, const uint16_t *actual, const uint16_t *expected, size_t count)
{
  for(size_t index = 0; index < count; index++)
  {
    if(actual[index] != expected[index])
    {
      printf("[ FAIL ] %s at %zu: got %u expected %u\n", name, index, actual[index], expected[index]);
      return 1;
    }
  }

  printf("[ PASS ] %s\n", name);
  return 0;
}

static int expectCharArray(const char *name, const char *actual, const char *expected, size_t count)
{
  for(size_t index = 0; index < count; index++)
  {
    if(actual[index] != expected[index])
    {
      printf("[ FAIL ] %s at %zu: got %d expected %d\n", name, index, actual[index], expected[index]);
      return 1;
    }
  }

  printf("[ PASS ] %s\n", name);
  return 0;
}

static int expectDeviceFloatArray(const char *name, void *devicePtr, const float *expected, size_t count)
{
  float actual[TEST_COUNT] = {0.0f};

  if(count > TEST_COUNT)
  {
    printf("[ FAIL ] %s count is too large for test buffer\n", name);
    return 1;
  }

  if(cudaMemcpy(actual, devicePtr, count * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
  {
    printf("[ FAIL ] %s cudaMemcpy device to host failed\n", name);
    return 1;
  }

  return expectFloatArray(name, actual, expected, count);
}

static int expectCpuInvariant(const char *name, HANDLER::Cpu& cpu, VIEW::Math& math)
{
  void *expected = (uint8_t *)cpu.getData() + math.getCpuOffset();
  return expectTrue(name, expected == math.getCpuPtr());
}

static int expectGpuInvariant(const char *name, HANDLER::Cuda& gpu, VIEW::Math& math)
{
  void *expected = (uint8_t *)gpu.getData() + math.getGpuOffset();
  return expectTrue(name, expected == math.getGpuPtr());
}

static int expectShape(const char *name, VIEW::Math& math, int dim0, int rank, VIEW::DType dtype)
{
  VIEW::Shape& layout = math.getLayout();
  return expectTrue(name,
    layout.getDim(0) == dim0 &&
    layout.getRank() == rank &&
    layout.getDType() == dtype);
}

static void clearFloatArray(float *data, size_t count)
{
  for(size_t index = 0; index < count; index++)
    data[index] = 0.0f;
}

int main()
{
  HANDLER::Cpu cpu(CORE::MEMORY_32_MB);
  HANDLER::Cuda gpu(CORE::MEMORY_32_MB);
  HANDLER::IO io(cpu, gpu);
  HANDLER::IO cpuOnly(cpu);
  HANDLER::IO gpuOnly(gpu);
  int failures = 0;

  cpuOnly.setHandler(cpu);
  gpuOnly.setHandler(gpu);
  io.setHandler(cpu, gpu);
  failures += expectIO(io, "setHandler cpu and gpu");
  failures += expectIO(cpuOnly, "setHandler cpu only");
  failures += expectIO(gpuOnly, "setHandler gpu only");
  failures += expectTrue("peekErr starts success", io.peekErr() == CORE::ioSuccess);
  failures += expectTrue("getErr clears success state", io.getErr() == CORE::ioSuccess);
  logSection(io, "handler setup and error helpers");

  int srcDims[VIEW::MAX_RANK] = {8, 0, 0, 0};
  int dstDims[VIEW::MAX_RANK] = {96, 0, 0, 0};
  int matrixDims[VIEW::MAX_RANK] = {2, 4, 0, 0};

  VIEW::Math hostSrc;
  VIEW::Math hostDst;
  VIEW::Math deviceSrc;
  VIEW::Math deviceDst;
  VIEW::Math sharedSrc;
  VIEW::Math sharedDst;
  VIEW::Math matrix;

  hostSrc.getLayout().setShape(srcDims, 1, VIEW::FLOAT);
  hostDst.getLayout().setShape(dstDims, 1, VIEW::FLOAT);
  deviceSrc.getLayout().setShape(srcDims, 1, VIEW::FLOAT);
  deviceDst.getLayout().setShape(dstDims, 1, VIEW::FLOAT);
  sharedSrc.getLayout().setShape(srcDims, 1, VIEW::FLOAT);
  sharedDst.getLayout().setShape(dstDims, 1, VIEW::FLOAT);
  matrix.getLayout().setShape(matrixDims, 2, VIEW::FLOAT);

  io.bindCpu(hostSrc);
  failures += expectIO(io, "bindCpu hostSrc");
  failures += expectCpuInvariant("hostSrc cpu invariant", cpu, hostSrc);
  io.bindCpu(hostDst);
  failures += expectIO(io, "bindCpu hostDst");
  failures += expectCpuInvariant("hostDst cpu invariant", cpu, hostDst);
  io.bindGpu(deviceSrc);
  failures += expectIO(io, "bindGpu deviceSrc");
  failures += expectGpuInvariant("deviceSrc gpu invariant", gpu, deviceSrc);
  io.bindGpu(deviceDst);
  failures += expectIO(io, "bindGpu deviceDst");
  failures += expectGpuInvariant("deviceDst gpu invariant", gpu, deviceDst);
  io.bind(sharedSrc);
  failures += expectIO(io, "bind sharedSrc");
  failures += expectCpuInvariant("sharedSrc cpu invariant", cpu, sharedSrc);
  failures += expectGpuInvariant("sharedSrc gpu invariant", gpu, sharedSrc);
  io.bind(sharedDst);
  failures += expectIO(io, "bind sharedDst");
  failures += expectCpuInvariant("sharedDst cpu invariant", cpu, sharedDst);
  failures += expectGpuInvariant("sharedDst gpu invariant", gpu, sharedDst);
  io.bindCpu(matrix);
  failures += expectIO(io, "bindCpu matrix");
  failures += expectShape("matrix keeps 2D float shape after bind", matrix, 2, 2, VIEW::FLOAT);
  logSection(io, "bind and invariants");

  float hostInput[TEST_COUNT] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  float hostOutput[TEST_COUNT] = {0.0f};
  float hostDeviceOutput[TEST_COUNT] = {0.0f};
  void *deviceInput = NULL;
  void *deviceOutput = NULL;

  cudaMalloc(&deviceInput, sizeof(hostInput));
  cudaMalloc(&deviceOutput, sizeof(hostInput));
  cudaMemcpy(deviceInput, hostInput, sizeof(hostInput), cudaMemcpyHostToDevice);

  io.copyHostToHost(hostSrc, hostInput, TEST_COUNT, VIEW::FLOAT);
  failures += expectIO(io, "copyHostToHost Math <- host pointer");
  failures += expectShape("hostSrc raw copy sets 1D float shape", hostSrc, TEST_COUNT, 1, VIEW::FLOAT);
  io.copyHostToHost(hostDst, hostSrc);
  failures += expectIO(io, "copyHostToHost Math <- Math");
  failures += expectShape("hostDst Math copy gets source shape", hostDst, TEST_COUNT, 1, VIEW::FLOAT);
  io.copyHostToHost(hostOutput, hostDst, TEST_COUNT, VIEW::FLOAT);
  failures += expectIO(io, "copyHostToHost host pointer <- Math");
  failures += expectFloatArray("host to host data", hostOutput, hostInput, TEST_COUNT);
  logSection(io, "host to host overloads");
  io.printData(hostDst, __FILE__, __LINE__);

  io.copyHostToHost(sharedSrc, hostInput, TEST_COUNT, VIEW::FLOAT);
  failures += expectIO(io, "copyHostToHost shared Math <- host pointer");
  io.copyHostToDevice(sharedSrc);
  failures += expectIO(io, "copyHostToDevice same Math");
  failures += expectDeviceFloatArray("same Math host to device data", sharedSrc.getGpuPtr(), hostInput, TEST_COUNT);
  clearFloatArray((float *)sharedSrc.getCpuPtr(), TEST_COUNT);
  io.copyDeviceToHost(sharedSrc);
  failures += expectIO(io, "copyDeviceToHost same Math");
  failures += expectFloatArray("same Math device to host data", (float *)sharedSrc.getCpuPtr(), hostInput, TEST_COUNT);
  logSection(io, "same Math host/device sync");

  io.copyHostToDevice(deviceDst, hostSrc);
  failures += expectIO(io, "copyHostToDevice GPU Math <- CPU Math");
  failures += expectDeviceFloatArray("host Math to device Math data", deviceDst.getGpuPtr(), hostInput, TEST_COUNT);
  io.copyHostToDevice(deviceSrc, hostInput, TEST_COUNT, VIEW::FLOAT);
  failures += expectIO(io, "copyHostToDevice GPU Math <- host pointer");
  failures += expectDeviceFloatArray("host pointer to device Math data", deviceSrc.getGpuPtr(), hostInput, TEST_COUNT);
  io.copyHostToDevice(deviceOutput, hostSrc, TEST_COUNT, VIEW::FLOAT);
  failures += expectIO(io, "copyHostToDevice device pointer <- CPU Math");
  failures += expectDeviceFloatArray("host Math to device pointer data", deviceOutput, hostInput, TEST_COUNT);
  logSection(io, "host to device overloads");

  io.copyDeviceToHost(hostDst, deviceSrc);
  failures += expectIO(io, "copyDeviceToHost CPU Math <- GPU Math");
  failures += expectFloatArray("device Math to host Math data", (float *)hostDst.getCpuPtr(), hostInput, TEST_COUNT);
  io.copyDeviceToHost(hostDst, deviceInput, TEST_COUNT, VIEW::FLOAT);
  failures += expectIO(io, "copyDeviceToHost CPU Math <- device pointer");
  failures += expectFloatArray("device pointer to host Math data", (float *)hostDst.getCpuPtr(), hostInput, TEST_COUNT);
  io.copyDeviceToHost(hostDeviceOutput, deviceDst, TEST_COUNT, VIEW::FLOAT);
  failures += expectIO(io, "copyDeviceToHost host pointer <- GPU Math");
  failures += expectFloatArray("device Math to host pointer data", hostDeviceOutput, hostInput, TEST_COUNT);
  logSection(io, "device to host overloads");
  io.printData(hostDst, __FILE__, __LINE__);

  io.copyDeviceToDevice(deviceDst, deviceSrc);
  failures += expectIO(io, "copyDeviceToDevice GPU Math <- GPU Math");
  failures += expectDeviceFloatArray("device Math to device Math data", deviceDst.getGpuPtr(), hostInput, TEST_COUNT);
  io.copyDeviceToDevice(deviceDst, deviceInput, TEST_COUNT, VIEW::FLOAT);
  failures += expectIO(io, "copyDeviceToDevice GPU Math <- device pointer");
  failures += expectDeviceFloatArray("device pointer to device Math data", deviceDst.getGpuPtr(), hostInput, TEST_COUNT);
  io.copyDeviceToDevice(deviceOutput, deviceSrc, TEST_COUNT, VIEW::FLOAT);
  failures += expectIO(io, "copyDeviceToDevice device pointer <- GPU Math");
  failures += expectDeviceFloatArray("device Math to device pointer data", deviceOutput, hostInput, TEST_COUNT);
  logSection(io, "device to device overloads");

  float matrixInput[TEST_COUNT] = {10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f};
  io.copyHostToHost(matrix, matrixInput, TEST_COUNT, VIEW::FLOAT);
  failures += expectIO(io, "copyHostToHost matrix data");
  matrix.getLayout().setShape(matrixDims, 2, VIEW::FLOAT);
  io.printData(matrix, __FILE__, __LINE__);
  logSection(io, "2D printData logic");

  int charDims[VIEW::MAX_RANK] = {8, 0, 0, 0};
  int f16Dims[VIEW::MAX_RANK] = {8, 0, 0, 0};
  VIEW::Math charMath;
  VIEW::Math f16Math;
  charMath.getLayout().setShape(charDims, 1, VIEW::CHAR);
  f16Math.getLayout().setShape(f16Dims, 1, VIEW::F16);
  io.bind(charMath);
  failures += expectIO(io, "bind CHAR Math");
  io.bind(f16Math);
  failures += expectIO(io, "bind F16 Math");

  char charInput[TEST_COUNT] = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'};
  char charOutput[TEST_COUNT] = {0};
  uint16_t f16Input[TEST_COUNT] = {100, 101, 102, 103, 104, 105, 106, 107};
  uint16_t f16Output[TEST_COUNT] = {0};

  io.copyHostToHost(charMath, charInput, TEST_COUNT, VIEW::CHAR);
  failures += expectIO(io, "copyHostToHost CHAR Math <- host pointer");
  io.copyHostToDevice(charMath);
  failures += expectIO(io, "copyHostToDevice CHAR same Math");
  for(size_t index = 0; index < TEST_COUNT; index++)
    ((char *)charMath.getCpuPtr())[index] = 0;
  io.copyDeviceToHost(charMath);
  failures += expectIO(io, "copyDeviceToHost CHAR same Math");
  io.copyHostToHost(charOutput, charMath, TEST_COUNT, VIEW::CHAR);
  failures += expectIO(io, "copyHostToHost CHAR host pointer <- Math");
  failures += expectCharArray("CHAR round trip data", charOutput, charInput, TEST_COUNT);

  io.copyHostToHost(f16Math, f16Input, TEST_COUNT, VIEW::F16);
  failures += expectIO(io, "copyHostToHost F16 Math <- host pointer");
  io.copyHostToDevice(f16Math);
  failures += expectIO(io, "copyHostToDevice F16 same Math");
  for(size_t index = 0; index < TEST_COUNT; index++)
    ((uint16_t *)f16Math.getCpuPtr())[index] = 0;
  io.copyDeviceToHost(f16Math);
  failures += expectIO(io, "copyDeviceToHost F16 same Math");
  io.copyHostToHost(f16Output, f16Math, TEST_COUNT, VIEW::F16);
  failures += expectIO(io, "copyHostToHost F16 host pointer <- Math");
  failures += expectU16Array("F16 round trip data", f16Output, f16Input, TEST_COUNT);
  logSection(io, "dtype coverage");

  io.unbind(charMath);
  failures += expectTrue("unbind clears bytes", charMath.getBytes() == 0);
  failures += expectTrue("unbind clears count", charMath.getCount() == 0);
  failures += expectTrue("unbind clears cpu pointer", charMath.getCpuPtr() == NULL);
  failures += expectTrue("unbind clears gpu pointer", charMath.getGpuPtr() == NULL);
  failures += expectShape("unbind clears shape", charMath, 0, 0, VIEW::CHAR);

  cpu.reset();
  gpu.reset();
  VIEW::Math resetMath;
  resetMath.getLayout().setShape(srcDims, 1, VIEW::FLOAT);
  io.bind(resetMath);
  failures += expectIO(io, "bind after handler reset");
  failures += expectTrue("reset cpu offset reused from zero", resetMath.getCpuOffset() == 0);
  failures += expectTrue("reset gpu offset reused from zero", resetMath.getGpuOffset() == 0);
  failures += expectCpuInvariant("resetMath cpu invariant", cpu, resetMath);
  failures += expectGpuInvariant("resetMath gpu invariant", gpu, resetMath);
  logSection(io, "unbind and reset");

  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  if(failures != 0)
    printf("\nIO full deep logic test failed: %d failure(s)\n", failures);
  else
    printf("\nIO full deep logic test passed\n");

  return failures == 0 ? 0 : 1;
}
