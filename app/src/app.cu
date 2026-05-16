#include "HANDLER/IO.hpp"
#include "VIEW/Math.hpp"

#include <cuda_runtime_api.h>
#include <stdio.h>

static void bindCpuForTest(HANDLER::Cpu& cpu, VIEW::Math& math)
{
  VIEW::Shape layout = math.getLayout();
  math.setCount(layout.getDim(0) * layout.getStride(0));
  math.setBytes(CORE::ALIGNE(math.getCount() * layout.getDType(), CORE::ALIGNE_TO_256));

  void *cpuPtr = NULL;
  size_t offset = cpu.getOffset();
  cpu.allocate(&cpuPtr, offset, math.getBytes());
  math.setCpuPtr(cpuPtr);
  math.setCpuOffset(offset);
}

static void bindGpuForTest(HANDLER::Cuda& gpu, VIEW::Math& math)
{
  VIEW::Shape layout = math.getLayout();
  math.setCount(layout.getDim(0) * layout.getStride(0));
  math.setBytes(CORE::ALIGNE(math.getCount() * layout.getDType(), CORE::ALIGNE_TO_256));

  void *gpuPtr = NULL;
  size_t offset = gpu.getOffset();
  gpu.allocate(&gpuPtr, offset, math.getBytes());
  math.setGpuPtr(gpuPtr);
  math.setGpuOffset(offset);
}

static void bindForTest(HANDLER::Cpu& cpu, HANDLER::Cuda& gpu, VIEW::Math& math)
{
  bindCpuForTest(cpu, math);
  bindGpuForTest(gpu, math);
}

static void logStep(HANDLER::IO& io, const char *name)
{
  printf("\n--- %s ---\n", name);
  io.info(CORE::WARN, __FILE__, __LINE__);
  io.clearErr();
}

int main()
{
  HANDLER::Cpu cpu(CORE::MEMORY_32_MB);
  HANDLER::Cuda gpu(CORE::MEMORY_32_MB);
  HANDLER::IO io(cpu, gpu);
  HANDLER::IO cpuOnly(cpu);
  HANDLER::IO gpuOnly(gpu);

  cpuOnly.setHandler(cpu);
  gpuOnly.setHandler(gpu);
  io.setHandler(cpu, gpu);

  int srcDims[VIEW::MAX_RANK] = {8, 0, 0, 0};
  int dstDims[VIEW::MAX_RANK] = {96, 0, 0, 0};

  VIEW::Math hostSrc;
  VIEW::Math hostDst;
  VIEW::Math deviceSrc;
  VIEW::Math deviceDst;
  VIEW::Math sharedSrc;
  VIEW::Math sharedDst;

  hostSrc.getLayout().setShape(srcDims, 1, VIEW::FLOAT);
  hostDst.getLayout().setShape(dstDims, 1, VIEW::FLOAT);
  deviceSrc.getLayout().setShape(srcDims, 1, VIEW::FLOAT);
  deviceDst.getLayout().setShape(dstDims, 1, VIEW::FLOAT);
  sharedSrc.getLayout().setShape(srcDims, 1, VIEW::FLOAT);
  sharedDst.getLayout().setShape(dstDims, 1, VIEW::FLOAT);

  float hostInput[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  float hostOutput[8] = {0.0f};
  float hostDeviceOutput[8] = {0.0f};

  void *deviceInput = NULL;
  void *deviceOutput = NULL;
  cudaMalloc(&deviceInput, sizeof(hostInput));
  cudaMalloc(&deviceOutput, sizeof(hostInput));
  cudaMemcpy(deviceInput, hostInput, sizeof(hostInput), cudaMemcpyHostToDevice);

  bindCpuForTest(cpu, hostSrc);
  bindCpuForTest(cpu, hostDst);
  bindGpuForTest(gpu, deviceSrc);
  bindGpuForTest(gpu, deviceDst);
  bindForTest(cpu, gpu, sharedSrc);
  bindForTest(cpu, gpu, sharedDst);
  logStep(io, "bind cpu, gpu, and shared math");

  io.copyHostToHost(hostSrc, hostInput, 8, VIEW::FLOAT);
  io.copyHostToHost(hostDst, hostSrc);
  io.copyHostToHost(hostOutput, hostDst, 8, VIEW::FLOAT);
  logStep(io, "host to host overloads");
  io.printData(hostDst, __FILE__, __LINE__);

  io.copyHostToHost(sharedSrc, hostInput, 8, VIEW::FLOAT);
  io.copyHostToDevice(sharedSrc);
  io.copyDeviceToHost(sharedSrc);
  logStep(io, "same math host to device and device to host");

  io.copyHostToDevice(deviceDst, hostSrc);
  io.copyHostToDevice(deviceSrc, hostInput, 8, VIEW::FLOAT);
  io.copyHostToDevice(deviceOutput, hostSrc, 8, VIEW::FLOAT);
  logStep(io, "host to device overloads");

  io.copyDeviceToHost(hostDst, deviceSrc);
  io.copyDeviceToHost(hostDst, deviceInput, 8, VIEW::FLOAT);
  io.copyDeviceToHost(hostDeviceOutput, deviceDst, 8, VIEW::FLOAT);
  logStep(io, "device to host overloads");
  io.printData(hostDst, __FILE__, __LINE__);

  io.copyDeviceToDevice(deviceDst, deviceSrc);
  io.copyDeviceToDevice(deviceDst, deviceInput, 8, VIEW::FLOAT);
  io.copyDeviceToDevice(deviceOutput, deviceSrc, 8, VIEW::FLOAT);
  logStep(io, "device to device overloads");

  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  return 0;
}
