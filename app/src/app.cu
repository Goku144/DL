#include "HANDLER/Workspace.hpp"

#include <cuda_runtime_api.h>
#include <stdio.h>

static bool check(bool condition, const char *message)
{
  if(condition)
  {
    printf("[PASS] %s\n", message);
    return true;
  }

  printf("[FAIL] %s\n", message);
  return false;
}

static bool checkWorkspaceSuccess(HANDLER::Workspace& workspace, const char *message)
{
  if(workspace.peekErr() == CORE::workspaceSuccess)
  {
    printf("[PASS] %s\n", message);
    return true;
  }

  printf("[FAIL] %s\n", message);
  workspace.info(CORE::WARN);
  return false;
}

static bool checkFileSuccess(HANDLER::File& file, const char *message)
{
  if(file.peekErr() == CORE::fileSuccess)
  {
    printf("[PASS] %s\n", message);
    return true;
  }

  printf("[FAIL] %s\n", message);
  file.info(CORE::WARN);
  return false;
}

static bool checkIOSuccess(HANDLER::IO& io, const char *message)
{
  if(io.peekErr() == CORE::ioSuccess)
  {
    printf("[PASS] %s\n", message);
    return true;
  }

  printf("[FAIL] %s\n", message);
  io.info(CORE::WARN);
  return false;
}

int main()
{
  HANDLER::Cpu cpu(CORE::MEMORY_32_MB);
  HANDLER::Cuda cuda(CORE::MEMORY_32_MB);
  HANDLER::IO io(cpu, cuda);
  HANDLER::File file(io);

  const size_t scratchBytes = 4096;
  HANDLER::Workspace workspace(io, file, scratchBytes);
  bool ok = true;

  printf("Workspace test start\n");

  ok &= checkWorkspaceSuccess(workspace, "workspace constructor");
  ok &= check(&workspace.getIO() == &io, "workspace returns original IO handler");
  ok &= check(&workspace.getFile() == &file, "workspace returns original File handler");
  ok &= check(workspace.getCudnnHandle() != NULL, "workspace owns a cuDNN handle");
  ok &= check(workspace.getCublasLtHandle() != NULL, "workspace owns a cuBLASLt handle");
  ok &= check(workspace.getScratchBytes() == scratchBytes, "workspace records scratchpad size");

  void *zeroScratch = workspace.getScratch(0);
  ok &= check(zeroScratch == NULL, "zero-byte scratch request returns NULL");
  ok &= checkWorkspaceSuccess(workspace, "zero-byte scratch request keeps success state");

  void *scratchSmall = workspace.getScratch(1);
  ok &= check(scratchSmall != NULL, "one-byte scratch request returns scratchpad");
  ok &= checkWorkspaceSuccess(workspace, "one-byte scratch request keeps success state");

  void *scratchFull = workspace.getScratch(scratchBytes);
  ok &= check(scratchFull == scratchSmall, "full scratch request reuses same scratchpad");
  ok &= checkWorkspaceSuccess(workspace, "full scratch request keeps success state");

  if(scratchFull != NULL)
  {
    cudaError_t memsetErr = cudaMemset(scratchFull, 0xA5, scratchBytes);
    ok &= check(memsetErr == cudaSuccess, "scratchpad accepts cudaMemset");

    cudaError_t syncErr = cudaDeviceSynchronize();
    ok &= check(syncErr == cudaSuccess, "scratchpad memset synchronizes cleanly");
  }

  void *tooLarge = workspace.getScratch(scratchBytes + 1);
  ok &= check(tooLarge == NULL, "oversized scratch request returns NULL");
  ok &= check(workspace.peekErr() == CORE::workspaceErrScratchOutOfBound,
              "oversized scratch request sets out-of-bound error");
  ok &= check(workspace.getErr() == CORE::workspaceErrScratchOutOfBound,
              "getErr returns and clears workspace error");
  ok &= check(workspace.peekErr() == CORE::workspaceSuccess,
              "workspace error is clear after getErr");

  printf("Image uint8 to half GPU test start\n");

  VIEW::Math imagePaths;
  VIEW::Math labels;
  VIEW::Math gpuImages;
  VIEW::Math cpuFloatImages;

  file.readCsv(imagePaths, labels, "public/target/meta/test.csv");
  ok &= checkFileSuccess(file, "image csv loaded");

  file.readImages(imagePaths, "public/target/meta");
  ok &= checkFileSuccess(file, "uint8 images loaded on CPU");

  file.copyImageToDevice();
  ok &= checkFileSuccess(file, "uint8 images converted and copied to GPU half");

  size_t batch = file.getImageN() < 8 ? file.getImageN() : 8;
  if(batch > 0)
  {
    file.pullGpuImage(gpuImages, batch);
    ok &= checkFileSuccess(file, "pulled GPU image batch");

    VIEW::Shape& layout = gpuImages.getLayout();
    size_t expectedCount = batch * file.getImageChannel() * file.getImageHeight() * file.getImageWidth();
    size_t expectedBytes = CORE::ALIGNE(expectedCount * VIEW::F16, CORE::ALIGNE_TO_256);

    ok &= check(layout.getRank() == 4, "pulled GPU image rank is NCHW");
    ok &= check(layout.getDType() == VIEW::F16, "pulled GPU image dtype is F16");
    ok &= check(layout.getDim(0) == (int)batch, "pulled GPU image batch dim is correct");
    ok &= check(layout.getDim(1) == file.getImageChannel(), "pulled GPU image channel dim is correct");
    ok &= check(layout.getDim(2) == file.getImageHeight(), "pulled GPU image height dim is correct");
    ok &= check(layout.getDim(3) == file.getImageWidth(), "pulled GPU image width dim is correct");
    ok &= check(gpuImages.getCount() == expectedCount, "pulled GPU image element count is correct");
    ok &= check(gpuImages.getBytes() == expectedBytes, "pulled GPU image uses half-sized storage");
    ok &= check(gpuImages.getGpuPtr() != NULL, "pulled GPU image has device pointer");

    io.copyHalfToCpuFloat(cpuFloatImages, gpuImages);
    ok &= checkIOSuccess(io, "copied GPU half image batch to CPU float");

    VIEW::Shape& cpuFloatLayout = cpuFloatImages.getLayout();
    size_t expectedFloatBytes = CORE::ALIGNE(expectedCount * VIEW::FLOAT, CORE::ALIGNE_TO_256);

    ok &= check(cpuFloatLayout.getRank() == 4, "CPU float image rank is NCHW");
    ok &= check(cpuFloatLayout.getDType() == VIEW::FLOAT, "CPU float image dtype is FLOAT");
    ok &= check(cpuFloatLayout.getDim(0) == (int)batch, "CPU float image batch dim is correct");
    ok &= check(cpuFloatLayout.getDim(1) == file.getImageChannel(), "CPU float image channel dim is correct");
    ok &= check(cpuFloatLayout.getDim(2) == file.getImageHeight(), "CPU float image height dim is correct");
    ok &= check(cpuFloatLayout.getDim(3) == file.getImageWidth(), "CPU float image width dim is correct");
    ok &= check(cpuFloatImages.getCount() == expectedCount, "CPU float image element count is correct");
    ok &= check(cpuFloatImages.getBytes() == expectedFloatBytes, "CPU float image uses float-sized storage");
    ok &= check(cpuFloatImages.getCpuPtr() != NULL, "CPU float image has host pointer");
  }
  else
  {
    ok &= check(false, "dataset has at least one image for GPU half pull test");
  }

  printf("App tests %s\n", ok ? "passed" : "failed");

  return ok ? 0 : 1;
}
