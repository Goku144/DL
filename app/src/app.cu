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

  printf("Workspace test %s\n", ok ? "passed" : "failed");

  return ok ? 0 : 1;
}
