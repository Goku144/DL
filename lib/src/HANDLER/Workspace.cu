#include "HANDLER/Workspace.hpp"

#include <cuda_runtime_api.h>

static const char *getWorkspaceErrorMessage(CORE::errWorkspace err)
{
  if(err == CORE::workspaceSuccess) return "Workspace success";
  if(err == CORE::workspaceErrCudnnCreate) return "Workspace failed to create cuDNN handle";
  if(err == CORE::workspaceErrCudnnDestroy) return "Workspace failed to destroy cuDNN handle";
  if(err == CORE::workspaceErrCublasLtCreate) return "Workspace failed to create cuBLASLt handle";
  if(err == CORE::workspaceErrCublasLtDestroy) return "Workspace failed to destroy cuBLASLt handle";
  if(err == CORE::workspaceErrScratchAlloc) return "Workspace failed to allocate scratch memory";
  if(err == CORE::workspaceErrScratchFree) return "Workspace failed to free scratch memory";
  if(err == CORE::workspaceErrScratchOutOfBound) return "Workspace scratch memory is too small";
  if(err == CORE::workspaceErrNull) return "Workspace null pointer";
  return "Workspace unknown error";
}

HANDLER::Workspace::Workspace(HANDLER::IO& io, HANDLER::File& file, size_t scratchBytes)
{
  this->io = &io;
  this->file = &file;
  file.setIO(io);

  if(cudnnCreate(&this->cudnnHandle) != CUDNN_STATUS_SUCCESS)
  {
    this->err = CORE::workspaceErrCudnnCreate;
    return;
  }

  if(cublasLtCreate(&this->cublasLtHandle) != CUBLAS_STATUS_SUCCESS)
  {
    this->err = CORE::workspaceErrCublasLtCreate;
    return;
  }

  this->scratchBytes = scratchBytes;
  if(scratchBytes > 0 && cudaMalloc(&this->scratchPtr, scratchBytes) != cudaSuccess)
  {
    this->scratchBytes = 0;
    this->err = CORE::workspaceErrScratchAlloc;
  }
}

HANDLER::Workspace::~Workspace()
{
  if(this->scratchPtr != NULL && cudaFree(this->scratchPtr) != cudaSuccess)
  {
    this->err = CORE::workspaceErrScratchFree;
    CORE::logWarn(__FILE__, __LINE__, "%s", getWorkspaceErrorMessage(this->err));
  }

  if(this->cublasLtHandle != NULL && cublasLtDestroy(this->cublasLtHandle) != CUBLAS_STATUS_SUCCESS)
  {
    this->err = CORE::workspaceErrCublasLtDestroy;
    CORE::logWarn(__FILE__, __LINE__, "%s", getWorkspaceErrorMessage(this->err));
  }

  if(this->cudnnHandle != NULL && cudnnDestroy(this->cudnnHandle) != CUDNN_STATUS_SUCCESS)
  {
    this->err = CORE::workspaceErrCudnnDestroy;
    CORE::logWarn(__FILE__, __LINE__, "%s", getWorkspaceErrorMessage(this->err));
  }
}

HANDLER::IO& HANDLER::Workspace::getIO()
{
  return *this->io;
}

HANDLER::File& HANDLER::Workspace::getFile()
{
  return *this->file;
}

cudnnHandle_t HANDLER::Workspace::getCudnnHandle()
{
  return this->cudnnHandle;
}

cublasLtHandle_t HANDLER::Workspace::getCublasLtHandle()
{
  return this->cublasLtHandle;
}

void *HANDLER::Workspace::getScratch(size_t requiredBytes)
{
  if(requiredBytes == 0) return NULL;

  if(this->scratchPtr == NULL)
  {
    this->err = CORE::workspaceErrNull;
    return NULL;
  }

  if(requiredBytes > this->scratchBytes)
  {
    this->err = CORE::workspaceErrScratchOutOfBound;
    return NULL;
  }

  return this->scratchPtr;
}

size_t HANDLER::Workspace::getScratchBytes() const
{
  return this->scratchBytes;
}

CORE::errWorkspace HANDLER::Workspace::getErr()
{
  CORE::errWorkspace err = this->err;
  this->err = CORE::workspaceSuccess;
  return err;
}

CORE::errWorkspace HANDLER::Workspace::peekErr() const
{
  return this->err;
}

void HANDLER::Workspace::clearErr()
{
  this->err = CORE::workspaceSuccess;
}

void HANDLER::Workspace::info(CORE::State level, const char *file, int line) const
{
  if(this->err == CORE::workspaceSuccess) return;

  if(level == CORE::FATAL)
  {
    CORE::logFatal(file, line, "%s", getWorkspaceErrorMessage(this->err));
    return;
  }

  CORE::logWarn(file, line, "%s", getWorkspaceErrorMessage(this->err));
}
