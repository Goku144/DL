#if !defined(HANDLER_WORKSPACE_HPP)
#define HANDLER_WORKSPACE_HPP

#include "HANDLER/File.hpp"

#include <cudnn.h>
#include <cublasLt.h>

namespace HANDLER
{

class Workspace
{
private:
  HANDLER::IO *io = NULL;
  HANDLER::File *file = NULL;
  CORE::errWorkspace err = CORE::workspaceSuccess;

  cudnnHandle_t cudnnHandle = NULL;
  cublasLtHandle_t cublasLtHandle = NULL;

  void *scratchPtr = NULL;
  size_t scratchBytes = 0;

public:
  Workspace(HANDLER::IO& io, HANDLER::File& file, size_t scratchBytes);
  ~Workspace();

  HANDLER::IO& getIO();
  HANDLER::File& getFile();

  cudnnHandle_t getCudnnHandle();
  cublasLtHandle_t getCublasLtHandle();

  void *getScratch(size_t requiredBytes);
  size_t getScratchBytes() const;

  CORE::errWorkspace getErr();
  CORE::errWorkspace peekErr() const;
  void clearErr();

  void info(CORE::State level = CORE::WARN, const char *file = __FILE__, int line = __LINE__) const;
};

}

#endif /* HANDLER_WORKSPACE_HPP */
