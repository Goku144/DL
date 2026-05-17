#if !defined(HANDLER_WORKSPACE_HPP)
#define HANDLER_WORKSPACE_HPP

#include "HANDLER/File.hpp"

#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <cublasLt.h>

namespace HANDLER
{

/**
 * @brief Shared execution context for GPU operators.
 *
 * Workspace owns a CUDA stream, cuDNN handle, cuBLASLt handle, and scratch
 * device buffer. Operators receive a Workspace reference so they share the same
 * stream and temporary memory.
 */
class Workspace
{
private:
  HANDLER::IO *io = NULL;
  HANDLER::File *file = NULL;
  CORE::errWorkspace err = CORE::workspaceSuccess;

  cudnnHandle_t cudnnHandle = NULL;
  cublasLtHandle_t cublasLtHandle = NULL;
  cudaStream_t stream = NULL;

  void *scratchPtr = NULL;
  size_t scratchBytes = 0;

public:
  /** @brief Create handles, stream, and scratch storage. 
   * @param io IO handler used by the project. 
   * @param file File handler connected to the IO handler. 
   * @param scratchBytes Scratch buffer size in bytes. 
   * 
   * @note Sets workspaceErrStreamCreate, workspaceErrCudnnCreate, 
   * workspaceErrCublasLtCreate, workspaceErrCudnnSetStream, 
   * or workspaceErrScratchAlloc on failure. 
   * */
  Workspace(HANDLER::IO& io, HANDLER::File& file, size_t scratchBytes);

  /** @brief Destroy scratch storage, handles, and stream. 
   * 
   * @note Logs warnings if destruction fails. 
   * */
  ~Workspace();

  /** @brief Get IO handler. 
   * @return Reference to IO handler. 
   * */
  HANDLER::IO& getIO();

  /** @brief Get File handler. 
   * @return Reference to File handler. 
   * */
  HANDLER::File& getFile();

  /** @brief Get cuDNN handle. 
   * @return cuDNN handle. 
   * */
  cudnnHandle_t getCudnnHandle();

  /** @brief Get cuBLASLt handle. 
   * @return cuBLASLt handle. 
   * */
  cublasLtHandle_t getCublasLtHandle();

  /** @brief Get CUDA stream. 
   * @return CUDA stream used by operators. 
   * */
  cudaStream_t getStream();

  /** @brief Get scratch buffer. 
   * @param requiredBytes Required bytes. 
   * @return Scratch pointer, or NULL for zero/failed requests. 
   * 
   * @note Sets workspaceErrNull or workspaceErrScratchOutOfBound on invalid requests. 
   * */
  void *getScratch(size_t requiredBytes);

  /** @brief Get scratch capacity. 
   * @return Scratch size in bytes. 
   * */
  size_t getScratchBytes() const;

  /** @brief Get and clear workspace error. 
   * @return Current error before clearing. 
   * */
  CORE::errWorkspace getErr();

  /** @brief Read workspace error without clearing. 
   * @return Current workspace error. 
   * */
  CORE::errWorkspace peekErr() const;

  /** @brief Clear workspace error state. */
  void clearErr();

  /** @brief Print current workspace error. 
   * @param level WARN or FATAL logging level. 
   * @param file Source file for log. 
   * @param line Source line for log. 
   * */
  void info(CORE::State level = CORE::WARN, const char *file = __FILE__, int line = __LINE__) const;
};

}

#endif /* HANDLER_WORKSPACE_HPP */
