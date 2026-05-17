#if !defined(IO_CUDA_HPP)
#define IO_CUDA_HPP

#include "VIEW/Math.hpp"

namespace HANDLER
{

#define IO_DEFAULT_BUFFER_SIZE 4096

/**
 * @brief Memory binding and transfer helper for VIEW::Math tensors.
 *
 * IO connects VIEW::Math metadata to Cpu/Cuda arenas. It allocates CPU/GPU
 * slices, validates that tensor pointers still belong to the active handlers,
 * copies data between host and device, and stores the last IO error.
 */
class __align__(CORE::ALIGNE_TO_256) IO
{
private:
  HANDLER::Cpu *handleCpu = NULL;
  HANDLER::Cuda *handleGpu = NULL;
  CORE::errIO err = CORE::ioSuccess;

public:
  /** @brief Create IO with CPU and GPU handlers. @param handleCpu CPU arena. @param handleGpu GPU arena. */
  IO(HANDLER::Cpu& handleCpu, HANDLER::Cuda& handleGpu);

  /** @brief Create CPU-only IO. @param handleCpu CPU arena. */
  IO(HANDLER::Cpu& handleCpu);

  /** @brief Create GPU-only IO. @param handleCpu GPU arena parameter name kept for ABI compatibility. */
  IO(HANDLER::Cuda& handleCpu);

  /** @brief Destroy IO. Does not own handler memory. */
  ~IO();

  /** @brief Get and clear last IO error. @return Current error before clearing. */
  CORE::errIO getErr();

  /** @brief Read last IO error without clearing. @return Current IO error. */
  CORE::errIO peekErr() const;

  /** @brief Clear last IO error. */
  void clearErr();

  /** @brief Replace CPU and GPU handlers. @param handleCpu CPU arena. @param handleGpu GPU arena. */
  void setHandler(HANDLER::Cpu& handleCpu, HANDLER::Cuda& handleGpu);

  /** @brief Replace CPU handler. @param handleCpu CPU arena. */
  void setHandler(HANDLER::Cpu& handleCpu);

  /** @brief Replace GPU handler. @param HandleGpu GPU arena. */
  void setHandler(HANDLER::Cuda& HandleGpu);

  /** @brief Allocate CPU memory for a tensor. @param math Tensor with shape already set. @note Sets ioErrOutOfBound or ioErrNull if allocation fails. */
  void bindCpu(VIEW::Math& math);

  /** @brief Allocate GPU memory for a tensor. @param math Tensor with shape already set. @note Sets ioErrOutOfBound or ioErrNull if allocation fails. */
  void bindGpu(VIEW::Math& math);

  /** @brief Allocate both CPU and GPU memory for a tensor. @param math Tensor with shape already set. @note Propagates bindCpu/bindGpu errors. */
  void bind(VIEW::Math& math);

  /** @brief Clear tensor pointers, offsets, size, count, and shape. @param math Tensor to clear. */
  void unbind(VIEW::Math& math);

  /** @brief Copy CPU tensor data to another CPU tensor and copy layout. @param dstMath Destination CPU tensor. @param srcMath Source CPU tensor. @note Sets ioErrInvalidState or ioErrOutOfBound on invalid operands. */
  void copyHostToHost(VIEW::Math& dstMath, VIEW::Math& srcMath);

  /** @brief Copy raw CPU data into a CPU tensor without changing layout. @param dstMath Destination CPU tensor. @param src Source CPU pointer. @param n Element count. @param dtype Element type. @note Sets ioErrNull, ioErrInvalidState, ioErrOutOfBound, or ioErrCopyToHost. */
  void copyHostToHost(VIEW::Math& dstMath, void *src, size_t n, VIEW::DType dtype = VIEW::CHAR);

  /** @brief Copy CPU tensor data into a raw CPU pointer. @param dst Destination CPU pointer. @param srcMath Source CPU tensor. @param n Element count. @param dtype Element type. @note Sets ioErrNull, ioErrInvalidState, ioErrOutOfBound, or ioErrCopyToHost. */
  void copyHostToHost(void *dst, VIEW::Math& srcMath, size_t n, VIEW::DType dtype = VIEW::CHAR);

  /** @brief Copy tensor CPU memory to its GPU memory. @param math Tensor bound on CPU and GPU. @note Sets ioErrInvalidState or ioErrCopyToDevice. */
  void copyHostToDevice(VIEW::Math& math);

  /** @brief Copy CPU data from one tensor into GPU data of another and copy layout. @param dstMath Destination GPU tensor. @param srcMath Source CPU tensor. @note Sets ioErrInvalidState, ioErrOutOfBound, or copy errors. */
  void copyHostToDevice(VIEW::Math& dstMath, VIEW::Math& srcMath);

  /** @brief Copy raw CPU data into GPU tensor memory without changing layout. @param dstMath Destination GPU tensor. @param src Source CPU pointer. @param n Element count. @param dtype Element type. @note Sets ioErrNull, ioErrInvalidState, ioErrOutOfBound, or ioErrCopyToHost. */
  void copyHostToDevice(VIEW::Math& dstMath, void *src, size_t n, VIEW::DType dtype = VIEW::CHAR);
  
  /** @brief Copy CPU tensor data into a raw GPU pointer. @param dst Destination GPU pointer. @param srcMath Source CPU tensor. @param n Element count. @param dtype Element type. @note Sets ioErrNull, ioErrInvalidState, ioErrOutOfBound, or ioErrCopyToHost. */
  void copyHostToDevice(void *dst, VIEW::Math& srcMath, size_t n, VIEW::DType dtype = VIEW::CHAR);

  /** @brief Copy tensor GPU memory to its CPU memory. @param math Tensor bound on CPU and GPU. @note Sets ioErrInvalidState or ioErrCopyToHost. */
  void copyDeviceToHost(VIEW::Math& math);

  /** @brief Copy GPU data from one tensor into CPU data of another and copy layout. @param dstMath Destination CPU tensor. @param srcMath Source GPU tensor. @note Sets ioErrInvalidState, ioErrOutOfBound, or ioErrCopyToHost. */
  void copyDeviceToHost(VIEW::Math& dstMath, VIEW::Math& srcMath);

  /** @brief Copy raw GPU data into CPU tensor memory without changing layout. @param dstMath Destination CPU tensor. @param src Source GPU pointer. @param n Element count. @param dtype Element type. @note Sets ioErrNull, ioErrInvalidState, ioErrOutOfBound, or ioErrCopyToHost. */
  void copyDeviceToHost(VIEW::Math& dstMath, void *src, size_t n, VIEW::DType dtype = VIEW::CHAR);

  /** @brief Copy GPU tensor data into a raw CPU pointer. @param dst Destination CPU pointer. @param srcMath Source GPU tensor. @param n Element count. @param dtype Element type. @note Sets ioErrNull, ioErrInvalidState, ioErrOutOfBound, or ioErrCopyToHost. */
  void copyDeviceToHost(void *dst, VIEW::Math& srcMath, size_t n, VIEW::DType dtype = VIEW::CHAR);

  /** @brief Convert an F16 GPU tensor into a FLOAT CPU tensor. @param dstMath Destination CPU float tensor, allocated by this call. @param srcMath Source GPU F16 tensor. @note Sets ioErrInvalidState, ioErrInvalidValue, ioErrOutOfMemory, or ioErrCopyToHost. */
  void copyHalfToCpuFloat(VIEW::Math& dstMath, VIEW::Math& srcMath);

  /** @brief Copy GPU tensor data to another GPU tensor and copy layout. @param dstMath Destination GPU tensor. @param srcMath Source GPU tensor. @note Sets ioErrInvalidState, ioErrOutOfBound, or copy errors. */
  void copyDeviceToDevice(VIEW::Math& dstMath, VIEW::Math& srcMath);

  /** @brief Copy raw GPU data into GPU tensor memory without changing layout. @param dstMath Destination GPU tensor. @param src Source GPU pointer. @param n Element count. @param dtype Element type. @note Sets ioErrNull, ioErrInvalidState, ioErrOutOfBound, or ioErrCopyToHost. */
  void copyDeviceToDevice(VIEW::Math& dstMath, void *src, size_t n, VIEW::DType dtype = VIEW::CHAR);

  /** @brief Copy GPU tensor data into a raw GPU pointer. @param dst Destination GPU pointer. @param srcMath Source GPU tensor. @param n Element count. @param dtype Element type. @note Sets ioErrNull, ioErrInvalidState, ioErrOutOfBound, or ioErrCopyToHost. */
  void copyDeviceToDevice(void *dst, VIEW::Math& srcMath, size_t n, VIEW::DType dtype = VIEW::CHAR);

  /** @brief Print CPU tensor data. @param math Tensor with CPU memory. @param file Source file for log. @param line Source line for log. */
  void printData(VIEW::Math& math, const char* file = __FILE__, int line = __LINE__) const;

  /** @brief Print current IO error. @param level WARN or FATAL logging level. @param file Source file for log. @param line Source line for log. */
  void info(CORE::State level = CORE::WARN, const char *file = __FILE__, int line = __LINE__) const;
};

}

#endif /* IO_CUDA_HPP */
