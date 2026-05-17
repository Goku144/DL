#if !defined(VIEW_MATH_HPP)
#define VIEW_MATH_HPP

#include "HANDLER/Cpu.hpp"
#include "HANDLER/Cuda.hpp"

#include "VIEW/Shape.hpp"

namespace VIEW
{

class Shape;

/**
 * @brief Tensor view that stores shape metadata and CPU/GPU memory pointers.
 *
 * Math does not allocate memory by itself. Set its Shape, then bind it through
 * HANDLER::IO to receive CPU and/or GPU pointers. Operators read/write these
 * pointers directly.
 */
class __align__(CORE::ALIGNE_TO_256) Math
{
private:
  void *cpuPtr = NULL;
  void *gpuPtr = NULL;
  size_t cpuOffset = 0;
  size_t gpuOffset = 0;
  size_t bytes = 0;
  size_t count = 0;
  Shape layout;

public:
  /** @brief Create a Math view with layout. @param layout Shape metadata copied into the view. */
  Math(VIEW::Shape& layout);

  /** @brief Create an empty Math view. */
  Math();

  /** @brief Destroy the view; memory ownership remains with handlers. */
  ~Math();

  /** @brief Get CPU pointer. @return CPU memory pointer or NULL. */
  void *getCpuPtr() const;

  /** @brief Get GPU pointer. @return GPU memory pointer or NULL. */
  void *getGpuPtr() const;

  /** @brief Get CPU arena offset. @return Offset in bytes. */
  size_t getCpuOffset() const;

  /** @brief Get GPU arena offset. @return Offset in bytes. */
  size_t getGpuOffset() const;

  /** @brief Get allocated byte size. @return Aligned byte count. */
  size_t getBytes() const;

  /** @brief Get logical element count. @return Number of elements. */
  size_t getCount() const;

  /** @brief Get mutable layout. @return Reference to Shape metadata. */
  VIEW::Shape& getLayout();

  /** @brief Set CPU pointer. @param cpuPtr CPU pointer. */
  void setCpuPtr(void *cpuPtr);

  /** @brief Set GPU pointer. @param gpuPtr GPU pointer. */
  void setGpuPtr(void *gpuPtr);

  /** @brief Set CPU arena offset. @param cpuOffset Offset in bytes. */
  void setCpuOffset(size_t cpuOffset);

  /** @brief Set GPU arena offset. @param gpuOffset Offset in bytes. */
  void setGpuOffset(size_t gpuOffset);

  /** @brief Set aligned byte size. @param bytes Bytes allocated. */
  void setBytes(size_t bytes);

  /** @brief Set logical element count. @param count Number of elements. */
  void setCount(size_t count);

  /** @brief Replace layout metadata. @param layout Shape metadata to copy. */
  void setLayout(VIEW::Shape& layout);

  /** @brief Get CPU address for an index. @param i First index. @param j Second index. @param k Third index. @param l Fourth index. @return Pointer to indexed CPU element. */
  void *getCpuDataAt(int i = 0, int j = 0, int k = 0, int l = 0);

  /** @brief Get GPU address for an index. @param i First index. @param j Second index. @param k Third index. @param l Fourth index. @return Pointer to indexed GPU element. */
  void *getGpuDataAt(int i = 0, int j = 0, int k = 0, int l = 0);

  /** @brief Print pointer, byte, count, and layout metadata. @param file Source file for log. @param line Source line for log. */
  void info(const char* file = __FILE__, int line = __LINE__) const;
};
  
}

#endif /* VIEW_MATH_HPP */
