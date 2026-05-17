#if !defined(HANDLER_CPU_HPP)
#define HANDLER_CPU_HPP

#include "CORE/State.hpp"

namespace HANDLER
{

/**
 * @brief Linear CPU memory arena used by IO::bindCpu.
 *
 * Cpu owns one aligned host allocation and hands out aligned slices to
 * VIEW::Math tensors. Individual tensor slices are not freed; call reset() to
 * reuse the arena from the beginning.
 */
class __align__(CORE::ALIGNE_TO_256) Cpu
{
private:
  void *data = NULL;
  size_t offset = 0;
  size_t capacity = 0;

public:
  /** @brief Allocate the CPU arena. 
   * @param capacity Arena size in bytes. 
   * @param file Source file for fatal allocation logs. 
   * @param line Source line for fatal allocation logs. 
   * */
  Cpu(size_t capacity = CORE::MEMORY_1_GB, const char* file = __FILE__, int line = __LINE__);

  /** @brief Free the CPU arena. */
  ~Cpu();

  /** @brief Get base arena pointer. 
   * @return Base CPU memory pointer.
   * */
  void *getData();

  /** @brief Get next free arena offset. 
   * @return Offset in bytes. 
   * */
  size_t getOffset();

  /** @brief Get arena capacity. 
   * @return Capacity in bytes. 
   * */
  size_t getCapacity();

  /** @brief Allocate a slice from the arena. 
   * @param cpuPtr Receives slice pointer. 
   * @param offset Receives slice offset. 
   * @param capacity Requested bytes. 
   * @return ioSuccess, ioErrNull, or ioErrOutOfBound. 
   * */
  CORE::errIO allocate(void **cpuPtr, size_t& offset, size_t capacity);

  /** @brief Reset the next allocation offset to zero. */
  void reset();
};

}

#endif /* HANDLER_CPU_HPP */
