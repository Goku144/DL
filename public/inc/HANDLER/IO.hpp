#if !defined(IO_CUDA_HPP)
#define IO_CUDA_HPP

#include "VIEW/Math.hpp"

namespace HANDLER
{

#define IO_DEFAULT_BUFFER_SIZE 4096

class __align__(CORE::ALIGNE_TO_256) IO
{
private:
  HANDLER::Cpu *handleCpu = NULL;
  HANDLER::Cuda *handleGpu = NULL;
  CORE::errIO err = CORE::ioSuccess;

public:
  IO(HANDLER::Cpu& handleCpu, HANDLER::Cuda& handleGpu);
  IO(HANDLER::Cpu& handleCpu);
  IO(HANDLER::Cuda& handleCpu);
  ~IO();

  CORE::errIO getErr();
  CORE::errIO peekErr() const;
  void clearErr();

  void setHandler(HANDLER::Cpu& handleCpu, HANDLER::Cuda& handleGpu);

  void setHandler(HANDLER::Cpu& handleCpu);

  void setHandler(HANDLER::Cuda& HandleGpu);

  void bindCpu(VIEW::Math& math);

  void bindGpu(VIEW::Math& math);

  void bind(VIEW::Math& math);

  void unbind(VIEW::Math& math);

  // DONT FORGET DATA TYPE
  void copyHostToHost(VIEW::Math& dstMath, VIEW::Math& srcMath);

  void copyHostToHost(VIEW::Math& dstMath, void *src, size_t n, VIEW::DType dtype = VIEW::CHAR);

  void copyHostToHost(void *dst, VIEW::Math& srcMath, size_t n, VIEW::DType dtype = VIEW::CHAR);

  void copyHostToDevice(VIEW::Math& math);

  void copyHostToDevice(VIEW::Math& dstMath, VIEW::Math& srcMath);

  void copyHostToDevice(VIEW::Math& dstMath, void *src, size_t n, VIEW::DType dtype = VIEW::CHAR);
  
  void copyHostToDevice(void *dst, VIEW::Math& srcMath, size_t n, VIEW::DType dtype = VIEW::CHAR);

  void copyDeviceToHost(VIEW::Math& math);

  void copyDeviceToHost(VIEW::Math& dstMath, VIEW::Math& srcMath);

  void copyDeviceToHost(VIEW::Math& dstMath, void *src, size_t n, VIEW::DType dtype = VIEW::CHAR);

  void copyDeviceToHost(void *dst, VIEW::Math& srcMath, size_t n, VIEW::DType dtype = VIEW::CHAR);

  void copyHalfToCpuFloat(VIEW::Math& dstMath, VIEW::Math& srcMath);

  void copyDeviceToDevice(VIEW::Math& dstMath, VIEW::Math& srcMath);

  void copyDeviceToDevice(VIEW::Math& dstMath, void *src, size_t n, VIEW::DType dtype = VIEW::CHAR);

  void copyDeviceToDevice(void *dst, VIEW::Math& srcMath, size_t n, VIEW::DType dtype = VIEW::CHAR);

  void printData(VIEW::Math& math, const char* file = __FILE__, int line = __LINE__) const;

  void info(CORE::State level = CORE::WARN, const char *file = __FILE__, int line = __LINE__) const;
};

}

#endif /* IO_CUDA_HPP */
