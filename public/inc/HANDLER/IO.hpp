#if !defined(IO_CUDA_HPP)
#define IO_CUDA_HPP

#include "VIEW/Math.hpp"

namespace HANDLER
{

class __align__(CORE::ALIGNE_TO_256) IO
{
private:
  HANDLER::Cpu *handleCpu = NULL;
  HANDLER::Cuda *handleGpu = NULL;
public:
  IO(HANDLER::Cpu& handleCpu, HANDLER::Cuda& HandleGpu);
  IO();
  ~IO();

  void setHandler(HANDLER::Cpu& handleCpu, HANDLER::Cuda& HandleGpu);

  void setHandler(HANDLER::Cpu& handleCpu);

  void setHandler(HANDLER::Cuda& HandleGpu);

  CORE::State writeIO(const char *path, const VIEW::Math& src);

  CORE::State readIO(VIEW::Math& dst, const char *path);

  CORE::State readCsv(VIEW::Math& filepath, VIEW::Math& label, const char *path);

  CORE::State readImage(VIEW::Math& dst, const char *path);

  void copyToHost(VIEW::Math& dstCpu, VIEW::Math& srcGpu);

  void copyToDevice(VIEW::Math& dstGpu, VIEW::Math& srcCpu);

  void copyHandlerToHost();

  void copyHandlerToDevice();
};

}

#endif /* IO_CUDA_HPP */
