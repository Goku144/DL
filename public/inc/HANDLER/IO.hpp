#if !defined(FILE_CUDA_HPP)
#define FILE_CUDA_HPP

#include "CORE/State.hpp"

namespace VIEW
{
  class Math;
}

namespace HANDLER
{

class __align__(CORE::ALIGNE_TO_256) File
{
private:
  /* var */
public:
  File(/* args */);
  ~File();

  CORE::State writeFile(const char *path, const VIEW::Math& src);

  CORE::State readFile(VIEW::Math& dst, const char *path);

  CORE::State readCsv(VIEW::Math& filepath, VIEW::Math& label, const char *path);

  CORE::State readImage(VIEW::Math& dst, const char *path);
};
  
}

#endif /* FILE_CUDA_HPP */
