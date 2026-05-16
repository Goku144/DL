#if !defined(VIEW_MATH_HPP)
#define VIEW_MATH_HPP

#include "HANDLER/Cpu.hpp"
#include "HANDLER/Cuda.hpp"

#include "VIEW/Shape.hpp"

namespace VIEW
{

class Shape;

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
  Math(VIEW::Shape& layout);
  Math();
  ~Math();

  void *getCpuPtr() const;
  void *getGpuPtr() const;
  size_t getCpuOffset() const;
  size_t getGpuOffset() const;
  size_t getBytes() const;
  size_t getCount() const;
  VIEW::Shape& getLayout();

  void setCpuPtr(void *cpuPtr);
  void setGpuPtr(void *gpuPtr);
  void setCpuOffset(size_t cpuOffset);
  void setGpuOffset(size_t gpuOffset);
  void setBytes(size_t bytes);
  void setCount(size_t count);
  void setLayout(VIEW::Shape& layout);

  void *getCpuDataAt(int i = 0, int j = 0, int k = 0, int l = 0);
  void *getGpuDataAt(int i = 0, int j = 0, int k = 0, int l = 0);

  void info(const char* file = __FILE__, int line = __LINE__) const;
};
  
}

#endif /* VIEW_MATH_HPP */
