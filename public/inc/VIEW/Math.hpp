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
  size_t cpuOffset = 0;
  size_t gpuOffset = 0;
  size_t bytes = 0;
  size_t count = 0;
  Shape *layout = NULL;

public:
  Math(HANDLER::Cpu& handler, VIEW::Shape& layout);
  Math(HANDLER::Cuda& handler, VIEW::Shape& layout);
  Math();
  ~Math();

  void bind(HANDLER::Cpu& handler, VIEW::Shape& layout);

  void bind(HANDLER::Cuda& handler, VIEW::Shape& layout);

  void unbind();

  void info() const;
};
  
}

#endif /* VIEW_MATH_HPP */
