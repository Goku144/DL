#include "VIEW/Shape.hpp"
#include "VIEW/Math.hpp"

VIEW::Math::Math(HANDLER::Cpu& handler, VIEW::Shape& layout)
{
  this->count = layout.getDim(0) * layout.getStride(0);
  size_t bytes = this->count * layout.getDType();
  bytes = CORE::ALIGNE(bytes, CORE::ALIGNE_TO_256);
  handler.allocate(this->cpuOffset, bytes);
  this->bytes = bytes;
  this->layout = &layout;
}

VIEW::Math::Math(HANDLER::Cuda& handler, VIEW::Shape& layout)
{
  this->count = layout.getDim(0) * layout.getStride(0);
  size_t bytes = this->count * layout.getDType();
  bytes = CORE::ALIGNE(bytes, CORE::ALIGNE_TO_256);
  handler.allocate(this->gpuOffset, bytes);
  this->bytes = bytes;
  this->layout = &layout;
}

VIEW::Math::~Math()
{}

void VIEW::Math::bind(HANDLER::Cpu& handler, VIEW::Shape& layout)
{
  this->count = layout.getDim(0) * layout.getStride(0);
  size_t bytes = this->count * layout.getDType();
  bytes = CORE::ALIGNE(bytes, CORE::ALIGNE_TO_256);
  handler.allocate(this->cpuOffset, bytes);
  this->bytes = bytes;
  this->layout = &layout;
}

void VIEW::Math::bind(HANDLER::Cuda& handler, VIEW::Shape& layout)
{
  this->count = layout.getDim(0) * layout.getStride(0);
  size_t bytes = this->count * layout.getDType();
  bytes = CORE::ALIGNE(bytes, CORE::ALIGNE_TO_256);
  handler.allocate(this->gpuOffset, bytes);
  this->bytes = bytes;
  this->layout = &layout;
}

void VIEW::Math::unbind()
{
  this->cpuOffset = 0;
  this->gpuOffset = 0;
  this->bytes = 0;
  this->count = 0;
  this->layout = NULL;
}