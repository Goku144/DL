#include "VIEW/Math.hpp"

VIEW::Math::Math(HANDLER::Cpu& handler, VIEW::Shape& layout)
{
  this->count = layout.getDim(0) * layout.getStride(0);
  this->bytes = CORE::ALIGNE(this->count * layout.getDType(), CORE::ALIGNE_TO_256);
  handler.allocate(this->cpuOffset, this->bytes);
  this->layout = &layout;
}

VIEW::Math::Math(HANDLER::Cuda& handler, VIEW::Shape& layout)
{
  this->count = layout.getDim(0) * layout.getStride(0);
  this->bytes = CORE::ALIGNE(this->count * layout.getDType(), CORE::ALIGNE_TO_256);
  handler.allocate(this->gpuOffset, this->bytes);
  this->layout = &layout;
}

VIEW::Math::Math()
{}

VIEW::Math::~Math()
{}

void VIEW::Math::bind(HANDLER::Cpu& handler, VIEW::Shape& layout)
{
  this->count = layout.getDim(0) * layout.getStride(0);
  this->bytes = CORE::ALIGNE(this->count * layout.getDType(), CORE::ALIGNE_TO_256);
  handler.allocate(this->cpuOffset, this->bytes);
  this->layout = &layout;
}

void VIEW::Math::bind(HANDLER::Cuda& handler, VIEW::Shape& layout)
{
  this->count = layout.getDim(0) * layout.getStride(0);
  this->bytes = CORE::ALIGNE(this->count * layout.getDType(), CORE::ALIGNE_TO_256);
  handler.allocate(this->gpuOffset, this->bytes);
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

void VIEW::Math::info() const
{
  CORE::logInfo("Cpu Offset: 0x%x", this->cpuOffset);
  CORE::logInfo("Gpu Offset: 0x%x", this->gpuOffset);
  CORE::logInfo("bytes: %zu", this->bytes);
  CORE::logInfo("elemnts: %zu", this->count);
  this->layout->info();
}