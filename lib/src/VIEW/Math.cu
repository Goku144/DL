#include "VIEW/Math.hpp"

VIEW::Math::Math( VIEW::Shape& layout)
{
  this->layout = layout;
}

VIEW::Math::Math()
{}

VIEW::Math::~Math()
{}

void *VIEW::Math::getCpuPtr() const
{
  return this->cpuPtr;
}

void *VIEW::Math::getGpuPtr() const
{
  return this->gpuPtr;
}

size_t VIEW::Math::getCpuOffset() const
{
  return this->cpuOffset;
}

size_t VIEW::Math::getGpuOffset() const
{
  return this->gpuOffset;
}

size_t VIEW::Math::getBytes() const
{
  return this->bytes;
}
  
size_t VIEW::Math::getCount() const
{
  return this->count;
}

VIEW::Shape& VIEW::Math::getLayout() 
{
  return this->layout;
}

void VIEW::Math::setCpuOffset(size_t cpuOffset)
{
  this->cpuOffset = cpuOffset;
}

void VIEW::Math::setGpuOffset(size_t gpuOffset)
{
  this->gpuOffset = gpuOffset;
}

void VIEW::Math::setCpuPtr(void *cpuPtr)
{
  this->cpuPtr = cpuPtr;
}

void VIEW::Math::setGpuPtr(void *gpuPtr)
{
  this->gpuPtr = gpuPtr;
}

void VIEW::Math::setBytes(size_t bytes)
{
  this->bytes = bytes;
}

void VIEW::Math::setCount(size_t count)
{
  this->count = count;
}

void VIEW::Math::setLayout(VIEW::Shape& layout)
{
  this->layout = layout;
}

void *VIEW::Math::getCpuDataAt(int i, int j, int k, int l)
{
  return (uint8_t *)this->cpuPtr + this->layout.getSuperPosition(i, j, k, l);
}

void *VIEW::Math::getGpuDataAt(int i, int j, int k, int l)
{
  return (uint8_t *)this->gpuPtr + this->layout.getSuperPosition(i, j, k, l);
}

void VIEW::Math::info(const char* file, int line) const
{
  CORE::logInfo(file, line, "Cpu Offset: 0x%x", this->cpuOffset);
  CORE::logInfo(file, line, "Gpu Offset: 0x%x", this->gpuOffset);
  CORE::logInfo(file, line, "bytes: %zu", this->bytes);
  CORE::logInfo(file, line, "elemnts: %zu", this->count);
  this->layout.info(file, line);
}