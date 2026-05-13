#include "HANDLER/IO.hpp"

HANDLER::IO::IO(HANDLER::Cpu& handleCpu, HANDLER::Cuda& HandleGpu)
{

}

HANDLER::IO::IO()
{

}

HANDLER::IO::~IO()
{

}


void HANDLER::IO::setHandler(HANDLER::Cpu& handleCpu, HANDLER::Cuda& HandleGpu)
{

}


void HANDLER::IO::setHandler(HANDLER::Cpu& handleCpu)
{

}


void HANDLER::IO::setHandler(HANDLER::Cuda& HandleGpu)
{

}


CORE::State HANDLER::IO::writeIO(const char *path, const VIEW::Math& src)
{

}


CORE::State HANDLER::IO::readIO(VIEW::Math& dst, const char *path)
{

}


CORE::State HANDLER::IO::readCsv(VIEW::Math& filepath, VIEW::Math& label, const char *path)
{

}


CORE::State HANDLER::IO::readImage(VIEW::Math& dst, const char *path)
{

}


void HANDLER::IO::copyToHost(VIEW::Math& dstCpu, VIEW::Math& srcGpu)
{

}


void HANDLER::IO::copyToDevice(VIEW::Math& dstGpu, VIEW::Math& srcCpu)
{

}


void HANDLER::IO::copyHandlerToHost()
{

}


void HANDLER::IO::copyHandlerToDevice()
{

}
