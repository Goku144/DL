#include "HANDLER/Workspace.hpp"
#include "OPERATOR/Conv2DReLU.hpp"
#include "OPERATOR/CrossEntropy.hpp"
#include "OPERATOR/Normalize.hpp"
#include "OPERATOR/Pool.hpp"
#include "OPERATOR/Relu.hpp"
#include "OPERATOR/SGD.hpp"
#include "OPERATOR/Softmax.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <math.h>
#include <stdio.h>
#include <sys/stat.h>


int main(void)
{
  VIEW::Math y,x,w,b, pry;
  HANDLER::Cpu cpu;
  HANDLER::Cuda gpu;
  HANDLER::IO io(cpu, gpu);
  HANDLER::File file;
  HANDLER::Workspace workspace(io, file, CORE::MEMORY_512_MB);
  int xDims[VIEW::MAX_RANK] = {1, 1, 4, 4};
  int wDims[VIEW::MAX_RANK] = {1, 1, 3, 3};
  int bDims[VIEW::MAX_RANK] = {1, 0, 0, 0};

  pry.getLayout().setShape(xDims, 4, VIEW::FLOAT);
  y.getLayout().setShape(xDims, 4, VIEW::F16);
  x.getLayout().setShape(xDims, 4, VIEW::F16);
  w.getLayout().setShape(wDims, 4, VIEW::F16);
  b.getLayout().setShape(bDims, 1, VIEW::F16);

  io.bind(pry);
  io.bind(y);
  io.bind(x);
  io.bind(w);
  io.bind(b);

  __half xv[16];

  for (int i = 0; i < 16; i++) {
    xv[i] = __float2half((float)i);
  }
  __half zero = __float2half(0.0f), one = __float2half(1.0f);

  __half wv[9] = 
  {
    zero, one, zero,
    one, zero, one,
    zero, one, zero,
  };

  __half bv[1] = {one};

  __half* xCpu = (__half*)x.getCpuPtr();
  for (int i = 0; i < 16; i++) {
    xCpu[i] = __float2half((float)i);
  }

  __half* wCpu = (__half*)w.getCpuPtr();
  for (int i = 0; i < 9; i++) {
    wCpu[i] = wv[i];
  }

  __half* bCpu = (__half*)b.getCpuPtr();
  bCpu[0] = one;

io.copyHostToDevice(x);
io.copyHostToDevice(w);
io.copyHostToDevice(b);

  OPERATOR::Conv2DRelu conv(workspace);
  conv.setOperand(y,x,w,b);
  conv.forward();
  io.copyHalfToCpuFloat(pry, y);
  io.printData(pry);
  return 0;
}
