#include "HANDLER/Workspace.hpp"
#include "OPERATOR/Conv2DReLU.hpp"
#include "OPERATOR/CrossEntropy.hpp"
#include "OPERATOR/MatrixMulBias.hpp"
#include "OPERATOR/Normalize.hpp"
#include "OPERATOR/Pool.hpp"
#include "OPERATOR/Relu.hpp"
#include "OPERATOR/SGD.hpp"
#include "OPERATOR/Softmax.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

static bool check(bool condition, const char *message)
{
  if(condition)
  {
    printf("[PASS] %s\n", message);
    return true;
  }

  printf("[FAIL] %s\n", message);
  return false;
}

static bool checkIO(HANDLER::IO& io, const char *message)
{
  if(io.peekErr() == CORE::ioSuccess)
  {
    printf("[PASS] %s\n", message);
    return true;
  }

  printf("[FAIL] %s\n", message);
  io.info(CORE::WARN);
  io.clearErr();
  return false;
}

static bool checkWorkspace(HANDLER::Workspace& workspace, const char *message)
{
  if(workspace.peekErr() == CORE::workspaceSuccess)
  {
    printf("[PASS] %s\n", message);
    return true;
  }

  printf("[FAIL] %s\n", message);
  workspace.info(CORE::WARN);
  workspace.clearErr();
  return false;
}

static bool syncWorkspace(HANDLER::Workspace& workspace, const char *message)
{
  cudaError_t err = cudaStreamSynchronize(workspace.getStream());
  return check(err == cudaSuccess, message);
}

static void setMath(HANDLER::IO& io, VIEW::Math& math, int dims[VIEW::MAX_RANK], int rank, VIEW::DType dtype)
{
  math.getLayout().setShape(dims, rank, dtype);
  io.bind(math);
}

static void copyFloatToHalfHost(VIEW::Math& math, const float *src)
{
  __half *dst = (__half *)math.getCpuPtr();
  for(size_t index = 0; index < math.getCount(); index++)
    dst[index] = __float2half_rn(src[index]);
}

static bool copyHalfOutputToFloat(HANDLER::IO& io, VIEW::Math& dst, VIEW::Math& src)
{
  io.copyHalfToCpuFloat(dst, src);
  return checkIO(io, "copy F16 GPU output to CPU float");
}

static bool allClose(const float *actual, const float *expected, size_t count, float tol)
{
  for(size_t index = 0; index < count; index++)
  {
    float diff = fabsf(actual[index] - expected[index]);
    if(diff > tol)
    {
      printf("  mismatch[%zu]: actual=%f expected=%f diff=%f\n", index, actual[index], expected[index], diff);
      return false;
    }
  }
  return true;
}

static bool testNormalize(HANDLER::Workspace& workspace, HANDLER::IO& io)
{
  printf("\nNormalize test\n");
  int dims[VIEW::MAX_RANK] = {16, 0, 0, 0};
  VIEW::Math in, out, outCpu;
  setMath(io, in, dims, 1, VIEW::F16);
  setMath(io, out, dims, 1, VIEW::F16);

  float x[16];
  float expected[16];
  for(int i = 0; i < 16; i++)
  {
    x[i] = (float)(i * 16);
    expected[i] = x[i] / 255.0f;
  }

  copyFloatToHalfHost(in, x);
  io.copyHostToDevice(in);

  OPERATOR::Normalize norm(workspace, out, in);
  norm.normByScalar(255.0f);

  bool ok = true;
  ok &= syncWorkspace(workspace, "Normalize stream sync");
  ok &= copyHalfOutputToFloat(io, outCpu, out);
  ok &= check(allClose((float *)outCpu.getCpuPtr(), expected, 16, 0.0025f), "Normalize divides by scalar");
  return ok;
}

static bool testRelu(HANDLER::Workspace& workspace, HANDLER::IO& io)
{
  printf("\nRelu test\n");
  int dims[VIEW::MAX_RANK] = {16, 0, 0, 0};
  VIEW::Math in, out, dOut, dIn, outCpu, dInCpu;
  setMath(io, in, dims, 1, VIEW::F16);
  setMath(io, out, dims, 1, VIEW::F16);
  setMath(io, dOut, dims, 1, VIEW::F16);
  setMath(io, dIn, dims, 1, VIEW::F16);

  float x[16] = {-4, -3, -2, -1, 0, 1, 2, 3, 4, -5, 6, -7, 8, -9, 10, -11};
  float dy[16];
  float expectedY[16];
  float expectedDX[16];
  for(int i = 0; i < 16; i++)
  {
    dy[i] = (float)(i + 1);
    expectedY[i] = x[i] > 0.0f ? x[i] : 0.0f;
    expectedDX[i] = x[i] > 0.0f ? dy[i] : 0.0f;
  }

  copyFloatToHalfHost(in, x);
  copyFloatToHalfHost(dOut, dy);
  io.copyHostToDevice(in);
  io.copyHostToDevice(dOut);

  OPERATOR::Relu relu(workspace, out, in);
  relu.forward();

  bool ok = true;
  ok &= syncWorkspace(workspace, "Relu forward stream sync");
  ok &= copyHalfOutputToFloat(io, outCpu, out);
  ok &= check(allClose((float *)outCpu.getCpuPtr(), expectedY, 16, 0.001f), "Relu forward");

  relu.setGradOperand(dIn, dOut);
  relu.backward();
  ok &= syncWorkspace(workspace, "Relu backward stream sync");
  ok &= copyHalfOutputToFloat(io, dInCpu, dIn);
  ok &= check(allClose((float *)dInCpu.getCpuPtr(), expectedDX, 16, 0.001f), "Relu backward");
  return ok;
}

static bool testSgd(HANDLER::Workspace& workspace, HANDLER::IO& io)
{
  printf("\nSGD test\n");
  int dims[VIEW::MAX_RANK] = {16, 0, 0, 0};
  VIEW::Math weight, grad, weightCpu;
  setMath(io, weight, dims, 1, VIEW::F16);
  setMath(io, grad, dims, 1, VIEW::F16);

  float w[16];
  float g[16];
  float expected[16];
  for(int i = 0; i < 16; i++)
  {
    w[i] = 1.0f + (float)i;
    g[i] = 0.25f * (float)(i + 1);
    expected[i] = w[i] - 0.1f * g[i];
  }

  copyFloatToHalfHost(weight, w);
  copyFloatToHalfHost(grad, g);
  io.copyHostToDevice(weight);
  io.copyHostToDevice(grad);

  OPERATOR::SGD sgd(workspace, weight, grad);
  sgd.update(0.1f);

  bool ok = true;
  ok &= syncWorkspace(workspace, "SGD stream sync");
  ok &= copyHalfOutputToFloat(io, weightCpu, weight);
  ok &= check(allClose((float *)weightCpu.getCpuPtr(), expected, 16, 0.01f), "SGD weight update");
  return ok;
}

static bool testSoftmaxCrossEntropy(HANDLER::Workspace& workspace, HANDLER::IO& io)
{
  printf("\nSoftmax + CrossEntropy test\n");
  int probDims[VIEW::MAX_RANK] = {2, 16, 0, 0};
  int labelDims[VIEW::MAX_RANK] = {2, 0, 0, 0};
  int lossDims[VIEW::MAX_RANK] = {1, 0, 0, 0};
  VIEW::Math logits, prob, dProb, labels, loss, probCpu, dProbCpu;
  setMath(io, logits, probDims, 2, VIEW::F16);
  setMath(io, prob, probDims, 2, VIEW::F16);
  setMath(io, dProb, probDims, 2, VIEW::F16);
  setMath(io, labels, labelDims, 1, VIEW::CHAR);
  setMath(io, loss, lossDims, 1, VIEW::FLOAT);

  float z[32];
  float expectedProb[32];
  float expectedGrad[32];
  uint8_t y[2] = {3, 7};
  for(int row = 0; row < 2; row++)
  {
    float sum = 0.0f;
    for(int col = 0; col < 16; col++)
    {
      z[row * 16 + col] = 0.125f * (float)(col - 8 + row);
      expectedProb[row * 16 + col] = expf(z[row * 16 + col]);
      sum += expectedProb[row * 16 + col];
    }
    for(int col = 0; col < 16; col++)
      expectedProb[row * 16 + col] /= sum;
  }
  for(int row = 0; row < 2; row++)
    for(int col = 0; col < 16; col++)
      expectedGrad[row * 16 + col] = (expectedProb[row * 16 + col] - (col == y[row] ? 1.0f : 0.0f)) * 0.5f;
  float expectedLoss = (-logf(expectedProb[3] + 1e-7f) - logf(expectedProb[16 + 7] + 1e-7f)) * 0.5f;

  copyFloatToHalfHost(logits, z);
  *((uint8_t *)labels.getCpuPtr() + 0) = y[0];
  *((uint8_t *)labels.getCpuPtr() + 1) = y[1];
  io.copyHostToDevice(logits);
  io.copyHostToDevice(labels);

  OPERATOR::Softmax softmax(workspace, prob, logits);
  softmax.forward();

  bool ok = true;
  ok &= syncWorkspace(workspace, "Softmax forward stream sync");
  ok &= copyHalfOutputToFloat(io, probCpu, prob);
  ok &= check(allClose((float *)probCpu.getCpuPtr(), expectedProb, 32, 0.0025f), "Softmax probabilities");

  OPERATOR::CrossEntropy ce(workspace, dProb, loss, prob, labels);
  ce.forwardBackward();
  ok &= syncWorkspace(workspace, "CrossEntropy stream sync");
  io.copyDeviceToHost(loss);
  ok &= checkIO(io, "CrossEntropy loss copied to CPU");
  ok &= check(fabsf(*(float *)loss.getCpuPtr() - expectedLoss) < 0.005f, "CrossEntropy loss");
  ok &= copyHalfOutputToFloat(io, dProbCpu, dProb);
  ok &= check(allClose((float *)dProbCpu.getCpuPtr(), expectedGrad, 32, 0.003f), "CrossEntropy gradient");
  return ok;
}

static bool testPool(HANDLER::Workspace& workspace, HANDLER::IO& io)
{
  printf("\nPool test\n");
  int inDims[VIEW::MAX_RANK] = {1, 1, 4, 4};
  int outDims[VIEW::MAX_RANK] = {1, 1, 2, 2};
  VIEW::Math in, out, dOut, dIn, outCpu, dInCpu;
  setMath(io, in, inDims, 4, VIEW::F16);
  setMath(io, out, outDims, 4, VIEW::F16);
  setMath(io, dOut, outDims, 4, VIEW::F16);
  setMath(io, dIn, inDims, 4, VIEW::F16);

  float x[16] = {1, 5, 2, 3, 4, 9, 6, 7, 8, 1, 10, 2, 3, 4, 5, 11};
  float dy[4] = {1, 2, 3, 4};
  float expectedY[4] = {9, 7, 8, 11};
  float expectedDX[16] = {0, 0, 0, 0, 0, 1, 0, 2, 3, 0, 0, 0, 0, 0, 0, 4};

  copyFloatToHalfHost(in, x);
  copyFloatToHalfHost(dOut, dy);
  io.copyHostToDevice(in);
  io.copyHostToDevice(dOut);

  OPERATOR::Pool pool(workspace, out, in);
  pool.setConfig(2, 2, 0, 0, 2, 2);
  pool.maxForward();

  bool ok = true;
  ok &= syncWorkspace(workspace, "Pool forward stream sync");
  ok &= copyHalfOutputToFloat(io, outCpu, out);
  ok &= check(allClose((float *)outCpu.getCpuPtr(), expectedY, 4, 0.001f), "Pool max forward");

  pool.setGradOperand(dIn, dOut);
  pool.maxBackward();
  ok &= syncWorkspace(workspace, "Pool backward stream sync");
  ok &= copyHalfOutputToFloat(io, dInCpu, dIn);
  ok &= check(allClose((float *)dInCpu.getCpuPtr(), expectedDX, 16, 0.001f), "Pool max backward");
  return ok;
}

static bool testConv2D(HANDLER::Workspace& workspace, HANDLER::IO& io)
{
  printf("\nConv2DRelu test\n");
  int xDims[VIEW::MAX_RANK] = {1, 1, 4, 4};
  int wDims[VIEW::MAX_RANK] = {1, 1, 3, 3};
  int bDims[VIEW::MAX_RANK] = {1, 0, 0, 0};
  VIEW::Math x, w, b, y, dY, dX, dW, dB, yCpu, dXCpu, dWCpu, dBCpu;
  setMath(io, x, xDims, 4, VIEW::F16);
  setMath(io, w, wDims, 4, VIEW::F16);
  setMath(io, b, bDims, 1, VIEW::F16);
  setMath(io, y, xDims, 4, VIEW::F16);
  setMath(io, dY, xDims, 4, VIEW::F16);
  setMath(io, dX, xDims, 4, VIEW::F16);
  setMath(io, dW, wDims, 4, VIEW::F16);
  setMath(io, dB, bDims, 1, VIEW::F16);

  float xv[16];
  float wv[9];
  float bv[1] = {0.5f};
  float dy[16];
  float expectedY[16];
  float expectedDX[16];
  float expectedDW[9];
  float expectedDB[1] = {16.0f};

  for(int i = 0; i < 16; i++)
  {
    xv[i] = (float)(i + 1);
    dy[i] = 1.0f;
    expectedY[i] = 0.5f;
    expectedDX[i] = 0.0f;
  }
  for(int i = 0; i < 9; i++)
  {
    wv[i] = 1.0f;
    expectedDW[i] = 0.0f;
  }

  for(int oh = 0; oh < 4; oh++)
    for(int ow = 0; ow < 4; ow++)
      for(int kh = 0; kh < 3; kh++)
        for(int kw = 0; kw < 3; kw++)
        {
          int ih = oh + kh - 1;
          int iw = ow + kw - 1;
          if(ih < 0 || ih >= 4 || iw < 0 || iw >= 4) continue;
          expectedY[oh * 4 + ow] += xv[ih * 4 + iw];
          expectedDW[kh * 3 + kw] += xv[ih * 4 + iw];
          expectedDX[ih * 4 + iw] += 1.0f;
        }

  copyFloatToHalfHost(x, xv);
  copyFloatToHalfHost(w, wv);
  copyFloatToHalfHost(b, bv);
  copyFloatToHalfHost(dY, dy);
  io.copyHostToDevice(x);
  io.copyHostToDevice(w);
  io.copyHostToDevice(b);
  io.copyHostToDevice(dY);

  OPERATOR::Conv2DRelu conv(workspace, y, x, w, b);
  conv.setConfig(1, 1, 1, 1, 1, 1);
  conv.forward();

  bool ok = true;
  ok &= syncWorkspace(workspace, "Conv2D forward stream sync");
  ok &= copyHalfOutputToFloat(io, yCpu, y);
  ok &= check(allClose((float *)yCpu.getCpuPtr(), expectedY, 16, 0.05f), "Conv2D forward computes conv plus bias");

  conv.setGradOperand(dX, dW, dB, dY);
  conv.backward();
  ok &= syncWorkspace(workspace, "Conv2D backward stream sync");
  ok &= copyHalfOutputToFloat(io, dXCpu, dX);
  ok &= check(allClose((float *)dXCpu.getCpuPtr(), expectedDX, 16, 0.05f), "Conv2D dInput");
  ok &= copyHalfOutputToFloat(io, dWCpu, dW);
  ok &= check(allClose((float *)dWCpu.getCpuPtr(), expectedDW, 9, 0.1f), "Conv2D dWeight");
  ok &= copyHalfOutputToFloat(io, dBCpu, dB);
  ok &= check(allClose((float *)dBCpu.getCpuPtr(), expectedDB, 1, 0.05f), "Conv2D dBias");
  return ok;
}

static bool testMatrixMulBias(HANDLER::Workspace& workspace, HANDLER::IO& io)
{
  printf("\nMatrixMulBias test\n");
  int xDims[VIEW::MAX_RANK] = {2, 8, 0, 0};
  int wDims[VIEW::MAX_RANK] = {8, 16, 0, 0};
  int bDims[VIEW::MAX_RANK] = {16, 0, 0, 0};
  int yDims[VIEW::MAX_RANK] = {2, 16, 0, 0};
  VIEW::Math x, w, b, y, dY, dX, dW, dB, yCpu, dXCpu, dWCpu, dBCpu;
  setMath(io, x, xDims, 2, VIEW::F16);
  setMath(io, w, wDims, 2, VIEW::F16);
  setMath(io, b, bDims, 1, VIEW::F16);
  setMath(io, y, yDims, 2, VIEW::F16);
  setMath(io, dY, yDims, 2, VIEW::F16);
  setMath(io, dX, xDims, 2, VIEW::F16);
  setMath(io, dW, wDims, 2, VIEW::F16);
  setMath(io, dB, bDims, 1, VIEW::F16);

  float xv[16];
  float wv[128];
  float bv[16];
  float dy[32];
  float expectedY[32];
  float expectedDX[16];
  float expectedDW[128];
  float expectedDB[16];

  for(int i = 0; i < 16; i++)
    xv[i] = 0.1f * (float)(i + 1);
  for(int i = 0; i < 128; i++)
    wv[i] = 0.01f * (float)((i % 17) - 8);
  for(int i = 0; i < 16; i++)
  {
    bv[i] = 0.05f * (float)i;
    expectedDB[i] = 0.0f;
  }
  for(int i = 0; i < 32; i++)
    dy[i] = 0.02f * (float)(i + 1);

  for(int n = 0; n < 2; n++)
    for(int m = 0; m < 16; m++)
    {
      float sum = bv[m];
      for(int k = 0; k < 8; k++)
        sum += xv[n * 8 + k] * wv[k * 16 + m];
      expectedY[n * 16 + m] = sum;
    }

  for(int n = 0; n < 2; n++)
    for(int k = 0; k < 8; k++)
    {
      float sum = 0.0f;
      for(int m = 0; m < 16; m++)
        sum += dy[n * 16 + m] * wv[k * 16 + m];
      expectedDX[n * 8 + k] = sum;
    }

  for(int k = 0; k < 8; k++)
    for(int m = 0; m < 16; m++)
    {
      float sum = 0.0f;
      for(int n = 0; n < 2; n++)
        sum += xv[n * 8 + k] * dy[n * 16 + m];
      expectedDW[k * 16 + m] = sum;
    }

  for(int m = 0; m < 16; m++)
    for(int n = 0; n < 2; n++)
      expectedDB[m] += dy[n * 16 + m];

  copyFloatToHalfHost(x, xv);
  copyFloatToHalfHost(w, wv);
  copyFloatToHalfHost(b, bv);
  copyFloatToHalfHost(dY, dy);
  io.copyHostToDevice(x);
  io.copyHostToDevice(w);
  io.copyHostToDevice(b);
  io.copyHostToDevice(dY);

  OPERATOR::MatrixMulBias linear(workspace, y, x, w, b);
  linear.forward();

  bool ok = true;
  ok &= syncWorkspace(workspace, "MatrixMulBias forward stream sync");
  ok &= copyHalfOutputToFloat(io, yCpu, y);
  ok &= check(allClose((float *)yCpu.getCpuPtr(), expectedY, 32, 0.02f), "MatrixMulBias forward xW + b");

  linear.setGradOperand(dX, dW, dB, dY);
  linear.backward();
  ok &= syncWorkspace(workspace, "MatrixMulBias backward stream sync");
  ok &= copyHalfOutputToFloat(io, dXCpu, dX);
  ok &= check(allClose((float *)dXCpu.getCpuPtr(), expectedDX, 16, 0.02f), "MatrixMulBias dInput");
  ok &= copyHalfOutputToFloat(io, dWCpu, dW);
  ok &= check(allClose((float *)dWCpu.getCpuPtr(), expectedDW, 128, 0.02f), "MatrixMulBias dWeight");
  ok &= copyHalfOutputToFloat(io, dBCpu, dB);
  ok &= check(allClose((float *)dBCpu.getCpuPtr(), expectedDB, 16, 0.02f), "MatrixMulBias dBias");
  return ok;
}

int main(void)
{
  HANDLER::Cpu cpu(CORE::MEMORY_32_MB);
  HANDLER::Cuda gpu(CORE::MEMORY_32_MB);
  HANDLER::IO io(cpu, gpu);
  HANDLER::File file;
  HANDLER::Workspace workspace(io, file, CORE::MEMORY_32_MB);

  bool ok = true;
  ok &= checkWorkspace(workspace, "Workspace created");
  ok &= testNormalize(workspace, io);
  ok &= testRelu(workspace, io);
  ok &= testSgd(workspace, io);
  ok &= testSoftmaxCrossEntropy(workspace, io);
  ok &= testPool(workspace, io);
  ok &= testConv2D(workspace, io);
  ok &= testMatrixMulBias(workspace, io);

  printf("\nOperator readiness: %s\n", ok ? "ready" : "needs fixes");
  return ok ? 0 : 1;
}
