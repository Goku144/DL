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

static bool checkWorkspaceSuccess(HANDLER::Workspace& workspace, const char *message)
{
  if(workspace.peekErr() == CORE::workspaceSuccess)
  {
    printf("[PASS] %s\n", message);
    return true;
  }

  printf("[FAIL] %s\n", message);
  workspace.info(CORE::WARN);
  return false;
}

static bool checkFileSuccess(HANDLER::File& file, const char *message)
{
  if(file.peekErr() == CORE::fileSuccess)
  {
    printf("[PASS] %s\n", message);
    return true;
  }

  printf("[FAIL] %s\n", message);
  file.info(CORE::WARN);
  return false;
}

static bool checkIOSuccess(HANDLER::IO& io, const char *message)
{
  if(io.peekErr() == CORE::ioSuccess)
  {
    printf("[PASS] %s\n", message);
    return true;
  }

  printf("[FAIL] %s\n", message);
  io.info(CORE::WARN);
  return false;
}

static void setGpuMath(HANDLER::IO& io, VIEW::Math& math, int dims[VIEW::MAX_RANK], int rank, VIEW::DType dtype)
{
  VIEW::Shape layout(dims, rank, dtype);
  math.setLayout(layout);
  io.bindGpu(math);
}

static bool copyHalfToGpu(VIEW::Math& math, const float *src)
{
  __half tmp[256];
  if(math.getCount() > 256) return false;

  for(size_t index = 0; index < math.getCount(); index++)
    tmp[index] = __float2half_rn(src[index]);

  return cudaMemcpy(math.getGpuPtr(), tmp, math.getCount() * VIEW::F16, cudaMemcpyHostToDevice) == cudaSuccess;
}

static bool copyCharToGpu(VIEW::Math& math, const uint8_t *src)
{
  return cudaMemcpy(math.getGpuPtr(), src, math.getCount() * VIEW::CHAR, cudaMemcpyHostToDevice) == cudaSuccess;
}

static bool readGpuFloatScalar(VIEW::Math& math, float *dst)
{
  return cudaMemcpy(dst, math.getGpuPtr(), VIEW::FLOAT, cudaMemcpyDeviceToHost) == cudaSuccess;
}

static bool allClose(const float *actual, const float *expected, size_t count, float tol)
{
  for(size_t index = 0; index < count; index++)
    if(fabsf(actual[index] - expected[index]) > tol)
      return false;
  return true;
}

static bool testReluOperator(HANDLER::Workspace& workspace, HANDLER::IO& io)
{
  printf("Relu operator deep test start\n");

  int dims[VIEW::MAX_RANK] = {2, 8, 0, 0};
  VIEW::Math in;
  VIEW::Math out;
  VIEW::Math dOut;
  VIEW::Math dIn;
  VIEW::Math outCpu;
  VIEW::Math dInCpu;

  setGpuMath(io, in, dims, 2, VIEW::F16);
  setGpuMath(io, out, dims, 2, VIEW::F16);
  setGpuMath(io, dOut, dims, 2, VIEW::F16);
  setGpuMath(io, dIn, dims, 2, VIEW::F16);

  float x[16] = {-3.0f, -2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f,
                 3.0f, -4.0f, 5.0f, -6.0f, 7.0f, -8.0f, 9.0f, -10.0f};
  float dy[16];
  float expectedY[16];
  float expectedDx[16];
  for(int index = 0; index < 16; index++)
  {
    dy[index] = (float)(index + 1);
    expectedY[index] = x[index] > 0.0f ? x[index] : 0.0f;
    expectedDx[index] = x[index] > 0.0f ? dy[index] : 0.0f;
  }

  bool ok = true;
  ok &= check(copyHalfToGpu(in, x), "Relu input copied to GPU");
  ok &= check(copyHalfToGpu(dOut, dy), "Relu upstream gradient copied to GPU");

  OPERATOR::Relu relu(workspace, out, in);
  relu.forward();
  cudaError_t syncErr = cudaStreamSynchronize(workspace.getStream());
  ok &= check(syncErr == cudaSuccess, "Relu forward stream synchronizes cleanly");
  io.copyHalfToCpuFloat(outCpu, out);
  ok &= checkIOSuccess(io, "Relu forward copied back to CPU float");
  ok &= check(allClose((float *)outCpu.getCpuPtr(), expectedY, 16, 0.001f), "Relu forward clamps negatives to zero");

  relu.setGradOperand(dIn, dOut);
  relu.backward();
  syncErr = cudaStreamSynchronize(workspace.getStream());
  ok &= check(syncErr == cudaSuccess, "Relu backward stream synchronizes cleanly");
  io.copyHalfToCpuFloat(dInCpu, dIn);
  ok &= checkIOSuccess(io, "Relu backward copied back to CPU float");
  ok &= check(allClose((float *)dInCpu.getCpuPtr(), expectedDx, 16, 0.001f), "Relu backward gates gradients by input sign");

  return ok;
}

static bool testSgdOperator(HANDLER::Workspace& workspace, HANDLER::IO& io)
{
  printf("SGD operator deep test start\n");

  int dims[VIEW::MAX_RANK] = {16, 0, 0, 0};
  VIEW::Math weight;
  VIEW::Math grad;
  VIEW::Math weightCpu;

  setGpuMath(io, weight, dims, 1, VIEW::F16);
  setGpuMath(io, grad, dims, 1, VIEW::F16);

  float w[16];
  float g[16];
  float expected[16];
  for(int index = 0; index < 16; index++)
  {
    w[index] = 1.0f + (float)index;
    g[index] = 0.25f * (float)(index + 1);
    expected[index] = w[index] - 0.1f * g[index];
  }

  bool ok = true;
  ok &= check(copyHalfToGpu(weight, w), "SGD weight copied to GPU");
  ok &= check(copyHalfToGpu(grad, g), "SGD gradient copied to GPU");

  OPERATOR::SGD sgd(workspace, weight, grad);
  sgd.update(0.1f);
  cudaError_t syncErr = cudaStreamSynchronize(workspace.getStream());
  ok &= check(syncErr == cudaSuccess, "SGD update stream synchronizes cleanly");

  io.copyHalfToCpuFloat(weightCpu, weight);
  ok &= checkIOSuccess(io, "SGD updated weight copied back to CPU float");
  ok &= check(allClose((float *)weightCpu.getCpuPtr(), expected, 16, 0.01f), "SGD applies weight -= lr * grad");

  return ok;
}

static bool testCrossEntropyOperator(HANDLER::Workspace& workspace, HANDLER::IO& io)
{
  printf("CrossEntropy operator deep test start\n");

  int probDims[VIEW::MAX_RANK] = {2, 16, 0, 0};
  int labelDims[VIEW::MAX_RANK] = {2, 0, 0, 0};
  int lossDims[VIEW::MAX_RANK] = {1, 0, 0, 0};
  VIEW::Math prob;
  VIEW::Math labels;
  VIEW::Math loss;
  VIEW::Math dProb;
  VIEW::Math dProbCpu;

  setGpuMath(io, prob, probDims, 2, VIEW::F16);
  setGpuMath(io, dProb, probDims, 2, VIEW::F16);
  setGpuMath(io, labels, labelDims, 1, VIEW::CHAR);
  setGpuMath(io, loss, lossDims, 1, VIEW::FLOAT);

  float p[32];
  float expectedGrad[32];
  for(int index = 0; index < 32; index++)
    p[index] = 0.02f;
  p[3] = 0.70f;
  p[19] = 0.55f;
  uint8_t y[2] = {3, 3};
  float expectedLoss = (-logf(p[3] + 1e-7f) - logf(p[19] + 1e-7f)) * 0.5f;
  for(int row = 0; row < 2; row++)
    for(int col = 0; col < 16; col++)
      expectedGrad[row * 16 + col] = (p[row * 16 + col] - (col == 3 ? 1.0f : 0.0f)) * 0.5f;

  bool ok = true;
  ok &= check(copyHalfToGpu(prob, p), "CrossEntropy probability copied to GPU");
  ok &= check(copyCharToGpu(labels, y), "CrossEntropy labels copied to GPU");

  OPERATOR::CrossEntropy ce(workspace, dProb, loss, prob, labels);
  ce.forwardBackward();
  cudaError_t syncErr = cudaStreamSynchronize(workspace.getStream());
  ok &= check(syncErr == cudaSuccess, "CrossEntropy stream synchronizes cleanly");

  float gotLoss = 0.0f;
  ok &= check(readGpuFloatScalar(loss, &gotLoss), "CrossEntropy scalar loss copied back to CPU");
  ok &= check(fabsf(gotLoss - expectedLoss) < 0.003f, "CrossEntropy loss matches target probabilities");

  io.copyHalfToCpuFloat(dProbCpu, dProb);
  ok &= checkIOSuccess(io, "CrossEntropy gradient copied back to CPU float");
  ok &= check(allClose((float *)dProbCpu.getCpuPtr(), expectedGrad, 32, 0.0025f), "CrossEntropy gradient is (prob - onehot) / batch");

  return ok;
}

static bool testSoftmaxOperator(HANDLER::Workspace& workspace, HANDLER::IO& io)
{
  printf("Softmax operator deep test start\n");

  int dims[VIEW::MAX_RANK] = {2, 16, 0, 0};
  VIEW::Math in;
  VIEW::Math out;
  VIEW::Math outCpu;

  setGpuMath(io, in, dims, 2, VIEW::F16);
  setGpuMath(io, out, dims, 2, VIEW::F16);

  float logits[32];
  float expected[32];
  for(int row = 0; row < 2; row++)
  {
    float sum = 0.0f;
    for(int col = 0; col < 16; col++)
    {
      logits[row * 16 + col] = 0.1f * (float)(col - 8 + row);
      expected[row * 16 + col] = expf(logits[row * 16 + col]);
      sum += expected[row * 16 + col];
    }
    for(int col = 0; col < 16; col++)
      expected[row * 16 + col] /= sum;
  }

  bool ok = true;
  ok &= check(copyHalfToGpu(in, logits), "Softmax logits copied to GPU");

  OPERATOR::Softmax softmax(workspace, out, in);
  softmax.forward();
  cudaError_t syncErr = cudaStreamSynchronize(workspace.getStream());
  ok &= check(syncErr == cudaSuccess, "Softmax forward stream synchronizes cleanly");

  io.copyHalfToCpuFloat(outCpu, out);
  ok &= checkIOSuccess(io, "Softmax output copied back to CPU float");
  ok &= check(allClose((float *)outCpu.getCpuPtr(), expected, 32, 0.0025f), "Softmax output matches CPU row softmax");

  return ok;
}

static bool testPoolOperator(HANDLER::Workspace& workspace, HANDLER::IO& io)
{
  printf("Pool operator deep test start\n");

  int inDims[VIEW::MAX_RANK] = {1, 1, 4, 4};
  int outDims[VIEW::MAX_RANK] = {1, 1, 2, 2};
  VIEW::Math in;
  VIEW::Math out;
  VIEW::Math dOut;
  VIEW::Math dIn;
  VIEW::Math outCpu;
  VIEW::Math dInCpu;

  setGpuMath(io, in, inDims, 4, VIEW::F16);
  setGpuMath(io, out, outDims, 4, VIEW::F16);
  setGpuMath(io, dOut, outDims, 4, VIEW::F16);
  setGpuMath(io, dIn, inDims, 4, VIEW::F16);

  float x[16] = {1, 5, 2, 3,
                 4, 9, 6, 7,
                 8, 1, 10, 2,
                 3, 4, 5, 11};
  float dy[4] = {1, 2, 3, 4};
  float expectedY[4] = {9, 7, 8, 11};
  float expectedDx[16] = {0, 0, 0, 0,
                          0, 1, 0, 2,
                          3, 0, 0, 0,
                          0, 0, 0, 4};

  bool ok = true;
  ok &= check(copyHalfToGpu(in, x), "Pool input copied to GPU");
  ok &= check(copyHalfToGpu(dOut, dy), "Pool upstream gradient copied to GPU");

  OPERATOR::Pool pool(workspace, out, in);
  pool.setConfig(2, 2, 0, 0, 2, 2);
  pool.maxForward();
  cudaError_t syncErr = cudaStreamSynchronize(workspace.getStream());
  ok &= check(syncErr == cudaSuccess, "Pool forward stream synchronizes cleanly");
  io.copyHalfToCpuFloat(outCpu, out);
  ok &= checkIOSuccess(io, "Pool output copied back to CPU float");
  ok &= check(allClose((float *)outCpu.getCpuPtr(), expectedY, 4, 0.001f), "Pool forward picks 2x2 maxima");

  pool.setGradOperand(dIn, dOut);
  pool.maxBackward();
  syncErr = cudaStreamSynchronize(workspace.getStream());
  ok &= check(syncErr == cudaSuccess, "Pool backward stream synchronizes cleanly");
  io.copyHalfToCpuFloat(dInCpu, dIn);
  ok &= checkIOSuccess(io, "Pool gradient copied back to CPU float");
  ok &= check(allClose((float *)dInCpu.getCpuPtr(), expectedDx, 16, 0.001f), "Pool backward routes gradients to maxima");

  return ok;
}

static bool testConv2DOperator(HANDLER::Workspace& workspace, HANDLER::IO& io)
{
  printf("Conv2D operator deep test start\n");

  int xDims[VIEW::MAX_RANK] = {1, 1, 4, 4};
  int wDims[VIEW::MAX_RANK] = {1, 1, 3, 3};
  int bDims[VIEW::MAX_RANK] = {1, 0, 0, 0};
  VIEW::Math x;
  VIEW::Math w;
  VIEW::Math b;
  VIEW::Math y;
  VIEW::Math dY;
  VIEW::Math dX;
  VIEW::Math dW;
  VIEW::Math dB;
  VIEW::Math yCpu;
  VIEW::Math dXCpu;
  VIEW::Math dWCpu;
  VIEW::Math dBCpu;

  setGpuMath(io, x, xDims, 4, VIEW::F16);
  setGpuMath(io, y, xDims, 4, VIEW::F16);
  setGpuMath(io, dY, xDims, 4, VIEW::F16);
  setGpuMath(io, dX, xDims, 4, VIEW::F16);
  setGpuMath(io, w, wDims, 4, VIEW::F16);
  setGpuMath(io, dW, wDims, 4, VIEW::F16);
  setGpuMath(io, b, bDims, 1, VIEW::F16);
  setGpuMath(io, dB, bDims, 1, VIEW::F16);

  float xv[16];
  float wv[9];
  float bv[1] = {0.5f};
  float dy[16];
  float expectedY[16];
  float expectedDX[16];
  float expectedDW[9];
  float expectedDB[1] = {16.0f};

  for(int index = 0; index < 16; index++)
  {
    xv[index] = (float)(index + 1);
    dy[index] = 1.0f;
    expectedY[index] = 0.5f;
    expectedDX[index] = 0.0f;
  }
  for(int index = 0; index < 9; index++)
  {
    wv[index] = 1.0f;
    expectedDW[index] = 0.0f;
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

  bool ok = true;
  ok &= check(copyHalfToGpu(x, xv), "Conv2D input copied to GPU");
  ok &= check(copyHalfToGpu(w, wv), "Conv2D weight copied to GPU");
  ok &= check(copyHalfToGpu(b, bv), "Conv2D bias copied to GPU");
  ok &= check(copyHalfToGpu(dY, dy), "Conv2D upstream gradient copied to GPU");

  OPERATOR::Conv2DRelu conv(workspace, y, x, w, b);
  conv.setConfig(1, 1, 1, 1, 1, 1);
  conv.forward();
  cudaError_t syncErr = cudaStreamSynchronize(workspace.getStream());
  ok &= check(syncErr == cudaSuccess, "Conv2D forward stream synchronizes cleanly");
  io.copyHalfToCpuFloat(yCpu, y);
  ok &= checkIOSuccess(io, "Conv2D output copied back to CPU float");
  ok &= check(allClose((float *)yCpu.getCpuPtr(), expectedY, 16, 0.05f), "Conv2D forward computes padded 3x3 conv plus bias");

  conv.setGradOperand(dX, dW, dB, dY);
  conv.backward();
  syncErr = cudaStreamSynchronize(workspace.getStream());
  ok &= check(syncErr == cudaSuccess, "Conv2D backward stream synchronizes cleanly");

  io.copyHalfToCpuFloat(dBCpu, dB);
  ok &= checkIOSuccess(io, "Conv2D bias gradient copied back to CPU float");
  ok &= check(allClose((float *)dBCpu.getCpuPtr(), expectedDB, 1, 0.05f), "Conv2D backward bias gradient sums dOut");

  io.copyHalfToCpuFloat(dWCpu, dW);
  ok &= checkIOSuccess(io, "Conv2D weight gradient copied back to CPU float");
  ok &= check(allClose((float *)dWCpu.getCpuPtr(), expectedDW, 9, 0.1f), "Conv2D backward filter gradient matches CPU reference");

  io.copyHalfToCpuFloat(dXCpu, dX);
  ok &= checkIOSuccess(io, "Conv2D input gradient copied back to CPU float");
  ok &= check(allClose((float *)dXCpu.getCpuPtr(), expectedDX, 16, 0.05f), "Conv2D backward input gradient matches CPU reference");

  return ok;
}

int main()
{
  HANDLER::Cpu cpu(CORE::MEMORY_32_MB);
  HANDLER::Cuda cuda(CORE::MEMORY_32_MB);
  HANDLER::IO io(cpu, cuda);
  HANDLER::File file(io);

  const size_t scratchBytes = CORE::MEMORY_32_MB;
  HANDLER::Workspace workspace(io, file, scratchBytes);
  bool ok = true;

  printf("Workspace test start\n");

  ok &= checkWorkspaceSuccess(workspace, "workspace constructor");
  ok &= check(&workspace.getIO() == &io, "workspace returns original IO handler");
  ok &= check(&workspace.getFile() == &file, "workspace returns original File handler");
  ok &= check(workspace.getCudnnHandle() != NULL, "workspace owns a cuDNN handle");
  ok &= check(workspace.getCublasLtHandle() != NULL, "workspace owns a cuBLASLt handle");
  ok &= check(workspace.getScratchBytes() == scratchBytes, "workspace records scratchpad size");

  void *zeroScratch = workspace.getScratch(0);
  ok &= check(zeroScratch == NULL, "zero-byte scratch request returns NULL");
  ok &= checkWorkspaceSuccess(workspace, "zero-byte scratch request keeps success state");

  void *scratchSmall = workspace.getScratch(1);
  ok &= check(scratchSmall != NULL, "one-byte scratch request returns scratchpad");
  ok &= checkWorkspaceSuccess(workspace, "one-byte scratch request keeps success state");

  void *scratchFull = workspace.getScratch(scratchBytes);
  ok &= check(scratchFull == scratchSmall, "full scratch request reuses same scratchpad");
  ok &= checkWorkspaceSuccess(workspace, "full scratch request keeps success state");

  if(scratchFull != NULL)
  {
    cudaError_t memsetErr = cudaMemset(scratchFull, 0xA5, scratchBytes);
    ok &= check(memsetErr == cudaSuccess, "scratchpad accepts cudaMemset");

    cudaError_t syncErr = cudaDeviceSynchronize();
    ok &= check(syncErr == cudaSuccess, "scratchpad memset synchronizes cleanly");
  }

  void *tooLarge = workspace.getScratch(scratchBytes + 1);
  ok &= check(tooLarge == NULL, "oversized scratch request returns NULL");
  ok &= check(workspace.peekErr() == CORE::workspaceErrScratchOutOfBound,
              "oversized scratch request sets out-of-bound error");
  ok &= check(workspace.getErr() == CORE::workspaceErrScratchOutOfBound,
              "getErr returns and clears workspace error");
  ok &= check(workspace.peekErr() == CORE::workspaceSuccess,
              "workspace error is clear after getErr");

  printf("Image uint8 to half GPU test start\n");

  VIEW::Math imagePaths;
  VIEW::Math labels;
  VIEW::Math gpuImages;
  VIEW::Math cpuFloatImages;
  VIEW::Math normalizedGpuImages;
  VIEW::Math normalizedCpuFloatImages;

  file.readCsv(imagePaths, labels, "public/target/meta/test.csv");
  ok &= checkFileSuccess(file, "image csv loaded");

  file.readImages(imagePaths, "public/target/meta");
  ok &= checkFileSuccess(file, "uint8 images loaded on CPU");

  file.copyImageToDevice();
  ok &= checkFileSuccess(file, "uint8 images converted and copied to GPU half");

  size_t batch = file.getImageN() < 8 ? file.getImageN() : 8;
  if(batch > 0)
  {
    file.pullGpuImage(gpuImages, batch);
    ok &= checkFileSuccess(file, "pulled GPU image batch");

    VIEW::Shape& layout = gpuImages.getLayout();
    size_t expectedCount = batch * file.getImageChannel() * file.getImageHeight() * file.getImageWidth();
    size_t expectedBytes = CORE::ALIGNE(expectedCount * VIEW::F16, CORE::ALIGNE_TO_256);

    ok &= check(layout.getRank() == 4, "pulled GPU image rank is NCHW");
    ok &= check(layout.getDType() == VIEW::F16, "pulled GPU image dtype is F16");
    ok &= check(layout.getDim(0) == (int)batch, "pulled GPU image batch dim is correct");
    ok &= check(layout.getDim(1) == file.getImageChannel(), "pulled GPU image channel dim is correct");
    ok &= check(layout.getDim(2) == file.getImageHeight(), "pulled GPU image height dim is correct");
    ok &= check(layout.getDim(3) == file.getImageWidth(), "pulled GPU image width dim is correct");
    ok &= check(gpuImages.getCount() == expectedCount, "pulled GPU image element count is correct");
    ok &= check(gpuImages.getBytes() == expectedBytes, "pulled GPU image uses half-sized storage");
    ok &= check(gpuImages.getGpuPtr() != NULL, "pulled GPU image has device pointer");

    io.copyHalfToCpuFloat(cpuFloatImages, gpuImages);
    ok &= checkIOSuccess(io, "copied GPU half image batch to CPU float");

    VIEW::Shape& cpuFloatLayout = cpuFloatImages.getLayout();
    size_t expectedFloatBytes = CORE::ALIGNE(expectedCount * VIEW::FLOAT, CORE::ALIGNE_TO_256);

    ok &= check(cpuFloatLayout.getRank() == 4, "CPU float image rank is NCHW");
    ok &= check(cpuFloatLayout.getDType() == VIEW::FLOAT, "CPU float image dtype is FLOAT");
    ok &= check(cpuFloatLayout.getDim(0) == (int)batch, "CPU float image batch dim is correct");
    ok &= check(cpuFloatLayout.getDim(1) == file.getImageChannel(), "CPU float image channel dim is correct");
    ok &= check(cpuFloatLayout.getDim(2) == file.getImageHeight(), "CPU float image height dim is correct");
    ok &= check(cpuFloatLayout.getDim(3) == file.getImageWidth(), "CPU float image width dim is correct");
    ok &= check(cpuFloatImages.getCount() == expectedCount, "CPU float image element count is correct");
    ok &= check(cpuFloatImages.getBytes() == expectedFloatBytes, "CPU float image uses float-sized storage");
    ok &= check(cpuFloatImages.getCpuPtr() != NULL, "CPU float image has host pointer");

    VIEW::Shape normalizedLayout = gpuImages.getLayout();
    normalizedGpuImages.setLayout(normalizedLayout);
    io.bindGpu(normalizedGpuImages);
    ok &= checkIOSuccess(io, "allocated Normalize GPU output");

    bool normalizeOperandsOk =
      normalizedGpuImages.getGpuPtr() != NULL &&
      gpuImages.getGpuPtr() != NULL &&
      normalizedGpuImages.getCount() == gpuImages.getCount() &&
      normalizedGpuImages.getBytes() == gpuImages.getBytes() &&
      normalizedGpuImages.getLayout().getDType() == VIEW::F16 &&
      gpuImages.getLayout().getDType() == VIEW::F16;

    ok &= check(normalizeOperandsOk, "Normalize input/output have matching F16 storage");

    if(normalizeOperandsOk)
    {
      OPERATOR::Normalize normalize(workspace, normalizedGpuImages, gpuImages);
      normalize.normByScalar(255.0f);

      cudaError_t normalizeSyncErr = cudaStreamSynchronize(workspace.getStream());
      ok &= check(normalizeSyncErr == cudaSuccess, "Normalize kernel stream synchronizes cleanly");

      io.copyHalfToCpuFloat(normalizedCpuFloatImages, normalizedGpuImages);
      ok &= checkIOSuccess(io, "copied normalized GPU half image batch to CPU float");

      VIEW::Shape& normalizedCpuLayout = normalizedCpuFloatImages.getLayout();
      ok &= check(normalizedCpuLayout.getRank() == 4, "normalized CPU float image rank is NCHW");
      ok &= check(normalizedCpuLayout.getDType() == VIEW::FLOAT, "normalized CPU float image dtype is FLOAT");
      ok &= check(normalizedCpuFloatImages.getCount() == expectedCount, "normalized CPU float image count is correct");
      ok &= check(normalizedCpuFloatImages.getBytes() == expectedFloatBytes, "normalized CPU float image uses float-sized storage");

      float *before = (float *)cpuFloatImages.getCpuPtr();
      float *after = (float *)normalizedCpuFloatImages.getCpuPtr();
      bool normalizedValuesOk = before != NULL && after != NULL;
      for(size_t index = 0; normalizedValuesOk && index < expectedCount; index++)
      {
        float expected = before[index] / 255.0f;
        float diff = fabsf(after[index] - expected);
        if(after[index] < -0.001f || after[index] > 1.001f || diff > 0.0025f)
          normalizedValuesOk = false;
      }

      ok &= check(normalizedValuesOk, "Normalize result matches image / 255 within half precision");
    }
  }
  else
  {
    ok &= check(false, "dataset has at least one image for GPU half pull test");
  }

  ok &= testReluOperator(workspace, io);
  ok &= testSgdOperator(workspace, io);
  ok &= testCrossEntropyOperator(workspace, io);
  ok &= testSoftmaxOperator(workspace, io);
  ok &= testPoolOperator(workspace, io);
  ok &= testConv2DOperator(workspace, io);

  printf("App tests %s\n", ok ? "passed" : "failed");

  return ok ? 0 : 1;
}
