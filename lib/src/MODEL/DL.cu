#include "MODEL/DL.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

static constexpr int MODEL_CONV_FILTERS = 16;
static constexpr int MODEL_CONV_KERNEL = 3;
static constexpr int MODEL_POOL_WINDOW = 2;
static constexpr int MODEL_HIDDEN = 128;
static constexpr int MODEL_OUTPUT_CLASSES = 16;
static constexpr int MODEL_REAL_CLASSES = 10;

static void setShape(VIEW::Math& math, int d0, int d1, int d2, int d3, int rank, VIEW::DType dtype)
{
  int dims[VIEW::MAX_RANK] = {d0, d1, d2, d3};
  math.getLayout().setShape(dims, rank, dtype);
}

static void bindGpu(HANDLER::IO *io, VIEW::Math& math, int d0, int d1, int d2, int d3, int rank, VIEW::DType dtype)
{
  setShape(math, d0, d1, d2, d3, rank, dtype);
  io->bindGpu(math);
}

static void bindBoth(HANDLER::IO *io, VIEW::Math& math, int d0, int d1, int d2, int d3, int rank, VIEW::DType dtype)
{
  setShape(math, d0, d1, d2, d3, rank, dtype);
  io->bind(math);
}

static void relayout(VIEW::Math& math, int d0, int d1, int d2, int d3, int rank, VIEW::DType dtype)
{
  setShape(math, d0, d1, d2, d3, rank, dtype);
  math.setCount((size_t)d0 * math.getLayout().getStride(0));
}

static __global__ void initParamKernel(__half *dst, size_t count, float scale, unsigned int seed)
{
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index >= count) return;

  unsigned int x = (unsigned int)index + seed * 747796405u + 2891336453u;
  x = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
  x = (x >> 22u) ^ x;
  float r = ((float)(x & 0xffffu) / 32767.5f) - 1.0f;
  dst[index] = __float2half_rn(r * scale);
}

static __global__ void zeroHalfKernel(__half *dst, size_t count)
{
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index >= count) return;
  dst[index] = __float2half_rn(0.0f);
}

static void initParam(VIEW::Math& math, float scale, unsigned int seed)
{
  int threads = 256;
  int blocks = (int)((math.getCount() + threads - 1) / threads);
  initParamKernel<<<blocks, threads>>>((__half *)math.getGpuPtr(), math.getCount(), scale, seed);
}

static void zeroParam(VIEW::Math& math)
{
  int threads = 256;
  int blocks = (int)((math.getCount() + threads - 1) / threads);
  zeroHalfKernel<<<blocks, threads>>>((__half *)math.getGpuPtr(), math.getCount());
}

MODEL::DL::DL(size_t imageBatch, const char *csvPath)
{
  this->cpu = new HANDLER::Cpu(CORE::MODEL_CPU_MEMORY_DEFAULT);
  this->gpu = new HANDLER::Cuda(CORE::MODEL_GPU_MEMORY_DEFAULT);
  this->io = new HANDLER::IO(*this->cpu, *this->gpu);
  this->file = new HANDLER::File();
  this->workspace = new HANDLER::Workspace(*this->io, *this->file, CORE::MODEL_SCRATCH_MEMORY_DEFAULT);
  this->imageBatch = imageBatch == 0 ? CORE::MODEL_IMAGE_BATCH_DEFAULT : imageBatch;

  this->file->readCsv(this->filePaths, this->Y, csvPath);
  this->io->bindGpu(this->Y);
  this->io->copyHostToDevice(this->Y);
  this->file->readImages(this->filePaths, "public/target/meta", 1);
  this->file->copyImageToDevice();
  
  this->modImage = (size_t) this->Y.getLayout().getDim(0);

  int batch = (int)this->imageBatch;
  int imageC = this->file->getImageChannel();
  int imageH = this->file->getImageHeight();
  int imageW = this->file->getImageWidth();
  int convH = imageH;
  int convW = imageW;
  int poolH = convH / MODEL_POOL_WINDOW;
  int poolW = convW / MODEL_POOL_WINDOW;
  int flat = MODEL_CONV_FILTERS * poolH * poolW;

  bindGpu(this->io, this->x, batch, imageC, imageH, imageW, 4, VIEW::F16);

  bindBoth(this->io, this->w0, MODEL_CONV_FILTERS, imageC, MODEL_CONV_KERNEL, MODEL_CONV_KERNEL, 4, VIEW::F16);
  bindBoth(this->io, this->b0, MODEL_CONV_FILTERS, 0, 0, 0, 1, VIEW::F16);
  bindGpu(this->io, this->z0, batch, MODEL_CONV_FILTERS, convH, convW, 4, VIEW::F16);
  bindGpu(this->io, this->a0, batch, MODEL_CONV_FILTERS, convH, convW, 4, VIEW::F16);
  bindGpu(this->io, this->dz0, batch, MODEL_CONV_FILTERS, convH, convW, 4, VIEW::F16);
  bindGpu(this->io, this->dw0, MODEL_CONV_FILTERS, imageC, MODEL_CONV_KERNEL, MODEL_CONV_KERNEL, 4, VIEW::F16);
  bindGpu(this->io, this->db0, MODEL_CONV_FILTERS, 0, 0, 0, 1, VIEW::F16);

  bindGpu(this->io, this->p, batch, MODEL_CONV_FILTERS, poolH, poolW, 4, VIEW::F16);
  bindGpu(this->io, this->da0, batch, MODEL_CONV_FILTERS, convH, convW, 4, VIEW::F16);

  bindBoth(this->io, this->w1, flat, MODEL_HIDDEN, 0, 0, 2, VIEW::F16);
  bindBoth(this->io, this->b1, MODEL_HIDDEN, 0, 0, 0, 1, VIEW::F16);
  bindGpu(this->io, this->z1, batch, MODEL_HIDDEN, 0, 0, 2, VIEW::F16);
  bindGpu(this->io, this->a1, batch, MODEL_HIDDEN, 0, 0, 2, VIEW::F16);
  bindGpu(this->io, this->dp, batch, MODEL_CONV_FILTERS, poolH, poolW, 4, VIEW::F16);
  bindGpu(this->io, this->dz1, batch, MODEL_HIDDEN, 0, 0, 2, VIEW::F16);
  bindGpu(this->io, this->dw1, flat, MODEL_HIDDEN, 0, 0, 2, VIEW::F16);
  bindGpu(this->io, this->db1, MODEL_HIDDEN, 0, 0, 0, 1, VIEW::F16);

  bindBoth(this->io, this->w2, MODEL_HIDDEN, MODEL_OUTPUT_CLASSES, 0, 0, 2, VIEW::F16);
  bindBoth(this->io, this->b2, MODEL_OUTPUT_CLASSES, 0, 0, 0, 1, VIEW::F16);
  bindGpu(this->io, this->z2, batch, MODEL_OUTPUT_CLASSES, 0, 0, 2, VIEW::F16);
  bindGpu(this->io, this->out, batch, MODEL_OUTPUT_CLASSES, 0, 0, 2, VIEW::F16);
  bindGpu(this->io, this->da1, batch, MODEL_HIDDEN, 0, 0, 2, VIEW::F16);
  bindGpu(this->io, this->dz2, batch, MODEL_OUTPUT_CLASSES, 0, 0, 2, VIEW::F16);
  bindGpu(this->io, this->dw2, MODEL_HIDDEN, MODEL_OUTPUT_CLASSES, 0, 0, 2, VIEW::F16);
  bindGpu(this->io, this->db2, MODEL_OUTPUT_CLASSES, 0, 0, 0, 1, VIEW::F16);

  bindBoth(this->io, this->L, 1, 0, 0, 0, 1, VIEW::FLOAT);
  this->initializeParameters();
}

MODEL::DL::~DL()
{
  delete this->cpu;
  delete this->gpu;
  delete this->io;
  delete this->file;
  delete this->workspace;
}

size_t MODEL::DL::getImageBatch() const
{
  return this->imageBatch;
}

void MODEL::DL::setActiveBatch(size_t batch)
{
  int b = (int)batch;
  int imageC = this->file->getImageChannel();
  int imageH = this->file->getImageHeight();
  int imageW = this->file->getImageWidth();
  int poolH = imageH / MODEL_POOL_WINDOW;
  int poolW = imageW / MODEL_POOL_WINDOW;
  int flat = MODEL_CONV_FILTERS * poolH * poolW;

  relayout(this->x, b, imageC, imageH, imageW, 4, VIEW::F16);
  relayout(this->z0, b, MODEL_CONV_FILTERS, imageH, imageW, 4, VIEW::F16);
  relayout(this->a0, b, MODEL_CONV_FILTERS, imageH, imageW, 4, VIEW::F16);
  relayout(this->dz0, b, MODEL_CONV_FILTERS, imageH, imageW, 4, VIEW::F16);
  relayout(this->p, b, MODEL_CONV_FILTERS, poolH, poolW, 4, VIEW::F16);
  relayout(this->da0, b, MODEL_CONV_FILTERS, imageH, imageW, 4, VIEW::F16);
  relayout(this->z1, b, MODEL_HIDDEN, 0, 0, 2, VIEW::F16);
  relayout(this->a1, b, MODEL_HIDDEN, 0, 0, 2, VIEW::F16);
  relayout(this->dp, b, flat, 0, 0, 2, VIEW::F16);
  relayout(this->dz1, b, MODEL_HIDDEN, 0, 0, 2, VIEW::F16);
  relayout(this->z2, b, MODEL_OUTPUT_CLASSES, 0, 0, 2, VIEW::F16);
  relayout(this->out, b, MODEL_OUTPUT_CLASSES, 0, 0, 2, VIEW::F16);
  relayout(this->da1, b, MODEL_HIDDEN, 0, 0, 2, VIEW::F16);
  relayout(this->dz2, b, MODEL_OUTPUT_CLASSES, 0, 0, 2, VIEW::F16);
}

void MODEL::DL::initializeParameters()
{
  initParam(this->w0, 0.05f, 1);
  zeroParam(this->b0);
  initParam(this->w1, 0.02f, 2);
  zeroParam(this->b1);
  initParam(this->w2, 0.02f, 3);
  zeroParam(this->b2);
  cudaDeviceSynchronize();
}

void MODEL::DL::forward(size_t batchOffset, size_t batch)
{
  this->setActiveBatch(batch);
  this->file->pullGpuImage(this->x, batch, batchOffset);

  OPERATOR::Normalize normalize(*this->workspace, this->x, this->x);
  normalize.normByScalar(255.0f);

  OPERATOR::Conv2DRelu conv(*this->workspace, this->z0, this->x, this->w0, this->b0);
  conv.setConfig(1, 1, 1, 1, 1, 1);
  conv.forward();

  OPERATOR::Relu relu0(*this->workspace, this->a0, this->z0);
  relu0.forward();

  OPERATOR::Pool pool(*this->workspace, this->p, this->a0);
  pool.setConfig(2, 2, 0, 0, 2, 2);
  pool.maxForward();

  int flat = MODEL_CONV_FILTERS * (this->file->getImageHeight() / MODEL_POOL_WINDOW) * (this->file->getImageWidth() / MODEL_POOL_WINDOW);
  relayout(this->p, (int)batch, flat, 0, 0, 2, VIEW::F16);

  OPERATOR::MatrixMulBias fc1(*this->workspace, this->z1, this->p, this->w1, this->b1);
  fc1.forward();

  OPERATOR::Relu relu1(*this->workspace, this->a1, this->z1);
  relu1.forward();

  OPERATOR::MatrixMulBias fc2(*this->workspace, this->z2, this->a1, this->w2, this->b2);
  fc2.forward();

  OPERATOR::Softmax softmax(*this->workspace, this->out, this->z2);
  softmax.forward();
}

void MODEL::DL::backward()
{
  size_t batch = this->out.getLayout().getDim(0);
  int poolH = this->file->getImageHeight() / MODEL_POOL_WINDOW;
  int poolW = this->file->getImageWidth() / MODEL_POOL_WINDOW;
  int flat = MODEL_CONV_FILTERS * poolH * poolW;

  OPERATOR::MatrixMulBias fc2(*this->workspace, this->z2, this->a1, this->w2, this->b2);
  fc2.setGradOperand(this->da1, this->dw2, this->db2, this->dz2);
  fc2.backward();

  OPERATOR::Relu relu1(*this->workspace, this->a1, this->z1);
  relu1.setGradOperand(this->dz1, this->da1);
  relu1.backward();

  relayout(this->p, (int)batch, flat, 0, 0, 2, VIEW::F16);
  relayout(this->dp, (int)batch, flat, 0, 0, 2, VIEW::F16);
  OPERATOR::MatrixMulBias fc1(*this->workspace, this->z1, this->p, this->w1, this->b1);
  fc1.setGradOperand(this->dp, this->dw1, this->db1, this->dz1);
  fc1.backward();

  relayout(this->p, (int)batch, MODEL_CONV_FILTERS, poolH, poolW, 4, VIEW::F16);
  relayout(this->dp, (int)batch, MODEL_CONV_FILTERS, poolH, poolW, 4, VIEW::F16);
  OPERATOR::Pool pool(*this->workspace, this->p, this->a0);
  pool.setConfig(2, 2, 0, 0, 2, 2);
  pool.setGradOperand(this->da0, this->dp);
  pool.maxBackward();

  OPERATOR::Relu relu0(*this->workspace, this->a0, this->z0);
  relu0.setGradOperand(this->dz0, this->da0);
  relu0.backward();

  OPERATOR::Conv2DRelu conv(*this->workspace, this->z0, this->x, this->w0, this->b0);
  conv.setConfig(1, 1, 1, 1, 1, 1);
  conv.setGradOperand(this->x, this->dw0, this->db0, this->dz0);
  conv.backward();
}

void MODEL::DL::update(float learningRate)
{
  OPERATOR::SGD sgdW0(*this->workspace, this->w0, this->dw0);
  OPERATOR::SGD sgdB0(*this->workspace, this->b0, this->db0);
  OPERATOR::SGD sgdW1(*this->workspace, this->w1, this->dw1);
  OPERATOR::SGD sgdB1(*this->workspace, this->b1, this->db1);
  OPERATOR::SGD sgdW2(*this->workspace, this->w2, this->dw2);
  OPERATOR::SGD sgdB2(*this->workspace, this->b2, this->db2);

  sgdW0.update(learningRate);
  sgdB0.update(learningRate);
  sgdW1.update(learningRate);
  sgdB1.update(learningRate);
  sgdW2.update(learningRate);
  sgdB2.update(learningRate);
}

static size_t tensorBytes(VIEW::Math& math)
{
  return math.getCount() * math.getLayout().getDType();
}

static size_t checkpointBytes(VIEW::Math& w0, VIEW::Math& b0, VIEW::Math& w1, VIEW::Math& b1, VIEW::Math& w2, VIEW::Math& b2)
{
  return tensorBytes(w0) + tensorBytes(b0) + tensorBytes(w1) + tensorBytes(b1) + tensorBytes(w2) + tensorBytes(b2);
}

static bool appendTensor(HANDLER::IO *io, VIEW::Math& checkpoint, size_t& offset, VIEW::Math& math)
{
  io->clearErr();
  io->copyDeviceToHost(math);
  if(io->peekErr() != CORE::ioSuccess) return false;

  size_t bytes = tensorBytes(math);
  io->clearErr();
  io->copyHostToHost((uint8_t *)checkpoint.getCpuPtr() + offset, math, math.getCount(), math.getLayout().getDType());
  if(io->peekErr() != CORE::ioSuccess) return false;
  offset += bytes;
  return true;
}

static bool restoreTensor(HANDLER::IO *io, VIEW::Math& checkpoint, size_t& offset, VIEW::Math& math)
{
  size_t bytes = tensorBytes(math);
  if(offset + bytes > checkpoint.getCount()) return false;

  io->clearErr();
  io->copyHostToHost(math, (uint8_t *)checkpoint.getCpuPtr() + offset, math.getCount(), math.getLayout().getDType());
  if(io->peekErr() != CORE::ioSuccess) return false;
  offset += bytes;
  io->clearErr();
  io->copyHostToDevice(math);
  return io->peekErr() == CORE::ioSuccess;
}

void MODEL::DL::saveCheckpoint(const char *path)
{
  mkdir("public/checkpoints", 0755);
  size_t bytes = checkpointBytes(this->w0, this->b0, this->w1, this->b1, this->w2, this->b2);
  int dims[VIEW::MAX_RANK] = {(int)bytes, 0, 0, 0};
  VIEW::Shape layout(dims, 1, VIEW::CHAR);
  VIEW::Math checkpoint(layout);
  this->io->clearErr();
  this->io->bindCpu(checkpoint);
  if(this->io->peekErr() != CORE::ioSuccess)
  {
    CORE::logWarn(__FILE__, __LINE__, "Checkpoint CPU buffer allocation failed: %s", path);
    return;
  }

  size_t offset = 0;
  bool ok = true;
  ok &= appendTensor(this->io, checkpoint, offset, this->w0);
  ok &= appendTensor(this->io, checkpoint, offset, this->b0);
  ok &= appendTensor(this->io, checkpoint, offset, this->w1);
  ok &= appendTensor(this->io, checkpoint, offset, this->b1);
  ok &= appendTensor(this->io, checkpoint, offset, this->w2);
  ok &= appendTensor(this->io, checkpoint, offset, this->b2);

  if(!ok)
  {
    CORE::logWarn(__FILE__, __LINE__, "Checkpoint pack failed: %s", path);
    return;
  }

  this->file->clearErr();
  this->file->write(checkpoint, path);
  if(this->file->peekErr() != CORE::fileSuccess)
  {
    this->file->info(CORE::WARN, __FILE__, __LINE__);
    this->file->clearErr();
    return;
  }

  CORE::logInfo(__FILE__, __LINE__, "Saved checkpoint: %s", path);
}

bool MODEL::DL::loadCheckpoint(const char *path)
{
  if(path == NULL) return false;

  VIEW::Math checkpoint;
  this->file->clearErr();
  this->file->read(checkpoint, path, VIEW::CHAR);
  if(this->file->peekErr() != CORE::fileSuccess)
  {
    this->file->info(CORE::WARN, __FILE__, __LINE__);
    this->file->clearErr();
    return false;
  }

  size_t expectedBytes = checkpointBytes(this->w0, this->b0, this->w1, this->b1, this->w2, this->b2);
  if(checkpoint.getCount() != expectedBytes)
  {
    CORE::logWarn(__FILE__, __LINE__, "Checkpoint size mismatch: %s has %zu bytes, expected %zu", path, checkpoint.getCount(), expectedBytes);
    return false;
  }

  size_t offset = 0;
  bool ok = true;
  ok &= restoreTensor(this->io, checkpoint, offset, this->w0);
  ok &= restoreTensor(this->io, checkpoint, offset, this->b0);
  ok &= restoreTensor(this->io, checkpoint, offset, this->w1);
  ok &= restoreTensor(this->io, checkpoint, offset, this->b1);
  ok &= restoreTensor(this->io, checkpoint, offset, this->w2);
  ok &= restoreTensor(this->io, checkpoint, offset, this->b2);

  if(ok) {CORE::logInfo(__FILE__, __LINE__, "Loaded checkpoint: %s", path);}
  else {CORE::logWarn(__FILE__, __LINE__, "Checkpoint restore failed: %s", path);}
  return ok;
}

void MODEL::DL::train(size_t iterations, float learningRate, size_t checkpointEvery, const char *checkpointPrefix)
{
  if(this->modImage == 0 || iterations == 0) return;

  for(size_t iter = 0; iter < iterations; iter++)
  {
    size_t batch = this->imageBatch;
    if(batch > this->modImage) batch = this->modImage;

    size_t offset = this->imageOffset % this->modImage;
    if(offset + batch > this->modImage) offset = 0;

    this->forward(offset, batch);

    OPERATOR::CrossEntropy lossOp(*this->workspace, this->dz2, this->L, this->out, this->Y);
    lossOp.setTargetBatch((int)batch, (int)offset);
    lossOp.forwardBackward();

    this->backward();
    this->update(learningRate);
    cudaStreamSynchronize(this->workspace->getStream());

    bool shouldReport = checkpointEvery > 0 && ((iter + 1) % checkpointEvery == 0);
    if(shouldReport || iter + 1 == iterations)
    {
      VIEW::Math outCpu;
      this->io->copyHalfToCpuFloat(outCpu, this->out);
      this->io->copyDeviceToHost(this->L);

      float *prob = (float *)outCpu.getCpuPtr();
      uint8_t *labels = (uint8_t *)this->Y.getCpuPtr();
      size_t correct = 0;
      float confidenceSum = 0.0f;

      for(size_t row = 0; row < batch; row++)
      {
        int pred = 0;
        float best = prob[row * MODEL_OUTPUT_CLASSES];
        for(int cls = 1; cls < MODEL_REAL_CLASSES; cls++)
        {
          float p = prob[row * MODEL_OUTPUT_CLASSES + cls];
          if(p > best)
          {
            best = p;
            pred = cls;
          }
        }
        confidenceSum += best;
        if(pred == labels[offset + row]) correct++;
      }

      float accuracy = batch == 0 ? 0.0f : (float)correct / (float)batch;
      float confidence = batch == 0 ? 0.0f : confidenceSum / (float)batch;
      CORE::logInfo(__FILE__, __LINE__, "TRAIN iter=%zu/%zu loss=%f accuracy=%0.2f%% confidence=%0.2f%% offset=%zu lr=%g",
        iter + 1,
        iterations,
        *(float *)this->L.getCpuPtr(),
        accuracy * 100.0f,
        confidence * 100.0f,
        offset,
        learningRate);

      if(checkpointEvery > 0 && (iter + 1) % checkpointEvery == 0)
      {
        char path[512];
        snprintf(path, sizeof(path), "%s_iter_%zu.bin", checkpointPrefix, iter + 1);
        this->saveCheckpoint(path);
      }
    }

    this->imageOffset = (offset + batch) % this->modImage;
  }
}

void MODEL::DL::estimate(const char *imagePath, const char *checkpointPath)
{
  if(imagePath == NULL)
  {
    CORE::logWarn(__FILE__, __LINE__, "Estimate image path is null");
    return;
  }
  if(checkpointPath != NULL) this->loadCheckpoint(checkpointPath);

  size_t pathLen = strlen(imagePath) + 1;
  int pathDims[VIEW::MAX_RANK] = {1, (int)pathLen, 0, 0};
  VIEW::Shape pathLayout(pathDims, 2, VIEW::CHAR);
  VIEW::Math pathMath;
  pathMath.setLayout(pathLayout);
  this->io->clearErr();
  this->io->bindCpu(pathMath);
  if(this->io->peekErr() != CORE::ioSuccess)
  {
    CORE::logWarn(__FILE__, __LINE__, "Estimate path buffer allocation failed");
    return;
  }

  this->io->clearErr();
  this->io->copyHostToHost(pathMath, (void *)imagePath, pathLen, VIEW::CHAR);
  if(this->io->peekErr() != CORE::ioSuccess)
  {
    CORE::logWarn(__FILE__, __LINE__, "Estimate path copy failed");
    return;
  }

  this->file->readImages(pathMath, NULL, 1);
  this->file->copyImageToDevice();
  this->forward(0, 1);
  cudaStreamSynchronize(this->workspace->getStream());

  VIEW::Math outCpu;
  this->io->copyHalfToCpuFloat(outCpu, this->out);
  float *prob = (float *)outCpu.getCpuPtr();

  int pred = 0;
  float best = prob[0];
  CORE::logInfo(__FILE__, __LINE__, "ESTIMATE %s", imagePath);
  for(int cls = 0; cls < MODEL_REAL_CLASSES; cls++)
  {
    float p = prob[cls];
    if(p > best)
    {
      best = p;
      pred = cls;
    }
    CORE::logInfo(__FILE__, __LINE__, "class %d: %0.2f%%", cls, p * 100.0f);
  }
  CORE::logInfo(__FILE__, __LINE__, "answer: %d confidence=%0.2f%%", pred, best * 100.0f);
}
