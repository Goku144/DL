#include "HANDLER/File.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "CORE/stb_image.h"

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>

static __global__ void convertUint8ToHalfKernel(__half *dst, const uint8_t *src, size_t count)
{
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index >= count) return;

  dst[index] = __float2half((float)src[index]);
}

static char *joinRootPath(const char *rootPath, const char *path)
{
  if(rootPath == NULL) return NULL;

  size_t rootLen = strlen(rootPath);
  size_t pathLen = strlen(path);
  bool needSlash = rootLen > 0 && rootPath[rootLen - 1] != '/';
  char *joined = (char *) malloc(rootLen + needSlash + pathLen + 1);
  if(joined == NULL) return NULL;

  memcpy(joined, rootPath, rootLen);
  if(needSlash) joined[rootLen++] = '/';
  memcpy(joined + rootLen, path, pathLen + 1);
  return joined;
}

static const char *getFileErrorMessage(CORE::errFile err)
{
  if(err == CORE::fileSuccess) return "File success";
  if(err == CORE::fileErrRead) return "File failed to read";
  if(err == CORE::fileErrWrite) return "File failed to write";
  if(err == CORE::fileErrReadCsv) return "File failed to read csv";
  if(err == CORE::fileErrReadImg) return "File failed to read image";
  if(err == CORE::fileErrOpen) return "File failed to open";
  if(err == CORE::fileErrClose) return "File failed to close";
  if(err == CORE::fileErrNull) return "File null pointer";
  if(err == CORE::fileErrInvalidState) return "File invalid state";
  if(err == CORE::fileErrIO) return "File failed because IO layer has an error";
  return "File unknown error";
}

HANDLER::File::File()
{}

HANDLER::File::File(HANDLER::IO& io)
{
  this->io = &io;
}

HANDLER::File::~File()
{}

void HANDLER::File::setIO(HANDLER::IO& io)
{
  this->io = &io;
}

void HANDLER::File::read(VIEW::Math& dst, const char *path, VIEW::DType dtype)
{
  if(path == NULL || this->io == NULL)
  {
    this->err = CORE::fileErrNull;
    return;
  }

  struct stat st;
  if(stat(path, &st) == -1)
  {
    this->err = CORE::fileErrInvalidState;
    return;
  }

  int fd = open(path, O_RDONLY);
  if(fd < 0)
  {
    this->err = CORE::fileErrOpen;
    return;
  }

  size_t fileSize = st.st_size;
  int dims[VIEW::MAX_RANK] = {(int)fileSize, 0, 0, 0};
  VIEW::Shape layout(dims, 1, dtype);
  dst.setLayout(layout);

  io->bindCpu(dst);
  if(io->peekErr() != CORE::ioSuccess)
  {
    this->err = CORE::fileErrIO;
    return;
  };

  size_t offset = 0;
  while (true)
  {
    ssize_t n = ::read(fd, (uint8_t *)dst.getCpuPtr() + offset, fileSize - offset);
    if (n < 0)
    {
      this->err = CORE::fileErrRead;
      return;
    }
    if(n == 0) break;
    offset += n;
  }
  
  if(close(fd) < 0) this->err = CORE::fileErrClose;
}

void HANDLER::File::write(VIEW::Math& src, const char *path)
{
  if(path == NULL || this->io == NULL)
  {
    this->err = CORE::fileErrNull;
    return;
  }

  int fd = open(path, O_RDONLY);
  if(fd < 0)
  {
    this->err = CORE::fileErrOpen;
    return;
  }

  size_t offset = 0, fileSize = src.getCount() * src.getLayout().getDType();
  while (true)
  {
    ssize_t n = ::read(fd, (uint8_t *) src.getCpuPtr() + offset, fileSize - offset);
    if (n < 0)
    {
      this->err = CORE::fileErrWrite;
      return;
    }
    if(n == 0) break;
    offset += n;
  }
  
  if(close(fd) < 0) this->err = CORE::fileErrClose;
}

void HANDLER::File::readCsv(VIEW::Math& filePaths, VIEW::Math& labels, const char *path)
{
  if(path == NULL || this->io == NULL)
  {
    this->err = CORE::fileErrNull;
    return;
  }

  struct stat st;
  if(stat(path, &st) == -1)
  {
    this->err = CORE::fileErrInvalidState;
    return;
  }

  int fd = open(path, O_RDONLY);
  if(fd < 0)
  {
    this->err = CORE::fileErrOpen;
    return;
  }

  size_t offset = 0, fileSize = st.st_size;
  size_t parseIndex = 0, lineStart = 0, column = 0, row = 0;
  bool skipHeader = true;
  uint8_t *ptr = (uint8_t *) malloc(fileSize + 1);
  size_t *lineOffset = (size_t *) malloc(fileSize * sizeof(size_t));
  size_t *lineSize = (size_t *) malloc(fileSize * sizeof(size_t));
  uint8_t *label = (uint8_t *) malloc(fileSize);
  if(ptr == NULL || lineOffset == NULL || lineSize == NULL || label == NULL)
  {
    this->err = CORE::fileErrReadCsv;
    if(ptr != NULL) free(ptr);
    if(lineOffset != NULL) free(lineOffset);
    if(lineSize != NULL) free(lineSize);
    if(label != NULL) free(label);
    close(fd);
    return;
  }

  while (true)
  {
    ssize_t n = ::read(fd, ptr + offset, fileSize - offset);
    if (n < 0)
    {
      this->err = CORE::fileErrRead;
      free(ptr);
      free(lineOffset);
      free(lineSize);
      free(label);
      close(fd);
      return;
    }
    if(n == 0) break;
    offset += n;

    for (size_t index = parseIndex; index < offset; index++)
    {
      if(ptr[index] == '\n')
      {
        if(skipHeader)
        {
          skipHeader = false;
          lineStart = index + 1;
          continue;
        }

        size_t comma = index;
        while(comma > lineStart && ptr[comma] != ',')
          comma--;

        size_t size = comma - lineStart;
        lineOffset[row] = lineStart;
        lineSize[row] = size;
        if(size > column) column = size;
        label[row] = ptr[comma + 1] - '0';
        row++;
        lineStart = index + 1;
      }
    }

    parseIndex = offset;
  }

  ptr[offset] = '\0';
  if(lineStart < offset)
  {
    size_t index = offset;
    if(index > lineStart && ptr[index - 1] == '\r')
      index--;

    if(!skipHeader)
    {
      size_t comma = index;
      while(comma > lineStart && ptr[comma] != ',')
        comma--;

      size_t size = comma - lineStart;
      lineOffset[row] = lineStart;
      lineSize[row] = size;
      if(size > column) column = size;
      label[row] = ptr[comma + 1] - '0';
      row++;
    }
  }

  column += 1;
  int dimsf[VIEW::MAX_RANK] = {(int)row, (int)column, 0, 0};
  int dimsl[VIEW::MAX_RANK] = {(int)row, 0, 0, 0};
  VIEW::Shape layout(dimsf, 2, VIEW::CHAR);
  filePaths.setLayout(layout);
  layout.setShape(dimsl, 1, VIEW::CHAR);
  labels.setLayout(layout);
  io->bindCpu(filePaths);
  if(io->peekErr() != CORE::ioSuccess)
  {
    this->err = CORE::fileErrIO;
    free(ptr);
    free(lineOffset);
    free(lineSize);
    free(label);
    close(fd);
    return;
  }
  io->bindCpu(labels);
  if(io->peekErr() != CORE::ioSuccess)
  {
    this->err = CORE::fileErrIO;
    free(ptr);
    free(lineOffset);
    free(lineSize);
    free(label);
    close(fd);
    return;
  }

  for (size_t i = 0; i < row; i++)
  {
    *((uint8_t *)labels.getCpuDataAt((int)i)) = label[i];
    memcpy(filePaths.getCpuDataAt((int)i), ptr + lineOffset[i], lineSize[i]);
    *((uint8_t *)filePaths.getCpuDataAt((int)i, (int)lineSize[i])) = '\0';
  }
  
  free(ptr);
  free(lineOffset);
  free(lineSize);
  free(label);
  if(close(fd) < 0) this->err = CORE::fileErrClose;
}

void HANDLER::File::readImages(VIEW::Math& filePaths, const char *rootPath, int desiredChannels)
{
  if(this->io == NULL || filePaths.getCpuPtr() == NULL)
  {
    this->err = CORE::fileErrNull;
    return;
  }

  size_t n = filePaths.getLayout().getDim(0);
  if(n == 0)
  {
    this->err = CORE::fileErrInvalidState;
    return;
  }

  char *firstPath = (char *) filePaths.getCpuDataAt(0);
  char *fullPath = rootPath == NULL ? firstPath : joinRootPath(rootPath, firstPath);
  if(fullPath == NULL && rootPath != NULL)
  {
    this->err = CORE::fileErrReadImg;
    return;
  }

  int width = 0, height = 0, channel = 0;
  if(stbi_info(fullPath, &width, &height, &channel) == 0)
  {
    if(rootPath != NULL) free(fullPath);
    this->err = CORE::fileErrReadImg;
    return;
  }
  if(rootPath != NULL) free(fullPath);

  if(desiredChannels > 0)
    channel = desiredChannels;

  int dims[VIEW::MAX_RANK] = {(int)n, channel, height, width};
  VIEW::Shape layout(dims, 4, VIEW::CHAR);
  VIEW::Math images;
  images.setLayout(layout);

  io->bindCpu(images);
  if(io->peekErr() != CORE::ioSuccess)
  {
    this->err = CORE::fileErrIO;
    return;
  }

  size_t imageSize = (size_t) height * width * channel;
  for(size_t index = 0; index < n; index++)
  {
    char *imagePath = (char *) filePaths.getCpuDataAt((int)index);
    fullPath = rootPath == NULL ? imagePath : joinRootPath(rootPath, imagePath);
    if(fullPath == NULL && rootPath != NULL)
    {
      this->err = CORE::fileErrReadImg;
      return;
    }

    int currentWidth = 0, currentHeight = 0, currentChannel = 0;
    uint8_t *image = stbi_load(fullPath, &currentWidth, &currentHeight, &currentChannel, desiredChannels);
    if(rootPath != NULL) free(fullPath);
    if(image == NULL)
    {
      this->err = CORE::fileErrReadImg;
      return;
    }

    if(currentWidth != width || currentHeight != height)
    {
      stbi_image_free(image);
      this->err = CORE::fileErrInvalidState;
      return;
    }

    if(desiredChannels == 0 && currentChannel != channel)
    {
      stbi_image_free(image);
      this->err = CORE::fileErrInvalidState;
      return;
    }

    uint8_t *dstImage = (uint8_t *)images.getCpuPtr() + index * imageSize;
    for(int c = 0; c < channel; c++)
      for(int h = 0; h < height; h++)
        for(int w = 0; w < width; w++)
          dstImage[c * height * width + h * width + w] =
            image[(h * width + w) * channel + c];
    stbi_image_free(image);
  }

  this->imageCpuPtr = images.getCpuPtr();
  this->imageGpuPtr = images.getGpuPtr();
  this->imageCpuOffset = images.getCpuOffset();
  this->imageGpuOffset = images.getGpuOffset();
  this->imageN = n;
  this->imageHeight = height;
  this->imageWidth = width;
  this->imageChannel = channel;
}

void HANDLER::File::copyImageToDevice()
{
  if(this->io == NULL || this->imageCpuPtr == NULL)
  {
    this->err = CORE::fileErrNull;
    return;
  }

  if(this->imageN == 0 || this->imageChannel == 0 || this->imageHeight == 0 || this->imageWidth == 0)
  {
    this->err = CORE::fileErrInvalidState;
    return;
  }

  int dims[VIEW::MAX_RANK] = {(int)this->imageN, this->imageChannel, this->imageHeight, this->imageWidth};
  VIEW::Shape layout(dims, 4, VIEW::F16);
  VIEW::Math images;
  images.setLayout(layout);

  io->bindGpu(images);
  if(io->peekErr() != CORE::ioSuccess)
  {
    this->err = CORE::fileErrIO;
    return;
  }

  size_t count = (size_t)this->imageN * this->imageChannel * this->imageHeight * this->imageWidth;
  uint8_t *deviceU8 = NULL;
  if(cudaMalloc(&deviceU8, count * VIEW::CHAR) != cudaSuccess)
  {
    this->err = CORE::fileErrIO;
    return;
  }

  if(cudaMemcpy(deviceU8, this->imageCpuPtr, count * VIEW::CHAR, cudaMemcpyHostToDevice) != cudaSuccess)
  {
    cudaFree(deviceU8);
    this->err = CORE::fileErrIO;
    return;
  }

  int threads = 256;
  int blocks = (int)((count + threads - 1) / threads);
  convertUint8ToHalfKernel<<<blocks, threads>>>((__half *)images.getGpuPtr(), deviceU8, count);
  if(cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess)
  {
    cudaFree(deviceU8);
    this->err = CORE::fileErrIO;
    return;
  }

  if(cudaFree(deviceU8) != cudaSuccess)
  {
    this->err = CORE::fileErrIO;
    return;
  }

  this->imageGpuPtr = images.getGpuPtr();
  this->imageGpuOffset = images.getGpuOffset();
}

void HANDLER::File::pullCpuImage(VIEW::Math& dst, size_t n, size_t offset)
{
  if(this->io == NULL || this->imageCpuPtr == NULL)
  {
    this->err = CORE::fileErrInvalidState;
    return;
  }

  if(n == 0 || offset + n > this->imageN)
  {
    this->err = CORE::fileErrInvalidState;
    return;
  }

  size_t imageSize = (size_t) this->imageHeight * this->imageWidth * this->imageChannel;
  int dims[VIEW::MAX_RANK] = {(int)n, this->imageChannel, this->imageHeight, this->imageWidth};
  VIEW::Shape layout(dims, 4, VIEW::CHAR);
  dst.setLayout(layout);

  io->bindCpu(dst);
  if(io->peekErr() != CORE::ioSuccess)
  {
    this->err = CORE::fileErrIO;
    return;
  }

  io->copyHostToHost(dst, (uint8_t *)this->imageCpuPtr + offset * imageSize, n * imageSize, VIEW::CHAR);
  if(io->peekErr() != CORE::ioSuccess)
  {
    this->err = CORE::fileErrIO;
    return;
  }

  dst.setLayout(layout);
}

void HANDLER::File::pullGpuImage(VIEW::Math& dst, size_t n, size_t offset)
{
  if(this->io == NULL || this->imageGpuPtr == NULL)
  {
    this->err = CORE::fileErrInvalidState;
    return;
  }

  if(n == 0 || offset + n > this->imageN)
  {
    this->err = CORE::fileErrInvalidState;
    return;
  }

  size_t imageSize = (size_t) this->imageHeight * this->imageWidth * this->imageChannel;
  int dims[VIEW::MAX_RANK] = {(int)n, this->imageChannel, this->imageHeight, this->imageWidth};
  VIEW::Shape layout(dims, 4, VIEW::F16);
  dst.setLayout(layout);

  io->bindGpu(dst);
  if(io->peekErr() != CORE::ioSuccess)
  {
    this->err = CORE::fileErrIO;
    return;
  }

  io->copyDeviceToDevice(dst, (__half *)this->imageGpuPtr + offset * imageSize, n * imageSize, VIEW::F16);
  if(io->peekErr() != CORE::ioSuccess)
  {
    this->err = CORE::fileErrIO;
    return;
  }

  dst.setLayout(layout);
}

size_t HANDLER::File::getImageN() const
{
  return this->imageN;
}

int HANDLER::File::getImageHeight() const
{
  return this->imageHeight;
}

int HANDLER::File::getImageWidth() const
{
  return this->imageWidth;
}

int HANDLER::File::getImageChannel() const
{
  return this->imageChannel;
}

CORE::errFile HANDLER::File::getErr()
{
  CORE::errFile err = this->err;
  this->err = CORE::fileSuccess;
  return err;
}

CORE::errFile HANDLER::File::peekErr() const
{
  return this->err;
}

void HANDLER::File::clearErr()
{
  this->err = CORE::fileSuccess;
}

void HANDLER::File::info(CORE::State level, const char *file, int line) const
{
  if(this->err == CORE::fileSuccess) return;

  if(level == CORE::FATAL)
  {
    CORE::logFatal(file, line, "%s", getFileErrorMessage(this->err));
    return;
  }

  CORE::logWarn(file, line, "%s", getFileErrorMessage(this->err));
}
