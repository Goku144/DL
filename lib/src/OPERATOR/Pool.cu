#include "OPERATOR/Pool.hpp"

static void helperSetTensor4d(cudnnTensorDescriptor_t desc, VIEW::Math *math)
{
  VIEW::Shape& layout = math->getLayout();
  cudnnSetTensor4dDescriptor(
    desc,
    CUDNN_TENSOR_NCHW,
    CUDNN_DATA_HALF,
    layout.getDim(0),
    layout.getDim(1),
    layout.getDim(2),
    layout.getDim(3));
}

OPERATOR::Pool::Pool(HANDLER::Workspace& workspace)
{
  this->workspace = &workspace;
  this->windowH = 2;
  this->windowW = 2;
  this->padH = 0;
  this->padW = 0;
  this->strideH = 2;
  this->strideW = 2;
  cudnnCreateTensorDescriptor(&this->xDesc);
  cudnnCreateTensorDescriptor(&this->yDesc);
  cudnnCreatePoolingDescriptor(&this->poolDesc);
}

OPERATOR::Pool::Pool(HANDLER::Workspace& workspace, VIEW::Math& out, VIEW::Math& in)
{
  this->workspace = &workspace;
  this->out = &out;
  this->in = &in;
  this->windowH = 2;
  this->windowW = 2;
  this->padH = 0;
  this->padW = 0;
  this->strideH = 2;
  this->strideW = 2;
  cudnnCreateTensorDescriptor(&this->xDesc);
  cudnnCreateTensorDescriptor(&this->yDesc);
  cudnnCreatePoolingDescriptor(&this->poolDesc);
}

OPERATOR::Pool::~Pool()
{
  cudnnDestroyPoolingDescriptor(this->poolDesc);
  cudnnDestroyTensorDescriptor(this->yDesc);
  cudnnDestroyTensorDescriptor(this->xDesc);
}

VIEW::Math& OPERATOR::Pool::getInput()
{
  return *this->in;
}

VIEW::Math& OPERATOR::Pool::getOutput()
{
  return *this->out;
}

VIEW::Math& OPERATOR::Pool::getGradInput()
{
  return *this->dIn;
}

VIEW::Math& OPERATOR::Pool::getGradOutput()
{
  return *this->dOut;
}

void OPERATOR::Pool::setOperand(VIEW::Math& out, VIEW::Math& in)
{
  this->out = &out;
  this->in = &in;
}

void OPERATOR::Pool::setGradOperand(VIEW::Math& dIn, VIEW::Math& dOut)
{
  this->dIn = &dIn;
  this->dOut = &dOut;
}

void OPERATOR::Pool::setConfig(int windowH, int windowW, int padH, int padW, int strideH, int strideW)
{
  this->windowH = windowH;
  this->windowW = windowW;
  this->padH = padH;
  this->padW = padW;
  this->strideH = strideH;
  this->strideW = strideW;
}

void OPERATOR::Pool::setDescriptor()
{
  helperSetTensor4d(this->xDesc, this->in);
  helperSetTensor4d(this->yDesc, this->out);
  cudnnSetPooling2dDescriptor(
    this->poolDesc,
    CUDNN_POOLING_MAX,
    CUDNN_PROPAGATE_NAN,
    this->windowH,
    this->windowW,
    this->padH,
    this->padW,
    this->strideH,
    this->strideW);
}

void OPERATOR::Pool::maxForward()
{
  this->setDescriptor();
  const float alpha = 1.0f;
  const float beta = 0.0f;
  cudnnPoolingForward(
    this->workspace->getCudnnHandle(),
    this->poolDesc,
    &alpha,
    this->xDesc,
    this->in->getGpuPtr(),
    &beta,
    this->yDesc,
    this->out->getGpuPtr());
}

void OPERATOR::Pool::maxBackward()
{
  this->setDescriptor();
  const float alpha = 1.0f;
  const float beta = 0.0f;
  cudnnPoolingBackward(
    this->workspace->getCudnnHandle(),
    this->poolDesc,
    &alpha,
    this->yDesc,
    this->out->getGpuPtr(),
    this->yDesc,
    this->dOut->getGpuPtr(),
    this->xDesc,
    this->in->getGpuPtr(),
    &beta,
    this->xDesc,
    this->dIn->getGpuPtr());
}
