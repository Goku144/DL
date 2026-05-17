#include "OPERATOR/Softmax.hpp"

static void helperSetSoftmaxTensor(cudnnTensorDescriptor_t desc, VIEW::Math *math)
{
  VIEW::Shape& layout = math->getLayout();
  int n = layout.getRank() > 1 ? layout.getDim(0) : 1;
  int c = layout.getRank() > 1 ? layout.getDim(1) : layout.getDim(0);
  cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, n, c, 1, 1);
}

OPERATOR::Softmax::Softmax(HANDLER::Workspace& workspace)
{
  this->workspace = &workspace;
  cudnnCreateTensorDescriptor(&this->xDesc);
  cudnnCreateTensorDescriptor(&this->yDesc);
}

OPERATOR::Softmax::Softmax(HANDLER::Workspace& workspace, VIEW::Math& out, VIEW::Math& in)
{
  this->workspace = &workspace;
  this->out = &out;
  this->in = &in;
  cudnnCreateTensorDescriptor(&this->xDesc);
  cudnnCreateTensorDescriptor(&this->yDesc);
}

OPERATOR::Softmax::~Softmax()
{
  cudnnDestroyTensorDescriptor(this->yDesc);
  cudnnDestroyTensorDescriptor(this->xDesc);
}

VIEW::Math& OPERATOR::Softmax::getInput()
{
  return *this->in;
}

VIEW::Math& OPERATOR::Softmax::getOutput()
{
  return *this->out;
}

VIEW::Math& OPERATOR::Softmax::getGradInput()
{
  return *this->dIn;
}

VIEW::Math& OPERATOR::Softmax::getGradOutput()
{
  return *this->dOut;
}

void OPERATOR::Softmax::setOperand(VIEW::Math& out, VIEW::Math& in)
{
  this->out = &out;
  this->in = &in;
}

void OPERATOR::Softmax::setGradOperand(VIEW::Math& dIn, VIEW::Math& dOut)
{
  this->dIn = &dIn;
  this->dOut = &dOut;
}

void OPERATOR::Softmax::setDescriptor()
{
  helperSetSoftmaxTensor(this->xDesc, this->in);
  helperSetSoftmaxTensor(this->yDesc, this->out);
}

void OPERATOR::Softmax::forward()
{
  this->setDescriptor();
  const float alpha = 1.0f;
  const float beta = 0.0f;
  cudnnSoftmaxForward(
    this->workspace->getCudnnHandle(),
    CUDNN_SOFTMAX_ACCURATE,
    CUDNN_SOFTMAX_MODE_INSTANCE,
    &alpha,
    this->xDesc,
    this->in->getGpuPtr(),
    &beta,
    this->yDesc,
    this->out->getGpuPtr());
}

void OPERATOR::Softmax::backward()
{
  this->setDescriptor();
  const float alpha = 1.0f;
  const float beta = 0.0f;
  cudnnSoftmaxBackward(
    this->workspace->getCudnnHandle(),
    CUDNN_SOFTMAX_ACCURATE,
    CUDNN_SOFTMAX_MODE_INSTANCE,
    &alpha,
    this->yDesc,
    this->out->getGpuPtr(),
    this->yDesc,
    this->dOut->getGpuPtr(),
    &beta,
    this->xDesc,
    this->dIn->getGpuPtr());
}
