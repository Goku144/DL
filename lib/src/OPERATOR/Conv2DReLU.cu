#include "OPERATOR/Conv2DReLU.hpp"

OPERATOR::Conv2DRelu::Conv2DRelu(HANDLER::Workspace& workspace)
{
  this->workspace = &workspace;
  this->padH = 1;
  this->padW = 1;
  this->strideH = 1;
  this->strideW = 1;
  this->dilationH = 1;
  this->dilationW = 1;
  cudnnCreateTensorDescriptor(&this->xDesc);
  cudnnCreateFilterDescriptor(&this->wDesc);
  cudnnCreateTensorDescriptor(&this->yDesc);
  cudnnCreateTensorDescriptor(&this->biasDesc);
  cudnnCreateConvolutionDescriptor(&this->convDesc);
  cudnnCreateActivationDescriptor(&this->actDesc);
}

OPERATOR::Conv2DRelu::Conv2DRelu(HANDLER::Workspace& workspace, VIEW::Math& out, VIEW::Math& in, VIEW::Math& weight, VIEW::Math& bias)
{
  this->workspace = &workspace;
  this->out = &out;
  this->in = &in;
  this->weight = &weight;
  this->bias = &bias;
  this->padH = 1;
  this->padW = 1;
  this->strideH = 1;
  this->strideW = 1;
  this->dilationH = 1;
  this->dilationW = 1;
  cudnnCreateTensorDescriptor(&this->xDesc);
  cudnnCreateFilterDescriptor(&this->wDesc);
  cudnnCreateTensorDescriptor(&this->yDesc);
  cudnnCreateTensorDescriptor(&this->biasDesc);
  cudnnCreateConvolutionDescriptor(&this->convDesc);
  cudnnCreateActivationDescriptor(&this->actDesc);
}

OPERATOR::Conv2DRelu::~Conv2DRelu()
{
  cudnnDestroyActivationDescriptor(this->actDesc);
  cudnnDestroyConvolutionDescriptor(this->convDesc);
  cudnnDestroyTensorDescriptor(this->biasDesc);
  cudnnDestroyTensorDescriptor(this->yDesc);
  cudnnDestroyFilterDescriptor(this->wDesc);
  cudnnDestroyTensorDescriptor(this->xDesc);
}

VIEW::Math& OPERATOR::Conv2DRelu::getInput()
{
  return *this->in;
}

VIEW::Math& OPERATOR::Conv2DRelu::getWeight()
{
  return *this->weight;
}

VIEW::Math& OPERATOR::Conv2DRelu::getBias()
{
  return *this->bias;
}

VIEW::Math& OPERATOR::Conv2DRelu::getOutput()
{
  return *this->out;
}

void OPERATOR::Conv2DRelu::setOperand(VIEW::Math& out, VIEW::Math& in, VIEW::Math& weight, VIEW::Math& bias)
{
  this->out = &out;
  this->in = &in;
  this->weight = &weight;
  this->bias = &bias;
}

void OPERATOR::Conv2DRelu::setGradOperand(VIEW::Math& dIn, VIEW::Math& dWeight, VIEW::Math& dBias, VIEW::Math& dOut)
{
  this->dIn = &dIn;
  this->dWeight = &dWeight;
  this->dBias = &dBias;
  this->dOut = &dOut;
}

void OPERATOR::Conv2DRelu::setConfig(int padH, int padW, int strideH, int strideW, int dilationH, int dilationW)
{
  this->padH = padH;
  this->padW = padW;
  this->strideH = strideH;
  this->strideW = strideW;
  this->dilationH = dilationH;
  this->dilationW = dilationW;
}

void OPERATOR::Conv2DRelu::setDescriptor()
{
  VIEW::Shape& x = this->in->getLayout();
  VIEW::Shape& w = this->weight->getLayout();
  VIEW::Shape& y = this->out->getLayout();
  VIEW::Shape& b = this->bias->getLayout();

  cudnnSetTensor4dDescriptor(this->xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, x.getDim(0), x.getDim(1), x.getDim(2), x.getDim(3));
  cudnnSetFilter4dDescriptor(this->wDesc, CUDNN_DATA_HALF, CUDNN_TENSOR_NCHW, w.getDim(0), w.getDim(1), w.getDim(2), w.getDim(3));
  cudnnSetTensor4dDescriptor(this->yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, y.getDim(0), y.getDim(1), y.getDim(2), y.getDim(3));
  cudnnSetTensor4dDescriptor(this->biasDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, 1, b.getDim(0), 1, 1);
  cudnnSetConvolution2dDescriptor(this->convDesc, this->padH, this->padW, this->strideH, this->strideW, this->dilationH, this->dilationW, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
  cudnnSetActivationDescriptor(this->actDesc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0);
}

void OPERATOR::Conv2DRelu::forward()
{
  this->setDescriptor();

  const float alpha = 1.0f;
  const float beta = 0.0f;
  const float biasBeta = 1.0f;
  size_t workspaceBytes = 0;
  cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;

  cudnnGetConvolutionForwardWorkspaceSize(
    this->workspace->getCudnnHandle(),
    this->xDesc,
    this->wDesc,
    this->convDesc,
    this->yDesc,
    algo,
    &workspaceBytes);

  void *scratch = this->workspace->getScratch(workspaceBytes);
  cudnnConvolutionForward(
    this->workspace->getCudnnHandle(),
    &alpha,
    this->xDesc,
    this->in->getGpuPtr(),
    this->wDesc,
    this->weight->getGpuPtr(),
    this->convDesc,
    algo,
    scratch,
    workspaceBytes,
    &beta,
    this->yDesc,
    this->out->getGpuPtr());

  cudnnAddTensor(
    this->workspace->getCudnnHandle(),
    &alpha,
    this->biasDesc,
    this->bias->getGpuPtr(),
    &biasBeta,
    this->yDesc,
    this->out->getGpuPtr());
}

void OPERATOR::Conv2DRelu::backward()
{
  this->setDescriptor();

  const float alpha = 1.0f;
  const float beta = 0.0f;
  cudnnConvolutionBwdDataAlgo_t dataAlgo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
  cudnnConvolutionBwdFilterAlgo_t filterAlgo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
  size_t dataWorkspaceBytes = 0;
  size_t filterWorkspaceBytes = 0;

  cudnnConvolutionBackwardBias(
    this->workspace->getCudnnHandle(),
    &alpha,
    this->yDesc,
    this->dOut->getGpuPtr(),
    &beta,
    this->biasDesc,
    this->dBias->getGpuPtr());

  cudnnGetConvolutionBackwardFilterWorkspaceSize(
    this->workspace->getCudnnHandle(),
    this->xDesc,
    this->yDesc,
    this->convDesc,
    this->wDesc,
    filterAlgo,
    &filterWorkspaceBytes);

  cudnnConvolutionBackwardFilter(
    this->workspace->getCudnnHandle(),
    &alpha,
    this->xDesc,
    this->in->getGpuPtr(),
    this->yDesc,
    this->dOut->getGpuPtr(),
    this->convDesc,
    filterAlgo,
    this->workspace->getScratch(filterWorkspaceBytes),
    filterWorkspaceBytes,
    &beta,
    this->wDesc,
    this->dWeight->getGpuPtr());

  cudnnGetConvolutionBackwardDataWorkspaceSize(
    this->workspace->getCudnnHandle(),
    this->wDesc,
    this->yDesc,
    this->convDesc,
    this->xDesc,
    dataAlgo,
    &dataWorkspaceBytes);

  cudnnConvolutionBackwardData(
    this->workspace->getCudnnHandle(),
    &alpha,
    this->wDesc,
    this->weight->getGpuPtr(),
    this->yDesc,
    this->dOut->getGpuPtr(),
    this->convDesc,
    dataAlgo,
    this->workspace->getScratch(dataWorkspaceBytes),
    dataWorkspaceBytes,
    &beta,
    this->xDesc,
    this->dIn->getGpuPtr());
}
