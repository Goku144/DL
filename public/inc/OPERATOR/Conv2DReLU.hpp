#if !defined(OPERATOR_CONV2DRELU_HPP)
#define OPERATOR_CONV2DRELU_HPP

#include "HANDLER/Workspace.hpp"

#include <cudnn.h>

namespace OPERATOR
{

class __align__(CORE::ALIGNE_TO_256) Conv2DRelu
{
private:
  VIEW::Math *in;
  VIEW::Math *weight;
  VIEW::Math *bias;
  VIEW::Math *out;
  VIEW::Math *dIn;
  VIEW::Math *dWeight;
  VIEW::Math *dBias;
  VIEW::Math *dOut;
  HANDLER::Workspace *workspace;

  cudnnTensorDescriptor_t xDesc;
  cudnnFilterDescriptor_t wDesc;
  cudnnTensorDescriptor_t yDesc;
  cudnnTensorDescriptor_t biasDesc;
  cudnnConvolutionDescriptor_t convDesc;
  cudnnActivationDescriptor_t actDesc;

  int padH;
  int padW;
  int strideH;
  int strideW;
  int dilationH;
  int dilationW;

  void setDescriptor();

public:
  Conv2DRelu(HANDLER::Workspace& workspace);
  Conv2DRelu(HANDLER::Workspace& workspace, VIEW::Math& out, VIEW::Math& in, VIEW::Math& weight, VIEW::Math& bias);
  ~Conv2DRelu();

  VIEW::Math& getInput();
  VIEW::Math& getWeight();
  VIEW::Math& getBias();
  VIEW::Math& getOutput();

  void setOperand(VIEW::Math& out, VIEW::Math& in, VIEW::Math& weight, VIEW::Math& bias);
  void setGradOperand(VIEW::Math& dIn, VIEW::Math& dWeight, VIEW::Math& dBias, VIEW::Math& dOut);
  void setConfig(int padH = 1, int padW = 1, int strideH = 1, int strideW = 1, int dilationH = 1, int dilationW = 1);
  void forward();
  void backward();
};
  
}

#endif /* OPERATOR_CONV2DRELU_HPP */
