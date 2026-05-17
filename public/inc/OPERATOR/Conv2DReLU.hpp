#if !defined(OPERATOR_CONV2DRELU_HPP)
#define OPERATOR_CONV2DRELU_HPP

#include "HANDLER/Workspace.hpp"

#include <cudnn.h>

namespace OPERATOR
{

/**
 * @brief cuDNN 2D convolution plus bias operator for NCHW F16 tensors.
 *
 * Despite the class name, the current forward path performs convolution and
 * bias add only; it creates a ReLU descriptor but does not apply activation.
 * Backward computes gradients for input, filter, and bias.
 */
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
  /** @brief Create an unbound convolution operator. 
   * @param workspace Shared cuDNN workspace. 
   * */
  Conv2DRelu(HANDLER::Workspace& workspace);

  /** @brief Create a convolution operator and attach forward operands. 
   * @param workspace Shared cuDNN workspace. 
   * @param out Output tensor NCHW. 
   * @param in Input tensor NCHW. 
   * @param weight Filter tensor KCRS. 
   * @param bias Bias tensor [K]. 
   * */
  Conv2DRelu(HANDLER::Workspace& workspace, VIEW::Math& out, VIEW::Math& in, VIEW::Math& weight, VIEW::Math& bias);

  /** @brief Destroy cuDNN descriptors. */
  ~Conv2DRelu();

  /** @brief Get input tensor. 
   * @return Reference to input tensor. 
   * */
  VIEW::Math& getInput();

  /** @brief Get weight tensor. 
   * @return Reference to weight tensor. 
   * */
  VIEW::Math& getWeight();

  /** @brief Get bias tensor. 
   * @return Reference to bias tensor. 
   * */
  VIEW::Math& getBias();

  /** @brief Get output tensor. 
   * @return Reference to output tensor. 
   * */
  VIEW::Math& getOutput();

  /** @brief Attach forward operands. 
   * @param out Output tensor. 
   * @param in Input tensor. 
   * @param weight Filter tensor. 
   * @param bias Bias tensor. 
   * */
  void setOperand(VIEW::Math& out, VIEW::Math& in, VIEW::Math& weight, VIEW::Math& bias);

  /** @brief Attach backward operands. 
   * @param dIn Gradient written for input. 
   * @param dWeight Gradient written for weight. 
   * @param dBias Gradient written for bias. 
   * @param dOut Upstream gradient read for output. 
   * */
  void setGradOperand(VIEW::Math& dIn, VIEW::Math& dWeight, VIEW::Math& dBias, VIEW::Math& dOut);

  /** @brief Configure convolution. 
   * @param padH Height padding. 
   * @param padW Width padding. 
   * @param strideH Height stride. 
   * @param strideW Width stride. 
   * @param dilationH Height dilation. 
   * @param dilationW Width dilation. 
   * */
  void setConfig(int padH = 1, int padW = 1, int strideH = 1, int strideW = 1, int dilationH = 1, int dilationW = 1);

  /** @brief Launch convolution plus bias forward. 
   * 
   * @note Does not set a project error enum directly. 
   * */
  void forward();

  /** @brief Launch convolution backward gradients. 
   * 
   * @note Does not set a project error enum directly. 
   * */
  void backward();
};
  
}

#endif /* OPERATOR_CONV2DRELU_HPP */
