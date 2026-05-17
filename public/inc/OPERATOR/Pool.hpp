#if !defined(OPERATOR_POOL_HPP)
#define OPERATOR_POOL_HPP

#include "HANDLER/Workspace.hpp"

#include <cudnn.h>

namespace OPERATOR
{

/**
 * @brief cuDNN max-pooling operator for NCHW F16 tensors.
 *
 * Pool owns cuDNN tensor/pooling descriptors and supports max-pooling forward
 * and backward. Input/output tensors must already be shaped and bound.
 */
class __align__(CORE::ALIGNE_TO_256) Pool
{
private:
  VIEW::Math *in;
  VIEW::Math *out;
  VIEW::Math *dIn;
  VIEW::Math *dOut;
  HANDLER::Workspace *workspace;

  cudnnTensorDescriptor_t xDesc;
  cudnnTensorDescriptor_t yDesc;
  cudnnPoolingDescriptor_t poolDesc;

  int windowH;
  int windowW;
  int padH;
  int padW;
  int strideH;
  int strideW;

  void setDescriptor();

public:
  /** @brief Create an unbound Pool operator. 
   * @param workspace Shared cuDNN workspace. 
   * */
  Pool(HANDLER::Workspace& workspace);

  /** @brief Create a Pool operator and attach forward operands. 
   * @param workspace Shared cuDNN workspace. 
   * @param out Output tensor. 
   * @param in Input tensor. 
   * */
  Pool(HANDLER::Workspace& workspace, VIEW::Math& out, VIEW::Math& in);

  /** @brief Destroy cuDNN descriptors. */
  ~Pool();

  /** @brief Get input tensor. 
   * @return Reference to input tensor. 
   * */
  VIEW::Math& getInput();

  /** @brief Get output tensor. 
   * @return Reference to output tensor. 
   * */
  VIEW::Math& getOutput();

  /** @brief Get input gradient tensor. 
   * @return Reference to dInput tensor. 
   * */
  VIEW::Math& getGradInput();

  /** @brief Get output gradient tensor. 
   * @return Reference to dOutput tensor. 
   * */
  VIEW::Math& getGradOutput();

  /** @brief Attach forward operands. 
   * @param out Output tensor. 
   * @param in Input tensor. 
   * */
  void setOperand(VIEW::Math& out, VIEW::Math& in);

  /** @brief Attach backward operands. 
   * @param dIn Gradient written for input. 
   * @param dOut Upstream gradient read for output. 
   * */
  void setGradOperand(VIEW::Math& dIn, VIEW::Math& dOut);

  /** @brief Configure 2D pooling. 
   * @param windowH Window height. 
   * @param windowW Window width. 
   * @param padH Top/bottom padding. 
   * @param padW Left/right padding. 
   * @param strideH Vertical stride. 
   * @param strideW Horizontal stride. 
   * */
  void setConfig(int windowH = 2, int windowW = 2, int padH = 0, int padW = 0, int strideH = 2, int strideW = 2);

  /** @brief Launch max-pooling forward. 
   * 
   * @note Does not set a project error enum directly. 
   * */
  void maxForward();

  /** @brief Launch max-pooling backward. 
   * 
   * @note Does not set a project error enum directly. 
   * */
  void maxBackward();
};
  
}

#endif /* OPERATOR_POOL_HPP */
