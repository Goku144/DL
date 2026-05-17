#if !defined(OPERATOR_SOFTMAX_HPP)
#define OPERATOR_SOFTMAX_HPP

#include "HANDLER/Workspace.hpp"

#include <cudnn.h>

namespace OPERATOR
{

/**
 * @brief cuDNN softmax operator for F16 logits/probabilities.
 *
 * Softmax treats rank-2 tensors as [batch, classes] and rank-1 tensors as a
 * single row. It writes probabilities in forward and propagates gradients in
 * backward using cuDNN.
 */
class __align__(CORE::ALIGNE_TO_256) Softmax
{
private:
  VIEW::Math *in;
  VIEW::Math *out;
  VIEW::Math *dIn;
  VIEW::Math *dOut;
  HANDLER::Workspace *workspace;

  cudnnTensorDescriptor_t xDesc;
  cudnnTensorDescriptor_t yDesc;

  void setDescriptor();

public:
  /** @brief Create an unbound Softmax operator. 
   * @param workspace Shared cuDNN workspace. 
   * */
  Softmax(HANDLER::Workspace& workspace);

  /** @brief Create a Softmax operator and attach forward operands. 
   * @param workspace Shared cuDNN workspace. 
   * @param out Output probability tensor. 
   * @param in Input logits tensor. 
   * */
  Softmax(HANDLER::Workspace& workspace, VIEW::Math& out, VIEW::Math& in);

  /** @brief Destroy cuDNN descriptors. */
  ~Softmax();

  /** @brief Get the logits tensor. 
   * @return Reference to input tensor. 
   * */
  VIEW::Math& getInput();

  /** @brief Get the probability tensor. 
   * @return Reference to output tensor. 
   * */
  VIEW::Math& getOutput();

  /** @brief Get gradient with respect to logits. 
   * @return Reference to dInput tensor. 
   * */
  VIEW::Math& getGradInput();

  /** @brief Get upstream gradient with respect to probabilities. 
   * @return Reference to dOutput tensor. 
   * */
  VIEW::Math& getGradOutput();

  /** @brief Attach forward operands. 
   * @param out Output probability tensor. 
   * @param in Input logits tensor. 
   * */
  void setOperand(VIEW::Math& out, VIEW::Math& in);

  /** @brief Attach backward operands. 
   * @param dIn Gradient written for logits. 
   * @param dOut Upstream gradient read for probabilities. 
   * */
  void setGradOperand(VIEW::Math& dIn, VIEW::Math& dOut);

  /** @brief Launch cuDNN softmax forward. 
   * 
   * @note Does not set a project error enum directly. 
   * */
  void forward();

  /** @brief Launch cuDNN softmax backward. 
   * 
   * @note Does not set a project error enum directly. 
   * */
  void backward();
};
  
}

#endif /* OPERATOR_SOFTMAX_HPP */
