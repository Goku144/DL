#if !defined(OPERATOR_SGD_HPP)
#define OPERATOR_SGD_HPP

#include "HANDLER/Workspace.hpp"

namespace OPERATOR
{

/**
 * @brief Stochastic gradient descent update operator.
 *
 * SGD updates an F16 weight tensor in place using weight -= lr * grad. The
 * implementation uses uint4 vectorization, so the element count is expected to
 * be a multiple of 8.
 */
class __align__(CORE::ALIGNE_TO_256) SGD
{
private:
  VIEW::Math *weight;
  VIEW::Math *grad;
  HANDLER::Workspace *workspace;

public:
  /** @brief Create an unbound SGD operator. @param workspace Shared execution workspace. */
  SGD(HANDLER::Workspace& workspace);

  /** @brief Create an SGD operator and attach operands. @param workspace Shared execution workspace. @param weight Weight tensor updated in place. @param grad Gradient tensor read on GPU. */
  SGD(HANDLER::Workspace& workspace, VIEW::Math& weight, VIEW::Math& grad);

  /** @brief Destroy the operator. */
  ~SGD();

  /** @brief Get the weight tensor. @return Reference to the in-place weight tensor. */
  VIEW::Math& getWeight();

  /** @brief Get the gradient tensor. @return Reference to the gradient tensor. */
  VIEW::Math& getGrad();

  /** @brief Attach operands. @param weight Weight tensor updated in place. @param grad Gradient tensor read on GPU. */
  void setOperand(VIEW::Math& weight, VIEW::Math& grad);

  /** @brief Launch weight -= lr * grad. @param lr Learning rate. @note Does not set a project error enum directly. */
  void update(float lr);
};
  
}

#endif /* OPERATOR_SGD_HPP */
