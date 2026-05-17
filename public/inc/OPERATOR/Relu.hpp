#if !defined(OPERATOR_RELU_HPP)
#define OPERATOR_RELU_HPP

#include "HANDLER/Workspace.hpp"

namespace OPERATOR
{

/**
 * @brief Rectified linear unit operator for F16 tensors.
 *
 * Relu performs forward max(x, 0) and backward gradient gating. Operands must
 * be shaped and bound before execution. The kernels use uint4 vectorization, so
 * element counts are expected to be multiples of 8.
 */
class __align__(CORE::ALIGNE_TO_256) Relu
{
private:
  VIEW::Math *in;
  VIEW::Math *out;
  VIEW::Math *dIn;
  VIEW::Math *dOut;
  HANDLER::Workspace *workspace;

public:
  /** @brief Create an unbound ReLU operator. @param workspace Shared execution workspace. */
  Relu(HANDLER::Workspace& workspace);

  /**
   * @brief Create a ReLU operator and attach forward operands.
   * @param workspace Shared execution workspace.
   * @param out Output F16 tensor written on GPU.
   * @param in Input F16 tensor read on GPU.
   */
  Relu(HANDLER::Workspace& workspace, VIEW::Math& out, VIEW::Math& in);

  /** @brief Destroy the operator. */
  ~Relu();

  /** @brief Get the forward input. @return Reference to input tensor. */
  VIEW::Math& getInput();

  /** @brief Get the forward output. @return Reference to output tensor. */
  VIEW::Math& getOutput();

  /** @brief Get the backward output gradient. @return Reference to dInput tensor. */
  VIEW::Math& getGradInput();

  /** @brief Get the backward input gradient. @return Reference to dOutput tensor. */
  VIEW::Math& getGradOutput();

  /** @brief Attach forward operands. @param out Output tensor. @param in Input tensor. */
  void setOperand(VIEW::Math& out, VIEW::Math& in);

  /** @brief Attach backward operands. @param dIn Gradient written for input. @param dOut Upstream gradient read by the operator. */
  void setGradOperand(VIEW::Math& dIn, VIEW::Math& dOut);

  /** @brief Launch forward ReLU. @note Does not set a project error enum directly. */
  void forward();

  /** @brief Launch backward ReLU. @note Does not set a project error enum directly. */
  void backward();
};
  
}

#endif /* OPERATOR_RELU_HPP */
