#if !defined(OPERATOR_NORMALIZE_HPP)
#define OPERATOR_NORMALIZE_HPP

#include "HANDLER/Workspace.hpp"

namespace OPERATOR
{

/**
 * @brief Elementwise normalization operator for F16 tensors.
 *
 * Normalize reads an input tensor from GPU memory and writes an output tensor
 * in GPU memory. The current implementation divides each F16 element by a
 * scalar, using vectorized uint4 loads/stores, so the element count is expected
 * to be a multiple of 8.
 */
class __align__(CORE::ALIGNE_TO_256) Normalize
{
private:
  VIEW::Math *in;
  VIEW::Math *out;
  HANDLER::Workspace *workspace;
  
public:
  /**
   * @brief Create an unbound Normalize operator.
   * @param workspace Shared execution workspace with CUDA stream.
   */
  Normalize(HANDLER::Workspace& workspace);

  /**
   * @brief Create a Normalize operator and attach operands.
   * @param workspace Shared execution workspace with CUDA stream.
   * @param out Output F16 tensor written on GPU.
   * @param in Input F16 tensor read on GPU.
   */
  Normalize(HANDLER::Workspace& workspace, VIEW::Math& out, VIEW::Math& in);

  /** @brief Destroy the operator. */
  ~Normalize();

  /** @brief Get the input tensor. 
   * @return Reference to the attached input tensor. 
   * */
  VIEW::Math& getInput();

  /** @brief Get the output tensor. 
   * @return Reference to the attached output tensor. 
   * */
  VIEW::Math& getOutput();

  /**
   * @brief Attach operands.
   * @param out Output F16 tensor with GPU memory already bound.
   * @param in Input F16 tensor with GPU memory and data already bound.
   */
  void setOperand(VIEW::Math& out, VIEW::Math& in);

  /**
   * @brief Compute out = in / scalar.
   * @param scalar Divisor used for normalization.
   *
   * @note Does not set a project error enum directly; CUDA launch errors are
   * observable through stream/device synchronization.
   */
  void normByScalar(float scalar = 255.0f);
};
  
}

#endif /* OPERATOR_NORMALIZE_HPP */
