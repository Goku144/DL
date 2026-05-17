#if !defined(OPERATOR_MATRIXMULBIAS_HPP)
#define OPERATOR_MATRIXMULBIAS_HPP

#include "HANDLER/Workspace.hpp"

#include <cublasLt.h>

namespace OPERATOR
{

/**
 * @brief Linear layer operator that computes out = in * weight + bias.
 *
 * MatrixMulBias uses cuBLASLt GEMM with a fused bias epilogue for forward.
 * The public shape contract is in=[N,K], weight=[K,M], bias=[M], out=[N,M].
 * Backward computes dIn, dWeight, and dBias.
 */
class __align__(CORE::ALIGNE_TO_256) MatrixMulBias
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

public:
  /** @brief Create an unbound linear operator. 
   * @param workspace Shared cuBLASLt workspace. 
   * */
  MatrixMulBias(HANDLER::Workspace& workspace);

  /** @brief Create a linear operator and attach forward operands. 
   * @param workspace Shared cuBLASLt workspace. 
   * @param out Output tensor [N,M]. 
   * @param in Input tensor [N,K]. 
   * @param weight Weight tensor [K,M]. 
   * @param bias Bias tensor [M]. 
   * */
  MatrixMulBias(HANDLER::Workspace& workspace, VIEW::Math& out, VIEW::Math& in, VIEW::Math& weight, VIEW::Math& bias);

  /** @brief Destroy the operator. */
  ~MatrixMulBias();

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

  /** @brief Get input gradient. 
   * @return Reference to dInput tensor. 
   * */
  VIEW::Math& getGradInput();

  /** @brief Get weight gradient. 
   * @return Reference to dWeight tensor. 
   * */
  VIEW::Math& getGradWeight();

  /** @brief Get bias gradient. 
   * @return Reference to dBias tensor. 
   * */
  VIEW::Math& getGradBias();

  /** @brief Get output gradient. 
   * @return Reference to dOutput tensor. 
   * */
  VIEW::Math& getGradOutput();

  /** @brief Attach forward operands. 
   * @param out Output tensor [N,M]. 
   * @param in Input tensor [N,K]. 
   * @param weight Weight tensor [K,M]. 
   * @param bias Bias tensor [M]. 
   * */
  void setOperand(VIEW::Math& out, VIEW::Math& in, VIEW::Math& weight, VIEW::Math& bias);

  /** @brief Attach backward operands. 
   * @param dIn Gradient written for input [N,K]. 
   * @param dWeight Gradient written for weight [K,M]. 
   * @param dBias Gradient written for bias [M]. 
   * @param dOut Upstream gradient [N,M]. 
   * */
  void setGradOperand(VIEW::Math& dIn, VIEW::Math& dWeight, VIEW::Math& dBias, VIEW::Math& dOut);

  /** @brief Launch fused linear forward. 
   * 
   * @note Does not set a project error enum directly. 
   * */
  void forward();

  /** @brief Launch linear backward. 
   * 
   * @note Does not set a project error enum directly. 
   * */
  void backward();
};
  
}

#endif /* OPERATOR_MATRIXMULBIAS_HPP */
