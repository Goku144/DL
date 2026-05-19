#if !defined(OPERATOR_CROSSENTROPY_HPP)
#define OPERATOR_CROSSENTROPY_HPP

#include "HANDLER/Workspace.hpp"

namespace OPERATOR
{

/**
 * @brief Fused cross-entropy loss and probability gradient.
 *
 * CrossEntropy expects probability rows and class-index targets. It writes a
 * single FLOAT loss and an F16 dProb tensor. The fastest path is optimized for
 * 16 output columns, which can be used as 10 real digit classes plus 6 padding
 * columns.
 */
class __align__(CORE::ALIGNE_TO_256) CrossEntropy
{
private:
  VIEW::Math *prob;
  VIEW::Math *target;
  VIEW::Math *loss;
  VIEW::Math *dProb;
  HANDLER::Workspace *workspace;
  int targetBatchSize = 0;
  int targetOffset = 0;

public:
  /** @brief Create an unbound CrossEntropy operator. 
   * @param workspace Shared execution workspace. 
   * */
  CrossEntropy(HANDLER::Workspace& workspace);

  /** @brief Create a CrossEntropy operator and attach operands. 
   * @param workspace Shared execution workspace. 
   * @param dProb Output probability gradient. 
   * @param loss Output scalar loss. 
   * @param prob Input probabilities. 
   * @param target Input class labels as CHAR bytes. 
   * */
  CrossEntropy(HANDLER::Workspace& workspace, VIEW::Math& dProb, VIEW::Math& loss, VIEW::Math& prob, VIEW::Math& target);

  /** @brief Destroy the operator. */
  ~CrossEntropy();

  /** @brief Get probabilities. 
   * @return Reference to probability tensor. 
   * */
  VIEW::Math& getProb();

  /** @brief Get targets. 
   * @return Reference to target tensor. 
   * */
  VIEW::Math& getTarget();

  /** @brief Get loss. 
   * @return Reference to scalar loss tensor. 
   * */
  VIEW::Math& getLoss();

  /** @brief Get probability gradient. 
   * @return Reference to dProb tensor. 
   * */
  VIEW::Math& getGradProb();

  /** @brief Attach operands. 
   * @param dProb Output probability gradient. 
   * @param loss Output scalar loss. 
   * @param prob Input probabilities. 
   * @param target Input class labels. 
   * */
  void setOperand(VIEW::Math& dProb, VIEW::Math& loss, VIEW::Math& prob, VIEW::Math& target);

  /** @brief Select how many target rows to read and where to start.
   * @param batchSize Number of batch rows to process. Use 0 to follow prob rows.
   * @param offset Starting label index inside the full target tensor.
   * */
  void setTargetBatch(int batchSize, int offset);

  /** @brief Launch loss and dProb computation. 
   * 
   * @note Does not set a project error enum directly. 
   * */
  void forwardBackward();
};

}

#endif /* OPERATOR_CROSSENTROPY_HPP */
