#if !defined(OPERATOR_CROSSENTROPY_HPP)
#define OPERATOR_CROSSENTROPY_HPP

#include "HANDLER/Workspace.hpp"

namespace OPERATOR
{

class __align__(CORE::ALIGNE_TO_256) CrossEntropy
{
private:
  VIEW::Math *prob;
  VIEW::Math *target;
  VIEW::Math *loss;
  VIEW::Math *dProb;
  HANDLER::Workspace *workspace;

public:
  CrossEntropy(HANDLER::Workspace& workspace);
  CrossEntropy(HANDLER::Workspace& workspace, VIEW::Math& dProb, VIEW::Math& loss, VIEW::Math& prob, VIEW::Math& target);
  ~CrossEntropy();

  VIEW::Math& getProb();
  VIEW::Math& getTarget();
  VIEW::Math& getLoss();
  VIEW::Math& getGradProb();

  void setOperand(VIEW::Math& dProb, VIEW::Math& loss, VIEW::Math& prob, VIEW::Math& target);
  void forwardBackward();
};

}

#endif /* OPERATOR_CROSSENTROPY_HPP */
