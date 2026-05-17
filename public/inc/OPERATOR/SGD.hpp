#if !defined(OPERATOR_SGD_HPP)
#define OPERATOR_SGD_HPP

#include "HANDLER/Workspace.hpp"

namespace OPERATOR
{

class __align__(CORE::ALIGNE_TO_256) SGD
{
private:
  VIEW::Math *weight;
  VIEW::Math *grad;
  HANDLER::Workspace *workspace;

public:
  SGD(HANDLER::Workspace& workspace);
  SGD(HANDLER::Workspace& workspace, VIEW::Math& weight, VIEW::Math& grad);
  ~SGD();

  VIEW::Math& getWeight();
  VIEW::Math& getGrad();

  void setOperand(VIEW::Math& weight, VIEW::Math& grad);
  void update(float lr);
};
  
}

#endif /* OPERATOR_SGD_HPP */
