#if !defined(OPERATOR_RELU_HPP)
#define OPERATOR_RELU_HPP

#include "HANDLER/Workspace.hpp"

namespace OPERATOR
{

class __align__(CORE::ALIGNE_TO_256) Relu
{
private:
  VIEW::Math *in;
  VIEW::Math *out;
  VIEW::Math *dIn;
  VIEW::Math *dOut;
  HANDLER::Workspace *workspace;

public:
  Relu(HANDLER::Workspace& workspace);
  Relu(HANDLER::Workspace& workspace, VIEW::Math& out, VIEW::Math& in);
  ~Relu();

  VIEW::Math& getInput();
  VIEW::Math& getOutput();
  VIEW::Math& getGradInput();
  VIEW::Math& getGradOutput();

  void setOperand(VIEW::Math& out, VIEW::Math& in);
  void setGradOperand(VIEW::Math& dIn, VIEW::Math& dOut);
  void forward();
  void backward();
};
  
}

#endif /* OPERATOR_RELU_HPP */
