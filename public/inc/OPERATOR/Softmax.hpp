#if !defined(OPERATOR_SOFTMAX_HPP)
#define OPERATOR_SOFTMAX_HPP

#include "HANDLER/Workspace.hpp"

#include <cudnn.h>

namespace OPERATOR
{

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
  Softmax(HANDLER::Workspace& workspace);
  Softmax(HANDLER::Workspace& workspace, VIEW::Math& out, VIEW::Math& in);
  ~Softmax();

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

#endif /* OPERATOR_SOFTMAX_HPP */
