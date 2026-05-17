#if !defined(OPERATOR_POOL_HPP)
#define OPERATOR_POOL_HPP

#include "HANDLER/Workspace.hpp"

#include <cudnn.h>

namespace OPERATOR
{

class __align__(CORE::ALIGNE_TO_256) Pool
{
private:
  VIEW::Math *in;
  VIEW::Math *out;
  VIEW::Math *dIn;
  VIEW::Math *dOut;
  HANDLER::Workspace *workspace;

  cudnnTensorDescriptor_t xDesc;
  cudnnTensorDescriptor_t yDesc;
  cudnnPoolingDescriptor_t poolDesc;

  int windowH;
  int windowW;
  int padH;
  int padW;
  int strideH;
  int strideW;

  void setDescriptor();

public:
  Pool(HANDLER::Workspace& workspace);
  Pool(HANDLER::Workspace& workspace, VIEW::Math& out, VIEW::Math& in);
  ~Pool();

  VIEW::Math& getInput();
  VIEW::Math& getOutput();
  VIEW::Math& getGradInput();
  VIEW::Math& getGradOutput();

  void setOperand(VIEW::Math& out, VIEW::Math& in);
  void setGradOperand(VIEW::Math& dIn, VIEW::Math& dOut);
  void setConfig(int windowH = 2, int windowW = 2, int padH = 0, int padW = 0, int strideH = 2, int strideW = 2);
  void maxForward();
  void maxBackward();
};
  
}

#endif /* OPERATOR_POOL_HPP */
