#if !defined(OPERATOR_NORMALIZE_HPP)
#define OPERATOR_NORMALIZE_HPP

#include "HANDLER/Workspace.hpp"

namespace OPERATOR
{

class __align__(CORE::ALIGNE_TO_256) Normalize
{
private:
  VIEW::Math *in;
  VIEW::Math *out;
  HANDLER::Workspace *workspace;
  
public:
  Normalize(HANDLER::Workspace& workspace);
  Normalize(HANDLER::Workspace& workspace, VIEW::Math& out, VIEW::Math& in);
  ~Normalize();

  VIEW::Math& getInput();
  VIEW::Math& getOutput();

  void setOperand(VIEW::Math& out, VIEW::Math& in);
  void normByScalar(float scalar = 255.0f);
};
  
}

#endif /* OPERATOR_NORMALIZE_HPP */
