#if !defined(VIEW_SHAPE_HPP)
#define VIEW_SHAPE_HPP

#include "CORE/State.hpp"

namespace VIEW
{

static constexpr int maxRank = 4;

#define MAX_RANK maxRank

enum DType
{
  CHAR = sizeof (char),
  F16 = sizeof (uint16_t),
  FLOAT = sizeof (float),
};

class __align__(CORE::ALIGNE_TO_256) Shape
{ 
private:
  int dims[VIEW::MAX_RANK] = {0};
  int strides[VIEW::MAX_RANK] = {0};
  int rank = 0;
  VIEW::DType dtype = VIEW::CHAR;

public:

  Shape(int dims[VIEW::MAX_RANK], int rank = 0, VIEW::DType dtype = VIEW::CHAR);
  ~Shape();

  int getMaxRank() const;

  int getRank() const;
  int getDim(int index) const;
  int getStride(int index) const;

  VIEW::DType getDType() const;

  void info() const;
};
  
}

#endif /* VIEW_SHAPE_HPP */
