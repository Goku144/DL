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

  Shape(int dims[VIEW::MAX_RANK], int rank = 0, VIEW::DType dtype = VIEW::CHAR, const char* file = __FILE__, int line = __LINE__);
  Shape();
  ~Shape();

  int getDim(int index) const;
  int getStride(int index) const;
  int getRank() const;
  VIEW::DType getDType() const;

  void setDim(int dims[VIEW::MAX_RANK], const char* file, int line);
  void setRank(int rank, const char* file, int line);
  void setDtype(VIEW::DType dtype);

  int getMaxRank() const;
  int getSuperPosition(int i = 0, int j = 0, int k = 0, int l = 0) const;
  void setShape(int dims[VIEW::MAX_RANK], int rank = 0, VIEW::DType dtype = VIEW::CHAR, const char* file = __FILE__, int line = __LINE__);
  void info(const char* file = __FILE__, int line = __LINE__) const;
};
  
}

#endif /* VIEW_SHAPE_HPP */
