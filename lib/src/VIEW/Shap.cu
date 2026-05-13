#include "VIEW/Shape.hpp"

VIEW::Shape::Shape(int dims[VIEW::MAX_RANK], int rank, VIEW::DType dtype)
{
  if(rank > VIEW::MAX_RANK)
    CORE::logFatal("Shape Rank overflow (rank = %d)", rank);
  
  int tmp = 1, len = rank - 1;
  for (size_t index = 0; index < rank; index++)
  {
    this->strides[len - index] = tmp;
    this->dims[index] = dims[index];
    tmp *= dims[len - index];
  }
  this->rank = rank;
  this->dtype = dtype;
}

VIEW::Shape::Shape()
{}

VIEW::Shape::~Shape()
{}

int VIEW::Shape::getMaxRank() const
{
  return VIEW::MAX_RANK;
}

int VIEW::Shape::getRank() const
{
  return this->rank;
}

int VIEW::Shape::getDim(int index) const
{
  return this->dims[index];
}

int VIEW::Shape::getStride(int index) const
{
  return this->strides[index];
}

VIEW::DType VIEW::Shape::getDType() const
{
  return this->dtype;
}

void VIEW::Shape::setShape(int dims[VIEW::MAX_RANK], int rank, VIEW::DType dtype)
{
  if(rank > VIEW::MAX_RANK)
    CORE::logFatal("Shape Rank overflow (rank = %d)", rank);
  
  int tmp = 1, len = rank - 1;
  for (size_t index = 0; index < rank; index++)
  {
    this->strides[len - index] = tmp;
    this->dims[index] = dims[index];
    tmp *= dims[len - index];
  }
  this->rank = rank;
  this->dtype = dtype;
}

void VIEW::Shape::info() const
{
  CORE::logInfo("dims: %d %d %d %d", 
  this->dims[0], this->dims[1], this->dims[2], this->dims[3]);
  CORE::logInfo("strides: %d %d %d %d",
  this->strides[0], this->strides[1], this->strides[2], this->strides[3]);
  CORE::logInfo("rank: %d", this->rank);
  CORE::logInfo("data type: %d", this->dtype);
}