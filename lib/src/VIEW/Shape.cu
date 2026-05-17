#include "VIEW/Shape.hpp"

VIEW::Shape::Shape(int dims[VIEW::MAX_RANK], int rank, VIEW::DType dtype, const char* file, int line)
{
  if(rank > VIEW::MAX_RANK)
    CORE::logFatal(file, line, "Shape Rank overflow (rank = %d)", rank);
  
  int tmp = 1, len = rank - 1;
  #pragma unroll
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

int VIEW::Shape::getDim(int index) const
{
  return this->dims[index];
}

int VIEW::Shape::getStride(int index) const
{
  return this->strides[index];
}

int VIEW::Shape::getRank() const
{
  return this->rank;
}

VIEW::DType VIEW::Shape::getDType() const
{
  return this->dtype;
}

void VIEW::Shape::setDim(int dims[VIEW::MAX_RANK], const char* file, int line)
{
  this->setShape(dims, this->rank, this->dtype, file, line);
}

void VIEW::Shape::setRank(int rank, const char* file, int line)
{
  this->setShape(this->dims, rank, this->dtype, file, line);
}

void VIEW::Shape::setDtype(VIEW::DType dtype)
{
  this->dtype = dtype;
}

int VIEW::Shape::getMaxRank() const
{
  return VIEW::MAX_RANK;
}

int VIEW::Shape::getSuperPosition(int i, int j, int k, int l) const
{
  return (i * this->strides[0] + j * this->strides[1] + k * this->strides[0] + l * this->strides[0]) * this->dtype; 
}

void VIEW::Shape::setShape(int dims[VIEW::MAX_RANK], int rank, VIEW::DType dtype, const char* file, int line)
{
  if(rank > VIEW::MAX_RANK)
    CORE::logFatal(file, line, "Shape Rank overflow (rank = %d)", rank);

  int tmp = 1, len = rank - 1;
  for (size_t index = 0; index < VIEW::MAX_RANK; index++)
  {
    this->dims[index] = 0;
    this->strides[index] = 0;
    if(index < rank)
    {
      this->strides[len - index] = tmp;
      this->dims[index] = dims[index];
      tmp *= dims[len - index];
    }
  }
  
  this->rank = rank;
  this->dtype = dtype;
}

void VIEW::Shape::info(const char* file, int line) const
{
  CORE::logInfo(file, line, "dims: %d %d %d %d", 
  this->dims[0], this->dims[1], this->dims[2], this->dims[3]);
  CORE::logInfo(file, line, "strides: %d %d %d %d",
  this->strides[0], this->strides[1], this->strides[2], this->strides[3]);
  CORE::logInfo(file, line, "rank: %d", this->rank);
  CORE::logInfo(file, line, "data type: %d", this->dtype);
}
