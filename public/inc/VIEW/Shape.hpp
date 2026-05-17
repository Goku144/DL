#if !defined(VIEW_SHAPE_HPP)
#define VIEW_SHAPE_HPP

#include "CORE/State.hpp"

namespace VIEW
{

static constexpr int maxRank = 4;

#define MAX_RANK maxRank

/** @brief Element storage type, represented by byte size. */
enum DType
{
  CHAR = sizeof (char),
  F16 = sizeof (uint16_t),
  FLOAT = sizeof (float),
};

/**
 * @brief Fixed-rank tensor shape with row-major strides and dtype.
 *
 * Shape stores up to four dimensions, their computed strides, rank, and dtype.
 * IO uses it to compute element count and bytes during binding.
 */
class __align__(CORE::ALIGNE_TO_256) Shape
{ 
private:
  int dims[VIEW::MAX_RANK] = {0};
  int strides[VIEW::MAX_RANK] = {0};
  int rank = 0;
  VIEW::DType dtype = VIEW::CHAR;

public:
  /** @brief Construct a shape. @param dims Dimension array. @param rank Number of active dimensions. @param dtype Element type. @param file Source file for fatal logs. @param line Source line for fatal logs. */
  Shape(int dims[VIEW::MAX_RANK], int rank = 0, VIEW::DType dtype = VIEW::CHAR, const char* file = __FILE__, int line = __LINE__);

  /** @brief Construct an empty shape. */
  Shape();

  /** @brief Destroy the shape. */
  ~Shape();

  /** @brief Get dimension. @param index Dimension index. @return Dimension value. */
  int getDim(int index) const;

  /** @brief Get stride. @param index Dimension index. @return Stride value in elements. */
  int getStride(int index) const;

  /** @brief Get rank. @return Number of active dimensions. */
  int getRank() const;

  /** @brief Get dtype. @return Element type. */
  VIEW::DType getDType() const;

  /** @brief Replace dimensions and recompute strides. @param dims Dimension array. @param file Source file for fatal logs. @param line Source line for fatal logs. */
  void setDim(int dims[VIEW::MAX_RANK], const char* file, int line);

  /** @brief Replace rank and recompute strides. @param rank Number of active dimensions. @param file Source file for fatal logs. @param line Source line for fatal logs. */
  void setRank(int rank, const char* file, int line);

  /** @brief Set dtype. @param dtype Element type. */
  void setDtype(VIEW::DType dtype);

  /** @brief Get max supported rank. @return VIEW::MAX_RANK. */
  int getMaxRank() const;

  /** @brief Convert indices to byte offset. @param i First index. @param j Second index. @param k Third index. @param l Fourth index. @return Byte offset for the indexed element. */
  int getSuperPosition(int i = 0, int j = 0, int k = 0, int l = 0) const;

  /** @brief Replace dimensions, rank, and dtype. @param dims Dimension array. @param rank Number of active dimensions. @param dtype Element type. @param file Source file for fatal logs. @param line Source line for fatal logs. */
  void setShape(int dims[VIEW::MAX_RANK], int rank = 0, VIEW::DType dtype = VIEW::CHAR, const char* file = __FILE__, int line = __LINE__);

  /** @brief Print shape metadata. @param file Source file for log. @param line Source line for log. */
  void info(const char* file = __FILE__, int line = __LINE__) const;
};
  
}

#endif /* VIEW_SHAPE_HPP */
