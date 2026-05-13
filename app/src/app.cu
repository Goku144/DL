#include "VIEW/Shape.hpp"

int main()
{
  VIEW::Shape layout((int []) {1,4,4,3}, 4, VIEW::FLOAT);
  layout.info();
  return 0;
}