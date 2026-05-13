#include "VIEW/Math.hpp"

#include <stdio.h>

int main()
{
  HANDLER::Cpu cpu(CORE::MEMORY_32_MB);

  VIEW::Shape layout[4];
  VIEW::Math math[4];

  printf("\t\n ============================== \n\n");
  for (int i = 0; i < 4; i++)
  {
    layout[i].setShape((int[4]){i*2 + 1, i + 1, i * 3 + 2, i + 4}, 4);
    math[i].bind(cpu, layout[i]);
    math[i].info();
    printf("\t\n ============================== \n\n");
  }

  
  return 0;
}