#include "CORE/State.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstdarg>

void CORE::printState(State level, const char *file, int line, const char *fmt, ...)
{
  switch (level)
  {
    case CORE::INFO: fprintf(stderr, "\033[1;32m[  INFO   ]\033[0m "); break;
    case CORE::WARN: fprintf(stderr, "\033[1;38;5;220m[ WARNING ]\033[0m "); break;
    case CORE::FATAL: fprintf(stderr, "\033[1;38;5;196m[  FATAL  ]\033[0m "); break;
  }

  fprintf(stderr, "\033[90m%s:%d:\033[0m ", file, line);
  
  va_list args;
  va_start(args, fmt);
  vfprintf(stderr, fmt, args);
  va_end(args);

  fprintf(stderr, "\n");

  if (level == State::FATAL) exit(EXIT_FAILURE);
}

