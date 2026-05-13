#if !defined(CORE_STATE_HPP)
#define CORE_STATE_HPP

namespace CORE
{
  #define logInfo(x, ...) printState(CORE::INFO, __FILE__, __LINE__, x, ##__VA_ARGS__);
  #define logWarn(x, ...) printState(CORE::WARN, __FILE__, __LINE__, x, ##__VA_ARGS__);
  #define logFatal(x, ...) printState(CORE::FATAL, __FILE__, __LINE__, x, ##__VA_ARGS__);

  enum State
  {
    INFO = 0,
    WARN,
    FATAL,
  };
  
  void printState(State level, const char *file, int line, const char *fmt, ...);
}

#endif /* CORE_STATE_HPP */
