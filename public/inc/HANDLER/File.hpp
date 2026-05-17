#if !defined(HANDLER_FILE_HPP)
#define HANDLER_FILE_HPP

#include "CORE/State.hpp"
#include "HANDLER/IO.hpp"

namespace HANDLER
{

class __align__(CORE::ALIGNE_TO_256) File
{
private:
  HANDLER::IO *io = NULL;
  CORE::errFile err = CORE::fileSuccess;
  void *imageCpuPtr = NULL;
  void *imageGpuPtr = NULL;
  size_t imageCpuOffset = 0;
  size_t imageGpuOffset = 0;
  size_t imageN = 0;
  int imageHeight = 0;
  int imageWidth = 0;
  int imageChannel = 0;

public:
  File();
  File(HANDLER::IO& io);
  ~File();

  void setIO(HANDLER::IO& io);

  void read(VIEW::Math& dst, const char *path, VIEW::DType dtype = VIEW::CHAR);
  void write(VIEW::Math& src, const char *path);

  void readCsv(VIEW::Math& filePaths, VIEW::Math& labels, const char *path);

  void readImages(VIEW::Math& filePaths, const char *rootPath = NULL, int desiredChannels = 0);
  void copyImageToDevice();

  void pullCpuImage(VIEW::Math& dst, size_t n, size_t offset = 0);
  void pullGpuImage(VIEW::Math& dst, size_t n, size_t offset = 0);

  size_t getImageN() const;
  int getImageHeight() const;
  int getImageWidth() const;
  int getImageChannel() const;

  CORE::errFile getErr();
  CORE::errFile peekErr() const;
  void clearErr();

  void info(CORE::State level = CORE::WARN, const char *file = __FILE__, int line = __LINE__) const;
};
  
}

#endif /* HANDLER_FILE_HPP */
