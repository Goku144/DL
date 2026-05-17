#if !defined(HANDLER_FILE_HPP)
#define HANDLER_FILE_HPP

#include "CORE/State.hpp"
#include "HANDLER/IO.hpp"

namespace HANDLER
{

/**
 * @brief Dataset and raw file IO helper.
 *
 * File reads binary blobs, CSV image metadata, and image batches into
 * VIEW::Math tensors. It depends on HANDLER::IO for memory allocation and host
 * to device copies.
 */
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
  /** @brief Create a File handler without IO. */
  File();

  /** @brief Create a File handler with IO. @param io IO handler used for allocation/copies. */
  File(HANDLER::IO& io);

  /** @brief Destroy the File handler. */
  ~File();

  /** @brief Attach IO handler. @param io IO handler used for allocation/copies. */
  void setIO(HANDLER::IO& io);

  /** @brief Read a raw file into CPU tensor memory. @param dst Destination tensor. @param path File path. @param dtype Element type to assign. @note Sets fileErrOpen, fileErrRead, fileErrClose, fileErrNull, or fileErrIO. */
  void read(VIEW::Math& dst, const char *path, VIEW::DType dtype = VIEW::CHAR);

  /** @brief Write CPU tensor memory to a raw file. @param src Source tensor. @param path File path. @note Sets fileErrOpen, fileErrWrite, fileErrClose, or fileErrNull. */
  void write(VIEW::Math& src, const char *path);

  /** @brief Read CSV metadata into path and label tensors. @param filePaths Destination tensor for paths. @param labels Destination tensor for labels. @param path CSV file path. @note Sets fileErrOpen, fileErrReadCsv, or fileErrIO. */
  void readCsv(VIEW::Math& filePaths, VIEW::Math& labels, const char *path);

  /** @brief Read image files listed in filePaths into CPU memory. @param filePaths Tensor containing path strings. @param rootPath Optional root directory. @param desiredChannels Requested channel count, or 0 to keep source. @note Sets fileErrReadImg, fileErrInvalidState, or fileErrIO. */
  void readImages(VIEW::Math& filePaths, const char *rootPath = NULL, int desiredChannels = 0);

  /** @brief Convert loaded uint8 CPU images to F16 GPU images. @note Sets fileErrInvalidState or fileErrIO. */
  void copyImageToDevice();

  /** @brief Expose a CPU image batch as a Math view. @param dst Destination view. @param n Number of images. @param offset Image offset. @note Sets fileErrInvalidState or fileErrOutOfBound-style file error when invalid. */
  void pullCpuImage(VIEW::Math& dst, size_t n, size_t offset = 0);

  /** @brief Expose a GPU image batch as a Math view. @param dst Destination view. @param n Number of images. @param offset Image offset. @note Sets fileErrInvalidState or fileErrOutOfBound-style file error when invalid. */
  void pullGpuImage(VIEW::Math& dst, size_t n, size_t offset = 0);

  /** @brief Get loaded image count. @return Number of images. */
  size_t getImageN() const;

  /** @brief Get image height. @return Height in pixels. */
  int getImageHeight() const;

  /** @brief Get image width. @return Width in pixels. */
  int getImageWidth() const;

  /** @brief Get image channel count. @return Number of channels. */
  int getImageChannel() const;

  /** @brief Get and clear file error. @return Current error before clearing. */
  CORE::errFile getErr();

  /** @brief Read file error without clearing. @return Current file error. */
  CORE::errFile peekErr() const;

  /** @brief Clear file error. */
  void clearErr();

  /** @brief Print current file error. @param level WARN or FATAL logging level. @param file Source file for log. @param line Source line for log. */
  void info(CORE::State level = CORE::WARN, const char *file = __FILE__, int line = __LINE__) const;
};
  
}

#endif /* HANDLER_FILE_HPP */
