#if !defined(MODEL_DL_HPP)
#define MODEL_DL_HPP

#include "CORE/State.hpp"

#include "HANDLER/Cpu.hpp"
#include "HANDLER/Cuda.hpp"
#include "HANDLER/File.hpp"
#include "HANDLER/IO.hpp"
#include "HANDLER/Workspace.hpp"

#include "OPERATOR/Conv2DReLU.hpp"
#include "OPERATOR/CrossEntropy.hpp"
#include "OPERATOR/MatrixMulBias.hpp"
#include "OPERATOR/Normalize.hpp"
#include "OPERATOR/Pool.hpp"
#include "OPERATOR/Relu.hpp"
#include "OPERATOR/SGD.hpp"
#include "OPERATOR/Softmax.hpp"

#include "VIEW/Math.hpp"
#include "VIEW/Shape.hpp"

namespace MODEL
{

/**
 * @brief Concrete model orchestration class for the current digit classifier.
 *
 * DL owns the runtime handlers, dataset tensors, trainable parameters,
 * intermediate activations, gradients, loss tensor, checkpointing logic,
 * training loop, and single-image estimate path. It wires the low-level
 * VIEW/HANDLER/OPERATOR primitives into a fixed CNN-style model.
 */
class __align__(CORE::ALIGNE_TO_256) DL
{
private:
// layer 1
  VIEW::Math a0, z0, x, w0, b0;
  VIEW::Math /*da0,*/ dz0, dw0, db0;

// layer 1.5
  VIEW::Math p /*, a0*/;
  VIEW::Math da0 /*, dp*/;

// layer 2
  VIEW::Math a1, z1 /*, p*/, w1, b1;
  VIEW::Math dp /*, da1*/, dz1, dw1, db1;

// layer 3
  VIEW::Math out, z2 /*, a1*/, w2, b2;
  VIEW::Math da1 /*, dz2*/, dw2, db2;

// layer Lost
  VIEW::Math L /*, out*/, Y;
  VIEW::Math dz2;

  VIEW::Math filePaths;

  size_t imageOffset = 0;
  size_t imageBatch = 0;
  size_t modImage = 0;

  HANDLER::Cpu *cpu;
  HANDLER::Cuda *gpu;
  HANDLER::File *file;
  HANDLER::IO *io;
  HANDLER::Workspace *workspace;

  /** @brief Relayout batch-dependent tensors for the active batch size.
   * @param batch Number of images in the active batch.
   * */
  void setActiveBatch(size_t batch);

  /** @brief Initialize trainable parameters.
   *
   * Weights are initialized with deterministic F16 pseudo-random values and
   * biases are initialized to zero.
   * */
  void initializeParameters();

  /** @brief Save trainable parameters to a raw checkpoint file.
   * @param path Destination checkpoint path.
   *
   * The checkpoint stores parameters in fixed order: w0, b0, w1, b1, w2, b2.
   * */
  void saveCheckpoint(const char *path);

  /** @brief Load trainable parameters from a raw checkpoint file.
   * @param path Source checkpoint path.
   * @return true when all tensors are restored, false on null path, read error,
   * size mismatch, or restore failure.
   * */
  bool loadCheckpoint(const char *path);

  /** @brief Run forward propagation for one dataset batch.
   * @param batchOffset Starting image index inside the loaded image set.
   * @param batch Number of images to process.
   *
   * The current graph is image pull, normalization, convolution, ReLU, pooling,
   * first linear layer, ReLU, second linear layer, and softmax.
   * */
  void forward(size_t batchOffset, size_t batch);

  /** @brief Run backward propagation through the current graph.
   *
   * Consumes the output probability gradient in dz2 and writes parameter
   * gradients for all trainable weights and biases.
   * */
  void backward();

  /** @brief Apply SGD updates to every trainable parameter.
   * @param learningRate Scalar learning rate used by OPERATOR::SGD.
   * */
  void update(float learningRate);
  
public:
  /** @brief Create the model runtime, load dataset metadata/images, and bind tensors.
   * @param imageBatch Preferred training batch size. Passing 0 uses the model default.
   * @param csvPath Dataset CSV path containing image paths and labels.
   * */
  DL(size_t imageBatch, const char *csvPath = "public/target/meta/test.csv");

  /** @brief Destroy owned handlers and release runtime resources. */
  ~DL();

  /** @brief Get configured training batch size.
   * @return Configured image batch size.
   * */
  size_t getImageBatch() const;

  /** @brief Train the model with mini-batch SGD.
   * @param iterations Number of training iterations to run.
   * @param learningRate SGD learning rate.
   * @param checkpointEvery Save checkpoint every N iterations. Use 0 to disable
   * periodic checkpointing.
   * @param checkpointPrefix Prefix used for checkpoint filenames.
   * */
  void train(size_t iterations, float learningRate = 0.01f, size_t checkpointEvery = 0, const char *checkpointPrefix = "public/checkpoints/dl");

  /** @brief Estimate the digit class for one image.
   * @param imagePath Path to the image to classify.
   * @param checkpointPath Optional checkpoint path to load before inference.
   *
   * Logs probabilities for the real digit classes 0 through 9.
   * */
  void estimate(const char *imagePath, const char *checkpointPath = NULL);

};

}

#endif /* MODEL_DL_HPP */
