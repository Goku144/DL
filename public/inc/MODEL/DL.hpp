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

  void setActiveBatch(size_t batch);
  void initializeParameters();
  void saveCheckpoint(const char *path);
  bool loadCheckpoint(const char *path);
  void forward(size_t batchOffset, size_t batch);
  void backward();
  void update(float learningRate);
public:
  DL(size_t imageBatch, const char *csvPath = "public/target/meta/test.csv");
  ~DL();

  size_t getImageBatch() const;
  void train(size_t iterations, float learningRate = 0.01f, size_t checkpointEvery = 0, const char *checkpointPrefix = "public/checkpoints/dl");
  void estimate(const char *imagePath, const char *checkpointPath = NULL);

};

}

#endif /* MODEL_DL_HPP */
