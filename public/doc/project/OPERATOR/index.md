# OPERATOR Namespace

> **Reading Path**  
> Home: [Project Manual](../index.md) | Previous: [HANDLER](../HANDLER/index.md) | Next: [MODEL](../MODEL/index.md)

`OPERATOR` contains GPU operations. Each operator consumes and produces `VIEW::Math` tensors.

## Operators

| Operator | File | Purpose |
|---|---|---|
| `Normalize` | [normalize.md](normalize.md) | Divide F16 tensor values by a scalar. |
| `Relu` | [relu.md](relu.md) | ReLU forward and backward. |
| `SGD` | [sgd.md](sgd.md) | In-place weight update. |
| `Softmax` | [softmax.md](softmax.md) | cuDNN softmax forward/backward. |
| `CrossEntropy` | [cross_entropy.md](cross_entropy.md) | Loss and probability gradient. |
| `Pool` | [pool.md](pool.md) | cuDNN max-pooling forward/backward. |
| `Conv2DRelu` | [conv2d_relu.md](conv2d_relu.md) | Current conv+bias forward and convolution backward. |
| `MatrixMulBias` | [matrix_mul_bias.md](matrix_mul_bias.md) | Linear layer math `xW + b`. |

## General Contract

Before calling an operator:

1. Create tensors.
2. Set shapes.
3. Bind memory.
4. Put input data in CPU memory.
5. Copy inputs to GPU.
6. Attach operands.
7. Call operator function.

Outputs must already be shaped and bound.

