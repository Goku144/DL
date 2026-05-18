# OPERATOR::Conv2DRelu

> Namespace: [OPERATOR](index.md)

Current forward behavior:

```cpp
out = conv2d(in, weight) + bias
```

Despite the class name, ReLU activation is not currently applied in `forward()`.

## Shape Contract

```text
in     = [N, C, H, W]
weight = [K, C, R, S]
bias   = [K]
out    = [N, K, OH, OW]
```

## Constructors

```cpp
Conv2DRelu(HANDLER::Workspace& workspace);
Conv2DRelu(HANDLER::Workspace& workspace, VIEW::Math& out, VIEW::Math& in, VIEW::Math& weight, VIEW::Math& bias);
```

## Functions

```cpp
VIEW::Math& getInput();
VIEW::Math& getWeight();
VIEW::Math& getBias();
VIEW::Math& getOutput();
void setOperand(VIEW::Math& out, VIEW::Math& in, VIEW::Math& weight, VIEW::Math& bias);
void setGradOperand(VIEW::Math& dIn, VIEW::Math& dWeight, VIEW::Math& dBias, VIEW::Math& dOut);
void setConfig(int padH = 1, int padW = 1, int strideH = 1, int strideW = 1, int dilationH = 1, int dilationW = 1);
void forward();
void backward();
```

`backward` computes `dBias`, `dWeight`, and `dIn`.

