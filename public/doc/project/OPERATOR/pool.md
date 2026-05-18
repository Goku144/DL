# OPERATOR::Pool

> Namespace: [OPERATOR](index.md)

cuDNN max-pooling for NCHW F16 tensors.

## Constructors

```cpp
Pool(HANDLER::Workspace& workspace);
Pool(HANDLER::Workspace& workspace, VIEW::Math& out, VIEW::Math& in);
```

## Functions

```cpp
VIEW::Math& getInput();
VIEW::Math& getOutput();
VIEW::Math& getGradInput();
VIEW::Math& getGradOutput();
void setOperand(VIEW::Math& out, VIEW::Math& in);
void setGradOperand(VIEW::Math& dIn, VIEW::Math& dOut);
void setConfig(int windowH = 2, int windowW = 2, int padH = 0, int padW = 0, int strideH = 2, int strideW = 2);
void maxForward();
void maxBackward();
```

`setConfig` controls window, padding, and stride.

