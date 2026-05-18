# OPERATOR::Relu

> Namespace: [OPERATOR](index.md)

Forward:

```cpp
out = max(in, 0)
```

Backward:

```cpp
dIn = in > 0 ? dOut : 0
```

## Constructors

```cpp
Relu(HANDLER::Workspace& workspace);
Relu(HANDLER::Workspace& workspace, VIEW::Math& out, VIEW::Math& in);
```

## Functions

```cpp
VIEW::Math& getInput();
VIEW::Math& getOutput();
VIEW::Math& getGradInput();
VIEW::Math& getGradOutput();
void setOperand(VIEW::Math& out, VIEW::Math& in);
void setGradOperand(VIEW::Math& dIn, VIEW::Math& dOut);
void forward();
void backward();
```

Kernels use `uint4`, so element count should be multiple of 8.

