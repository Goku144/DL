# OPERATOR::MatrixMulBias

> Namespace: [OPERATOR](index.md)

Linear layer math:

```cpp
out = in * weight + bias
```

## Shape Contract

```text
in     = [N, K]
weight = [K, M]
bias   = [M]
out    = [N, M]
```

## Constructors

```cpp
MatrixMulBias(HANDLER::Workspace& workspace);
MatrixMulBias(HANDLER::Workspace& workspace, VIEW::Math& out, VIEW::Math& in, VIEW::Math& weight, VIEW::Math& bias);
```

## Functions

```cpp
VIEW::Math& getInput();
VIEW::Math& getWeight();
VIEW::Math& getBias();
VIEW::Math& getOutput();
VIEW::Math& getGradInput();
VIEW::Math& getGradWeight();
VIEW::Math& getGradBias();
VIEW::Math& getGradOutput();
void setOperand(VIEW::Math& out, VIEW::Math& in, VIEW::Math& weight, VIEW::Math& bias);
void setGradOperand(VIEW::Math& dIn, VIEW::Math& dWeight, VIEW::Math& dBias, VIEW::Math& dOut);
void forward();
void backward();
```

Backward computes:

```cpp
dIn = dOut * weight^T
dWeight = in^T * dOut
dBias = sum(dOut over batch)
```

