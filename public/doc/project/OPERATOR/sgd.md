# OPERATOR::SGD

> Namespace: [OPERATOR](index.md)

In-place optimizer/update operation:

```cpp
weight = weight - lr * grad
```

## Constructors

```cpp
SGD(HANDLER::Workspace& workspace);
SGD(HANDLER::Workspace& workspace, VIEW::Math& weight, VIEW::Math& grad);
```

## Functions

```cpp
VIEW::Math& getWeight();
VIEW::Math& getGrad();
void setOperand(VIEW::Math& weight, VIEW::Math& grad);
void update(float lr);
```

`update` launches the vectorized update kernel.

