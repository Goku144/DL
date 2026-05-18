# OPERATOR::Normalize

> Namespace: [OPERATOR](index.md)

Computes:

```cpp
out = in / scalar
```

## Constructors

```cpp
Normalize(HANDLER::Workspace& workspace);
Normalize(HANDLER::Workspace& workspace, VIEW::Math& out, VIEW::Math& in);
```

## Functions

```cpp
VIEW::Math& getInput();
VIEW::Math& getOutput();
void setOperand(VIEW::Math& out, VIEW::Math& in);
void normByScalar(float scalar = 255.0f);
```

`normByScalar` launches the GPU kernel.

Input/output are F16. Element count should be multiple of 8.

