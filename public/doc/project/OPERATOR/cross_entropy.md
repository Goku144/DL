# OPERATOR::CrossEntropy

> Namespace: [OPERATOR](index.md)

Fused loss and gradient.

Inputs:

- `prob`: probability tensor
- `target`: class labels

Outputs:

- `loss`: scalar FLOAT tensor
- `dProb`: F16 probability gradient

## Constructors

```cpp
CrossEntropy(HANDLER::Workspace& workspace);
CrossEntropy(HANDLER::Workspace& workspace, VIEW::Math& dProb, VIEW::Math& loss, VIEW::Math& prob, VIEW::Math& target);
```

## Functions

```cpp
VIEW::Math& getProb();
VIEW::Math& getTarget();
VIEW::Math& getLoss();
VIEW::Math& getGradProb();
void setOperand(VIEW::Math& dProb, VIEW::Math& loss, VIEW::Math& prob, VIEW::Math& target);
void forwardBackward();
```

`forwardBackward` computes average negative log-likelihood and `dProb`.

