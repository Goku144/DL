# OPERATOR::Softmax

> Namespace: [OPERATOR](index.md)

cuDNN softmax operator.

Rank behavior:

- rank 2: `[batch, classes]`
- rank 1: one row with `classes = dim0`

## Constructors

```cpp
Softmax(HANDLER::Workspace& workspace);
Softmax(HANDLER::Workspace& workspace, VIEW::Math& out, VIEW::Math& in);
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

`forward` computes probabilities. `backward` computes gradient with respect to logits.

