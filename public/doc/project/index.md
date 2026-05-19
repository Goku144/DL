# DL Project Manual

> **Reading Path**  
> Home: **Project Manual** | Previous: None | Next: [Architecture](architecture.md)

This manual explains the DL runtime from the top of the hierarchy down to each public class and function.

DL is organized as a small CUDA/C++ neural-network runtime. It is not yet a complete deep-learning framework. It currently gives you the runtime building blocks: tensor metadata, memory handlers, GPU execution handles, IO helpers, and operators.

## Source Hierarchy

```text
CORE/
  Shared constants, alignment, errors, and logging.

VIEW/
  Tensor shape and tensor view objects.

HANDLER/
  CPU/GPU memory arenas, tensor binding, file IO, and execution workspace.

OPERATOR/
  GPU operators that consume and produce VIEW::Math tensors.

MODEL/
  Current model orchestration, including MODEL::DL.
```

## Documentation Map

| Order | Document | What It Explains |
|---:|---|---|
| 1 | [Architecture](architecture.md) | How all classes are orchestrated together. |
| 2 | [CORE](CORE/index.md) | Shared project definitions and error vocabulary. |
| 3 | [VIEW](VIEW/index.md) | Shape and tensor view objects. |
| 4 | [HANDLER](HANDLER/index.md) | Memory, data movement, files, and workspace classes. |
| 5 | [OPERATOR](OPERATOR/index.md) | Every operator class, operand contract, and function. |
| 6 | [MODEL](MODEL/index.md) | Current model orchestration and future model/layer direction. |
| 7 | [Usage](usage.md) | Minimal practical usage recipe. |
| 8 | [Full Overview](overview.md) | Long-form narrative overview. |

## Most Important Rule

Every operator follows the same life cycle:

```text
create handlers
create VIEW::Math tensors
set tensor shapes
bind memory
fill input CPU memory
copy inputs to GPU
attach operands to operator
run operation
copy outputs back if needed
```

Outputs must already exist, have a shape, and be bound before an operator writes to them.

---

> **Continue Reading**  
> Next: [Architecture](architecture.md)
