# App Test Harness

> **Reading Path**  
> Home: [Project Guide](index.md) | Previous: [OPERATOR](operator.md) | Next: [Usage](usage.md)

`app/src/app.cu` is now a hungry operator readiness harness.

It creates:

```cpp
HANDLER::Cpu cpu(CORE::MEMORY_32_MB);
HANDLER::Cuda gpu(CORE::MEMORY_32_MB);
HANDLER::IO io(cpu, gpu);
HANDLER::File file;
HANDLER::Workspace workspace(io, file, CORE::MEMORY_32_MB);
```

Then it tests:

- `Normalize`
- `Relu` forward and backward
- `SGD`
- `Softmax`
- `CrossEntropy`
- `Pool` forward and backward
- `Conv2DRelu` forward and backward
- `MatrixMulBias` forward and backward

Each test builds a small CPU reference result and compares GPU output after copying back to CPU float.

The harness prints `[PASS]` or `[FAIL]` per check and returns nonzero if any operator fails.

This file is meant to answer: "Are these operators ready to become math layers?"

---

> **Continue Reading**  
> Previous: [OPERATOR](operator.md) | Next: [Usage](usage.md)
