# VIEW

Navigation: Previous: [core.md](core.md) | Next: [handler.md](handler.md)

`public/inc/VIEW` defines the tensor view system.

## `VIEW::Shape`

`Shape` stores:

- `dims`: up to 4 dimensions
- `strides`: row-major strides
- `rank`: number of active dimensions
- `dtype`: byte size/type enum

Example:

```cpp
int dims[VIEW::MAX_RANK] = {2, 16, 0, 0};
VIEW::Shape shape(dims, 2, VIEW::F16);
```

`setShape()` recomputes dimensions and strides. Operators rely on shape metadata to build cuDNN/cuBLASLt descriptors.

## `VIEW::Math`

`Math` is the tensor handle. It stores:

- CPU pointer
- GPU pointer
- CPU/GPU offsets in their arenas
- aligned byte size
- logical element count
- `VIEW::Shape`

`Math` does not allocate memory. Set the shape first, then bind it through `HANDLER::IO`.

```cpp
VIEW::Math x;
x.getLayout().setShape(dims, 2, VIEW::F16);
io.bind(x);
```

After binding:

```cpp
x.getCpuPtr(); // host memory
x.getGpuPtr(); // device memory
```

Navigation: Previous: [core.md](core.md) | Next: [handler.md](handler.md)
