# Usage

> **Reading Path**  
> Home: [Project Manual](index.md) | Previous: [MODEL](MODEL/index.md) | Next: [Full Overview](huge.md)

This is the shortest practical recipe for using the runtime.

## 1. Create Handlers

```cpp
HANDLER::Cpu cpu(CORE::MEMORY_32_MB);
HANDLER::Cuda gpu(CORE::MEMORY_32_MB);
HANDLER::IO io(cpu, gpu);
HANDLER::File file;
HANDLER::Workspace workspace(io, file, CORE::MEMORY_32_MB);
```

Meaning:

- `Cpu` owns host arena memory
- `Cuda` owns device arena memory
- `IO` binds tensors and copies data
- `File` can load datasets/files
- `Workspace` owns stream, cuDNN, cuBLASLt, and scratch memory

## 2. Create Tensors

```cpp
VIEW::Math x, y;
int dims[VIEW::MAX_RANK] = {16, 0, 0, 0};

x.getLayout().setShape(dims, 1, VIEW::F16);
y.getLayout().setShape(dims, 1, VIEW::F16);
```

## 3. Bind Memory

```cpp
io.bind(x);
io.bind(y);
```

After this:

```cpp
x.getCpuPtr(); // valid CPU pointer
x.getGpuPtr(); // valid GPU pointer
```

## 4. Fill Input CPU Memory

```cpp
__half *xCpu = (__half *)x.getCpuPtr();
for(int i = 0; i < 16; i++) {
  xCpu[i] = __float2half((float)i);
}
```

## 5. Copy Inputs To GPU

```cpp
io.copyHostToDevice(x);
```

## 6. Run Operator

```cpp
OPERATOR::Relu relu(workspace, y, x);
relu.forward();
```

## 7. Copy Output Back

For F16 output that you want to read as float:

```cpp
VIEW::Math yFloat;
io.copyHalfToCpuFloat(yFloat, y);
io.printData(yFloat);
```

For same dtype CPU copy:

```cpp
io.copyDeviceToHost(y);
```

## Error Handling

Handlers store errors internally:

```cpp
if(io.peekErr() != CORE::ioSuccess) {
  io.info();
}
```

Use `getErr()` when you want to read and clear the error:

```cpp
CORE::errIO err = io.getErr();
```

## App Folder

`app/src/app.cu` is user-owned application space. It can be used for experiments, examples, manual tests, or real programs. It is not a required framework layer.

---

> **Continue Reading**  
> Previous: [MODEL](MODEL/index.md) | Next: [Full Overview](overview.md)
