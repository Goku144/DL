# Usage

## Create Handlers

```cpp
HANDLER::Cpu cpu(CORE::MEMORY_32_MB);
HANDLER::Cuda gpu(CORE::MEMORY_32_MB);
HANDLER::IO io(cpu, gpu);
HANDLER::File file;
HANDLER::Workspace workspace(io, file, CORE::MEMORY_32_MB);
```

## Create Tensors

```cpp
VIEW::Math x, y;
int dims[VIEW::MAX_RANK] = {16, 0, 0, 0};

x.getLayout().setShape(dims, 1, VIEW::F16);
y.getLayout().setShape(dims, 1, VIEW::F16);
```

## Bind Memory

```cpp
io.bind(x);
io.bind(y);
```

## Fill CPU Memory

```cpp
__half *xCpu = (__half *)x.getCpuPtr();
for(int i = 0; i < 16; i++) {
  xCpu[i] = __float2half((float)i);
}
```

## Copy Inputs To GPU

```cpp
io.copyHostToDevice(x);
```

## Run Operator

```cpp
OPERATOR::Relu relu(workspace, y, x);
relu.forward();
```

## Copy Output Back

```cpp
VIEW::Math yFloat;
io.copyHalfToCpuFloat(yFloat, y);
io.printData(yFloat);
```

## Error Handling

Handlers store errors internally:

```cpp
if(io.peekErr() != CORE::ioSuccess) {
  io.info();
}
```

Use `getErr()` when you want to read and clear the error.

