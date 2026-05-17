# CORE

> **Reading Path**  
> Home: [Project Guide](index.md) | Previous: [Project Guide](index.md) | Next: [VIEW](view.md)

`public/inc/CORE` contains the low-level shared definitions.

## `CORE::Aligne`

Alignment constants. The project mostly uses `CORE::ALIGNE_TO_256` so tensor memory is friendly to vectorized kernels.

## `CORE::MemorySize`

Named memory sizes such as `CORE::MEMORY_32_MB` and `CORE::MEMORY_1_GB`. They are used to size CPU/GPU arenas and workspace scratch buffers.

## Error Enums

`CORE::errIO` is used by `HANDLER::IO`, `HANDLER::Cpu`, and `HANDLER::Cuda`.

`CORE::errFile` is used by `HANDLER::File`.

`CORE::errWorkspace` is used by `HANDLER::Workspace`.

The pattern is:

```cpp
io.copyHostToDevice(x);
if(io.peekErr() != CORE::ioSuccess) {
  io.info();
}
```

`getErr()` returns and clears an error. `peekErr()` returns without clearing.

## Logging

`CORE::logInfo`, `CORE::logWarn`, and `CORE::logFatal` call `CORE::printState`.

Fatal logs are used for unrecoverable initialization failures such as failing to allocate the CPU or GPU arena.

---

> **Continue Reading**  
> Previous: [Project Guide](index.md) | Next: [VIEW](view.md)
