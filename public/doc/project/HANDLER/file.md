# HANDLER::File

> Namespace: [HANDLER](index.md)

File and dataset helper. It reads raw files, CSV metadata, and images into `VIEW::Math` tensors.

## Constructors

```cpp
File();
File(HANDLER::IO& io);
```

The second constructor attaches IO immediately.

## `setIO`

```cpp
void setIO(HANDLER::IO& io);
```

Attach or replace IO handler.

## `read`

```cpp
void read(VIEW::Math& dst, const char *path, VIEW::DType dtype = VIEW::CHAR);
```

Reads raw file data into CPU tensor memory.

Errors may include open/read/close failures, null path, or IO errors.

## `write`

```cpp
void write(VIEW::Math& src, const char *path);
```

Writes CPU tensor memory to a raw file.

## `readCsv`

```cpp
void readCsv(VIEW::Math& filePaths, VIEW::Math& labels, const char *path);
```

Reads CSV metadata into path and label tensors.

## `readImages`

```cpp
void readImages(VIEW::Math& filePaths, const char *rootPath = NULL, int desiredChannels = 0);
```

Reads image files listed by `filePaths`.

## `copyImageToDevice`

```cpp
void copyImageToDevice();
```

Converts loaded uint8 CPU image data to F16 GPU image data.

## `pullCpuImage`

```cpp
void pullCpuImage(VIEW::Math& dst, size_t n, size_t offset = 0);
```

Exposes a CPU image batch as a `VIEW::Math`.

## `pullGpuImage`

```cpp
void pullGpuImage(VIEW::Math& dst, size_t n, size_t offset = 0);
```

Exposes a GPU image batch as a `VIEW::Math`.

## Image Metadata Getters

```cpp
size_t getImageN() const;
int getImageHeight() const;
int getImageWidth() const;
int getImageChannel() const;
```

Return loaded image metadata.

## Error Functions

```cpp
CORE::errFile getErr();
CORE::errFile peekErr() const;
void clearErr();
void info(CORE::State level = CORE::WARN, const char *file = __FILE__, int line = __LINE__) const;
```

Same pattern as `IO`.

