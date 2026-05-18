# VIEW::Shape

> Namespace: [VIEW](index.md)

`Shape` stores tensor layout metadata.

## Fields

```cpp
int dims[VIEW::MAX_RANK];
int strides[VIEW::MAX_RANK];
int rank;
VIEW::DType dtype;
```

## `VIEW::DType`

```cpp
CHAR
F16
FLOAT
```

The enum values equal byte size.

## Constructors

```cpp
Shape(int dims[VIEW::MAX_RANK], int rank = 0, VIEW::DType dtype = VIEW::CHAR, const char* file = __FILE__, int line = __LINE__);
Shape();
```

The full constructor stores dims/rank/dtype and computes strides.

## Destructor

```cpp
~Shape();
```

No memory ownership.

## Getters

```cpp
int getDim(int index) const;
int getStride(int index) const;
int getRank() const;
VIEW::DType getDType() const;
int getMaxRank() const;
```

Return stored metadata.

## Setters

```cpp
void setDim(int dims[VIEW::MAX_RANK], const char* file, int line);
void setRank(int rank, const char* file, int line);
void setDtype(VIEW::DType dtype);
void setShape(int dims[VIEW::MAX_RANK], int rank = 0, VIEW::DType dtype = VIEW::CHAR, const char* file = __FILE__, int line = __LINE__);
```

`setShape` is the main function. It clears old metadata, stores new dims/rank/dtype, and recomputes strides.

## `getSuperPosition`

```cpp
int getSuperPosition(int i = 0, int j = 0, int k = 0, int l = 0) const;
```

Returns byte offset for an indexed element.

## `info`

```cpp
void info(const char* file = __FILE__, int line = __LINE__) const;
```

Prints shape metadata.

