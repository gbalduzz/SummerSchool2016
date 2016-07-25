#pragma once
#include <cuda.h>
#include <vector>
#include <stdexcept>
#include <assert.h>

template <typename T>
class cudaVector{
public:
  cudaVector(const int n);
  cudaVector(const std::vector<T>& v);
  cudaVector(const T* v, int s);
  ~cudaVector();
  operator T*(){return data;}
  void copyTo(T* v);
  void copyTo(std::vector<T>& v);
  void copyFrom(T* v);
  void copyFrom(const std::vector<T>& v);
private:
  T* data;
  int size;
};

template <typename T>
cudaVector<T>::cudaVector(const int n){
  size = n;
  cudaError_t status = cudaMalloc(&data, size * sizeof(T));
  if(status != cudaSuccess){
     throw(std::runtime_error("ERROR: cuda could not allocate memory"));
}
}

template <typename T>
cudaVector<T>::cudaVector(const T* v, int n):
cudaVector(n)
{
   cudaMemcpy (data, v, size * sizeof(T), cudaMemcpyHostToDevice);
}

template <typename T>
cudaVector<T>::cudaVector(const std::vector<T>& v):
  cudaVector(v.data(), (int) v.size())
{}

template <typename T>
cudaVector<T>::~cudaVector(){
  cudaFree(data);
}

template <typename T>
void cudaVector<T>::copyTo(T* v){
  cudaMemcpy (v, data, size * sizeof(T), cudaMemcpyDeviceToHost);
}

template <typename T>
void cudaVector<T>::copyTo(std::vector<T>& v){
  assert(v.size() == size);
  cudaMemcpy(&v[0], data, size * sizeof(T), cudaMemcpyDeviceToHost);
}

template <typename T>
void cudaVector<T>::copyFrom(T* v){
  cudaMemcpy (data, v,size * sizeof(T), cudaMemcpyHostToDevice);
}

template <typename T>
void cudaVector<T>::copyFrom(const std::vector<T>& v){
  assert(v.size() == size);
  cudaMemcpy (data, v.data(), size * sizeof(T), cudaMemcpyHostToDevice);
}
