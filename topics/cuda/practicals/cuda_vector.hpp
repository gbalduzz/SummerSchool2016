#pragma once
#include <cuda.h>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <assert.h>

template <typename T>
class cudaVector{
public: 
  cudaVector(const std::vector<T>& v);
  cudaVector(const T* v, int s);
  ~cudaVector();
  operator T*(){return data;}
  void copyTo(T* v);
  void copyTo(std::vector<T>& v);
private:
  T* data;
  int size;
  int byte_size;
};

template <typename T>
cudaVector<T>::cudaVector(const T* v, int s){
  size = s;
  byte_size = s * sizeof(T);
  cudaError err = cudaMalloc(&data, byte_size);
  if(err != cudaSuccess){
    throw(std::logic_error("ERROR: Allocation on device failed.\n"));
  }
  cudaMemcpy (data, v, byte_size, cudaMemcpyHostToDevice);
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
  cudaMemcpy (v, data, byte_size, cudaMemcpyDeviceToHost);
}

template <typename T>
void cudaVector<T>::copyTo(std::vector<T>& v){
  assert(v.size() == size);
  cudaMemcpy(&v[0], data, byte_size, cudaMemcpyDeviceToHost);
}
