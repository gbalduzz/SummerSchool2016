#pragma once
#include <cuda.h>
#include <vector>

template <typename T>
class cudaVector{
  cudaVector(const std::vector<T>& v);
  cudaVector(T* v, int s);
  ~cudaVector();
  operator T*(){return data;}
  void operator =(T* v, int s);
  void operator =(const std::vector<T>& v);
  void copyTo(T* v);
  void copyTo(const std::vecotr<T>& v);
private:
  T* data;
  int size;
  int byte_size;
};

template <typename T>
cudaVector<T>::cudaVector(T* v, int s){
  size = s;
  byte_size = s * sizeof(T);
  cudaError err = cudaMalloc(&data, byte_size);
  if(err != cudaSuccess){
    std::cout<<"ERROR: Allocation on device failed\n";
    throw(std::malloc());
  }
  cudaMemcpy (data, v, byte_size, cudaMemcpyHostToDevice);
}

template <typename T>
cudaVector<T>::cudaVector(const std::vector<T>& v):
  cudaVector(v.data(), v.size())
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
void cudaVector<T>::copyTo(const std::vector<T>& v){
  assert(v.size == size);
  cudaMemcpy (v.data(), data, byte_size, cudaMemcpyDeviceToHost);
}
