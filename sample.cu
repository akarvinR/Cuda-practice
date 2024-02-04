#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <chrono>
#include <opencv2/opencv.hpp> 
const int n = 500'000'000;
template<typename T>
__global__ void vectorAdd(T* a, T* b, T* c, int numElements){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < numElements){
        c[i] = b[i] + a[i];
    }
}
template<typename T>
void vectorAddCPU(T *a, T *b ,T *c, int numElements){
    for(int i = 0;i<numElements;i++){
        c[i] = a[i] + b[i];
    }
}

bool check(float *c, float *a, float *b){
    for(int i = 0;i<n;i++){
        if(a[i] + b[i] != c[i]){
            std::cout << a[i] << " " << b[i] << " " << c[i] << std::endl;
            return false;
        }
    }
    return true;
}

void fill(float *a){
    for(int i = 0;i<n;i++){
        a[i] = rand()%1000000; 
    }
}
int main(){
    srand (time(NULL));
    
    cv::Mat image = cv::imread("./eagle.jpeg"); 
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float* a = new float[n];
    float* b = new float[n];
    float *c = new float[n];
    float *rc = new float[n];
    float *d_a, *d_b, *d_c; 

    fill(a);
    fill(b);
    
    size_t size = sizeof(float)*n;

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c, size, cudaMemcpyHostToDevice);


    cudaEventRecord(start);

    int threadSize = 1024;
    int blockSize = n/threadSize + 1;
    vectorAdd<float><<<blockSize, threadSize>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    if(check(c, a, b)){
        std::cout << "\nSucccesss in" << " " << milliseconds;
    }else{
        std::cout << "\n Failed in" << " " << milliseconds;
    }

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    vectorAddCPU<float>(rc, a, b, n);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << " CPU TIME :" << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;   
}

