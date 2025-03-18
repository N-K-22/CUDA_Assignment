#include <stdio.h>
#include <stdlib.h>
#include  <omp.h>
#include <time.h>
#define _XOPEN_SOURCE 700
long long nsecs() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec*1000000000 + t.tv_nsec;
}
__global__ void parallelSum(int* inputArray, int* outputResult, int arraySize) {
    extern __shared__ int sharedMemory[];
    int threadID = threadIdx.x;
    int globalID = blockIdx.x * blockDim.x + threadIdx.x;
    // Load data into shared memory
    sharedMemory[threadID] = (globalID < arraySize) ? inputArray[globalID] : 0;
    __syncthreads();
    // Perform parallel reduction using shared memory
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int index = 2 * stride * threadID;
        if (index < blockDim.x) {
            sharedMemory[index] += sharedMemory[index + stride];
        }
        __syncthreads();
    }

    // Write the result back to global memory
    if (threadID == 0) {
        outputResult[blockIdx.x] = sharedMemory[0];
    }
}
void parallelSumCPUVersion(int* input,int* output ,int size){
    int sum = 0;
    #pragma omp parallel  for reduction (+:sum)
    {
        for (int  i = 0; i < size; i++){
            sum += input[i];
        }
        //#pragma omp atomic update
        *output = sum;
    }
   
}
int main() {
    // TODO : Write the main() function

    const int width = 65536;
    int A [ width ] , B [ width  ] , C [ width ];
    int *d_A;
    int *d_B;
    for(int index = 0; index < width; index++){ A[index] = rand();}
    //start of cuda event record start time creation
    cudaEvent_t GPU_start, GPU_stop;
    float GPU_time = 0;
    cudaEventCreate(&GPU_start);
    cudaEventCreate(&GPU_stop);
    cudaEventRecord(GPU_start);

    cudaMallocManaged((void**)&d_A, width*sizeof(int));
    cudaMemcpy(d_A, A, width*sizeof(int), cudaMemcpyHostToDevice);
    
    cudaMallocManaged((void**)&d_B, width*sizeof(int));
    cudaMemset(d_B, 0, width*sizeof(int)); //initalize empty space to zero just for sanity purposes and to make debugging semi-easier
    
    // Define grid and block dimensions
    dim3 dimGrid (( width + 255) / 256 , 1 , 1); //grid size --> blocks that will be executed in parallel
    dim3 dimBlock (16*16 , 1 , 1); //block size --> group of threads that execute togther in shared memory --> takes the 16 from the y-dimension/second parameter to standardize the number of threads running to 256, this can be changed as needed in the code
    int sharedMemorySpace = 16*16 * sizeof(int);
//Call parallel sum
    parallelSum<<<dimGrid, dimBlock, sharedMemorySpace>>> (d_A, d_B,width);
    cudaMemcpy(B, d_B, width*sizeof(int), cudaMemcpyDeviceToHost);
    int gpu_sum = 0;
    for (int i = 0; i < ((width + 255) / 256); i++) { //was originally missing this part of it, the shared memory resulted in all of the values being spread out in output array
        gpu_sum += B[i];
    }
    printf("GPU sum: %d\n", gpu_sum); //debugging purposes
    //timing the GPU kernel
    cudaEventRecord(GPU_stop);
    cudaEventSynchronize (GPU_stop); //waits for GPU_stop to complete
    cudaEventElapsedTime(&GPU_time, GPU_start, GPU_stop);
    printf("Time taken for GPU version: %ld ms\n", GPU_time);
//CPU parallelSum
    //int* d_C = (int*)malloc(sizeof(int) * width);
    long long start = nsecs();
    parallelSumCPUVersion(A,C,width);
    long long end = nsecs();
    printf("Time taken for CPU parallel sum kernel: %ld\n", (end-start));

    
    cudaError_t error = cudaDeviceSynchronize();
    if (error != cudaSuccess){
        printf("Error: %s\n", cudaGetErrorString(error));
    }
    
    for (int i = 0; i < width; i++){ //extraneous for-loop 
        if( gpu_sum != C[0]){
            printf("Value for CPU and GPU does not align at %d \n", i);
            printf("CPU: %d\n", C[i]);
            printf("GPU: %d\n", B[i]);
            break;
        }
    }
    cudaFree(d_A);
    cudaFree(d_B);
    return 0;
}
