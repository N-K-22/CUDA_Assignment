#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define _XOPEN_SOURCE 700
// CUDA kernel for matrix addition
__global__ void matrixAddition ( int * A , int * B , int * C , int width , int height ) {
    int row = blockIdx . y * blockDim . y + threadIdx . y ;
    int col = blockIdx . x * blockDim . x + threadIdx . x ;
    if ( row < height && col < width ) {
        C [ row * width + col ] = A [ row * width + col ] +
        B [ row * width + col ];
        }
}
void matrixAdditionCPUVersion(int* a, int* b, int* c, int number_of_iterations){
    for (int i = 0; i < number_of_iterations; i++){
    c[i] = a[i] + b[i]; //adds matrix values at the same index and stores it in the resulting matrix
    }
}
long long nsecs() { //timing of CPU kernel
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec*1000000000 + t.tv_nsec;
}

int main () {
    const int width = 128; // Matrix width
    const int height = 128; // Matrix height
    int A[width * height] , B[width * height] , C[width * height], D[width * height];
    int size = width * height; //holder variable
    // TODO : Initialize matrices ’A ’ and ’B ’ with random values (host matrices)
    for (int i =0; i < size; i++){
        A[i] = rand();
        B[i] = rand();
    }
    size = size * sizeof(int);
    // TODO : Declare pointers for device matrices
    int *d_A;
    int *d_B;
    int *d_C;
    int *d_D;
    cudaEvent_t GPU_start, GPU_stop;
    float GPU_time = 0;
    cudaEventCreate(&GPU_start);
    cudaEventCreate(&GPU_stop);
    cudaEventRecord(GPU_start);
    // TODO : Allocate device memory for matrices ’A ’, ’B ’, and ’C ’
    cudaMallocManaged((void **)&d_A, width*height*sizeof(int));
    cudaMallocManaged((void**)&d_B, width*height*sizeof(int));
    cudaMallocManaged((void**)&d_C, width*height*sizeof(int));
    
    // TODO : Copy matrices ’A ’ and ’B ’ from host to device
    cudaMemcpy(d_A,A,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    
    // Define grid and block dimensions
    dim3 dimGrid (( width + 15) / 16 , ( height + 15) / 16 , 1);
    dim3 dimBlock (16 , 16 , 1);
    
    // Launch the matrix addition kernel
    matrixAddition <<<dimGrid,dimBlock>>> (d_A , d_B , d_C , width , height );
    
    // TODO : Copy the result matrix ’C ’ from device to host

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(GPU_stop);
    cudaEventSynchronize (GPU_stop); //waits for GPU_stop to complete --> ensures that the GPU kernel has finished executing before calculating the completion time
    cudaEventElapsedTime(&GPU_time, GPU_start, GPU_stop);
    printf("Time taken for GPU version: %f ms\n", GPU_time);
   
    //CPU KERNEL Modifications for execution
    long long start, end;
    long long cpu_time_used;
    d_D = (int*)malloc(sizeof(int)*width*height); //allocating memory for the CPU version of the resulting matrix
    start = nsecs(); //CPU kernel time start
    matrixAdditionCPUVersion(A,B,d_D,height*width); //CPU matrix addition function
    memcpy(D, d_D, sizeof(int)*width*height); // storing the result of hte matrix for the CPU version
    end = nsecs(); //CPU kernel time end
    cpu_time_used = end - start;
    printf("Time taken for CPU version: %d ns\n", cpu_time_used);
    
    
    // TODO : Verify the correctness of the result
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess){
        printf("Error: %s\n", cudaGetErrorString(error));
    }//ensures that the GPU kernel ran successfully and there were no errors in the kernel call that were masked by the progra,
    
    for (int i = 0; i < width*height;i++){ //verify outputs for CPU and GPU
    
        if (D[i] != C[i]){
            printf("Value for CPU and GPU does not align at %d \n", i);
            break;
        }
    }
    
    // TODO : Free allocated memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(d_D);
    return 0;
}
