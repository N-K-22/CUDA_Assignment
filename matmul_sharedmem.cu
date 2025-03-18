#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define _XOPEN_SOURCE 700
long long nsecs() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec*1000000000 + t.tv_nsec;
}

__global__ void matrixMultiplication ( int * A , int * B , int * C , int width ) {
    // TODO : Implement matrix multiplication kernel
    int component = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ int sharedA[16][16];
    __shared__ int sharedB[16][16];
    //int sum = 0; //each thread does one component in the output matrix
   for (int i = 0; i < (width+15)/16; i++){
        
        if (row < width && (i*16 + threadIdx.x) < width){ // sets the component 

            sharedA[threadIdx.y][threadIdx.x] = A[row*width + i*16 + threadIdx.x]; //if the component falls within the area of shared memory covered by a thread

        } else{
            sharedA[threadIdx.y][threadIdx.x] = 0; // leave other parts blank
        }

        if ((i*16 + threadIdx.y) < width && col < width){
            sharedB[threadIdx.y][threadIdx.x] = B[(i*16 + threadIdx.y)*width + col]; //if matrix coordinate falls within the area of shared memory covered by a thread
        }
        else{
            sharedB[threadIdx.y][threadIdx.x] = 0;
        }
        __syncthreads();
        for (int  i = 0; i< 16; i++){
            component += sharedA[threadIdx.y][i] * sharedB[i][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < width && col < width){
    
            C[row*width + col] = component;
       

    }
   
}
void matrixMultiplicationCPUVersion(int* a, int* b, int* c, int width){
    int component;
    for(int row = 0; row < width; row++){
        for(int column = 0; column < width; column++){
            component = 0;
            for(int i = 0; i < width; i++){
                component += a[row*width + i] * b[i*width+column];
            }
            c[row*width +column] = component;
        }
    }

}
int main () {
    const int width = 128; // Matrix width
    int A [ width * width ] , B [ width * width ] , C [ width * width ], D [width*width]; // Host matrices
    int *d_A, *d_B, *d_C;

    // TODO : Initialize matrices ’A’ and ’B’ with random values
    for (int i =0; i < width*width; i++){
        A[i] = rand();
        B[i] = rand();
    }
    cudaEvent_t GPU_start, GPU_stop;
    float GPU_time = 0;
    cudaEventCreate(&GPU_start);
    cudaEventCreate(&GPU_stop);
    cudaEventRecord(GPU_start);

    // TODO : Allocate device memory for matrices ’A ’, ’B ’, and ’C ’
    //int *d_A;
   // int *d_B;
   // int *d_C;
   // int *d_D;
    
    cudaMallocManaged((void **)&d_A, width*width*sizeof(int));
    cudaMallocManaged((void**)&d_B, width*width*sizeof(int));
    cudaMallocManaged((void**)&d_C, width*width*sizeof(int));
    //cudaMallocManaged((void**)&d_D, width*width*sizeof(int));

    // TODO : Copy matrices 'A' and 'B' from host to device

    cudaMemcpy(d_A, A, width * width * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,B, width * width * sizeof(int), cudaMemcpyHostToDevice);


    // Define grid and block dimensions
    dim3 dimGrid (( width + 15) / 16 , ( width + 15) / 16 , 1);
    dim3 dimBlock (16 , 16 , 1);

    // Launch the matrix multiplication kernel
    matrixMultiplication<<<dimGrid , dimBlock, 2*16*16*sizeof(int) >>>(d_A , d_B , d_C , width );

    // TODO : Copy the result matrix ’C ’ from device to host
    cudaMemcpy(C, d_C, width*width*sizeof(int), cudaMemcpyDeviceToHost);

    //timing the GPU kernel
    cudaEventRecord(GPU_stop);
    cudaEventSynchronize (GPU_stop); //waits for GPU_stop to complete
    cudaEventElapsedTime(&GPU_time, GPU_start, GPU_stop);
    printf("Time taken for GPU version: %ld ms\n", GPU_time);



//CPU Matrix Multiplication
    long long start, end;
    long long cpu_time_used;
    start = nsecs();
    matrixMultiplicationCPUVersion(A, B, D, width);
    end = nsecs();
    cpu_time_used = end - start;
    printf("Time taken for CPU version: %ld ns\n", cpu_time_used);
    //memcpy(D, d_D, width*width*sizeof(int));

    // TODO : Verify the correctness of the result


    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess){
        printf("Error: %s\n", cudaGetErrorString(error));
    }
    
    for (int i = 0; i < width*width; i++){
        if(D[i] != C[i]){
            printf("Value for CPU and GPU does not align at %d \n", i);
            printf("C: %d, D: %d \n", C[i], D[i]);

            break;
        }
    }
    

    // TODO : What is needed here ? --> Free memory

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    //free(d_D);
    return 0;

}
