#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define _XOPEN_SOURCE 700
long long nsecs() { //code for timing CPU kernel
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec*1000000000 + t.tv_nsec;
}

__global__ void matrixMultiplication ( int * A , int * B , int * C , int width ) { //GPU Kernel for matrix multiplication
    // TODO : Implement matrix multiplication kernel
    int component = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y; //getting row values
    int col = blockIdx.x * blockDim.x + threadIdx.x; //getting column values
    //int sum = 0; //each thread does one component in the output matrix
    if (row < width && col < width){ //ensure that the row and column are within the specified bounds of width so it is not accessing values outside the matrix memory space
        for (int  i = 0; i< width; i++){
            component += A[row*width + i] * B[i*width + col]; // multiplying the row value with different indices of column and summing it all together to place into the output matrix
        }
        C[row*width + col] = component; // location of where the value is stored in resulting matrix
    }
}
void matrixMultiplicationCPUVersion(int* a, int* b, int* c, int width){ //CPU version of matrix multiplication
    int component;
    for(int row = 0; row < width; row++){
        for(int column = 0; column < width; column++){
            component = 0;
            for(int i = 0; i < width; i++){
                component += a[row*width + i] * b[i*width+column]; //completing the same steps are the GPU version except it requires 2 for loops to cycle through the row and column
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
    for (int i =0; i < width*width; i++){ //filling the matrices with random values
        A[i] = rand();
        B[i] = rand();
    }
    cudaEvent_t GPU_start, GPU_stop;
    float GPU_time = 0;
    cudaEventCreate(&GPU_start);
    cudaEventCreate(&GPU_stop); 
    cudaEventRecord(GPU_start); //starts recording the GPU kernel time

    // TODO : Allocate device memory for matrices ’A ’, ’B ’, and ’C ’
   
    cudaMallocManaged((void **)&d_A, width*width*sizeof(int));
    cudaMallocManaged((void**)&d_B, width*width*sizeof(int));
    cudaMallocManaged((void**)&d_C, width*width*sizeof(int));

    // TODO : Copy matrices 'A' and 'B' from host to device

    cudaMemcpy(d_A, A, width * width * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,B, width * width * sizeof(int), cudaMemcpyHostToDevice);


    // Define grid and block dimensions
    dim3 dimGrid (( width + 15) / 16 , ( width + 15) / 16 , 1);
    dim3 dimBlock (16 , 16 , 1);

    // Launch the matrix multiplication kernel
    matrixMultiplication<<<dimGrid , dimBlock >>>(d_A , d_B , d_C , width );

    // TODO : Copy the result matrix ’C ’ from device to host
    cudaMemcpy(C, d_C, width*width*sizeof(int), cudaMemcpyDeviceToHost);

    //timing the GPU kernel
    cudaEventRecord(GPU_stop);
    cudaEventSynchronize (GPU_stop); //waits for GPU_stop to complete
    cudaEventElapsedTime(&GPU_time, GPU_start, GPU_stop);
    printf("Time taken for GPU version: %ld ms\n", GPU_time); //prints out GPU kernel time

//CPU Matrix Multiplication
    long long start, end;
    long long cpu_time_used;
    start = nsecs(); //starts measuring CPU kernel time
    matrixMultiplicationCPUVersion(A, B, D, width);
    end = nsecs(); //ends measuring CPU kernel time
    cpu_time_used = end - start;
    printf("Time taken for CPU version: %ld ns\n", cpu_time_used);

    // TODO : Verify the correctness of the result


    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess){
        printf("Error: %s\n", cudaGetErrorString(error));
    } //ensures that the GPU kernel ran successfully and there were no errors in the kernel call that were masked by the program
    
    for (int i = 0; i < width*width; i++){ //verifies the CPU and GPU outputs, if they are the same that means that the code sucessfully executed
        if(D[i] != C[i]){
            printf("Value for CPU and GPU does not align at %d \n", i);
            break;
        }
    }
    

    // TODO : What is needed here ? --> Free memory

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;

}
