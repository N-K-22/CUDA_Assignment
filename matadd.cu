#include <stdio.h>
#include <stdlib.h>
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
    c[i] = a[i] + b[i];
    }
}
int main () {
    const int width = 128; // Matrix width
    const int height = 128; // Matrix height
    int A[width * height] , B[width * height] , C[width * height], D[width * height];
    int size = width * height;
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
    d_D = (int*)malloc(sizeof(int)*width*height);
    
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
    cudaEventSynchronize (GPU_stop); //waits for GPU_stop to complete
    cudaEventElapsedTime(&GPU_time, GPU_start, GPU_stop);
    printf("Time taken for GPU version: %f ms\n", GPU_time);
    //CPU Modifications
    clock_t start, end;
    double cpu_time_used;
    start = clock();
    matrixAdditionCPUVersion(A,B,d_D,height*width); //CPU matrix addition function
    memcpy(D, d_D, sizeof(int)*width*height); // storing the result of hte matrix for the CPU version
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC * 1000;
    printf("Time taken for CPU version: %f ms\n", cpu_time_used);
    
    
    // TODO : Verify the correctness of the result
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess){
        printf("Error: %s\n", cudaGetErrorString(error));
    }
    
    for (int i = 0; i < width*height;i++){
    
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
