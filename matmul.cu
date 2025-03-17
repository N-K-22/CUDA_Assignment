#include <stdio.h>
#include <stdlib.h>
__global__ void matrixMultiplication ( int * A , int * B , int * C , int width ) {
    // TODO : Implement matrix multiplication kernel
    int component = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    //int sum = 0; //each thread does one component in the output matrix
    if (row < width && col < width){
        for (int  i = 0; i< width; i++){
            component += A[row*width + i] * B[row*width + i];
        }
        C[row*width + col] = component; // location of where the value is stored in resulting matrix
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

    // TODO : Allocate device memory for matrices ’A ’, ’B ’, and ’C ’
    //int *d_A;
   // int *d_B;
   // int *d_C;
    int *d_D;
    
    cudaMallocManaged((void **)&d_A, width*width*sizeof(int));
    cudaMallocManaged((void**)&d_B, width*width*sizeof(int));
    cudaMallocManaged((void**)&d_C, width*width*sizeof(int));
    cudaMallocManaged((void**)&d_D, width*width*sizeof(int));

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



//CPU Matrix Multiplication

    matrixMultiplicationCPUVersion(d_A, d_B, d_D, width);

    memcpy(D, d_D, width*width*sizeof(int));

    // TODO : Verify the correctness of the result


    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess){
        printf("Error: %s\n", cudaGetErrorString(error));
    }
    
    for (int i = 0; i < width*width; i++){
        if(D[i] != C[i]){
            printf("Value for CPU and GPU does not align at %d \n", i);
            break;
        }
    }
    

    // TODO : What is needed here ? --> Free memory

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(d_D);
    return 0;

}
