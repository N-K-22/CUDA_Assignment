#include <stdio.h>
#include <stdlib.h>
#include  <omp.h>
 
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
        #pragma omp atomic update
        output = &sum;
        
    }
}
int main() {
    // TODO : Write the main() function

    const int width = 65536;
    int A [ width ] , B [ width  ] , C [ width ];
    int *d_A;
    int *d_B;
    for(int index = 0; index < width; index++){ A[index] = rand();}
    cudaMallocManaged((void**)&d_A, width*sizeof(int));
    cudaMemcpy(d_A, A, width*sizeof(int), cudaMemcpyHostToDevice);
    
    cudaMallocManaged((void**)&d_B, width*sizeof(int));
    cudaMemset(d_B, 0, width*sizeof(int)); //initalize empty space to zero just for sanity purposes and to make debugging semi-easier

    int* d_C = (int*)malloc(sizeof(int) * width);
    
    // Define grid and block dimensions
    dim3 dimGrid (( width + 15) / 16 , 1 , 1); //grid size --> blocks that will be executed in parallel
    dim3 dimBlock (16*16 , 1 , 1); //block size --> group of threads that execute togther in shared memory --> takes the 16 from the y-dimension/second parameter to standardize the number of threads running to 256, this can be changed as needed in the code
    int sharedMemorySpace = width * sizeof(int);
//Call parallel sum
    parallelSum<<<dimGrid, dimBlock, sharedMemorySpace>>> (d_A, d_B,width);
    cudaMemcpy(B, d_B, width*sizeof(int), cudaMemcpyDeviceToHost);
    
//CPU parallelSum
    parallelSumCPUVersion(A,C,width);

    
    cudaError_t error = cudaDeviceSynchronize();
    if (error != cudaSuccess){
        printf("Error: %s\n", cudaGetErrorString(error));
    }
    
    for (int i = 0; i < width; i++){
        if(B[i] != C[i]){
            printf("Value for CPU and GPU does not align at %d \n", i);
            break;
        }
    }
    cudaFree(d_A);
    cudaFree(d_B);
    free(d_C);
    return 0;
}
