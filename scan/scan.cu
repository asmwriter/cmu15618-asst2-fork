#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include "CycleTimer.h"

#define TPB 8
#define DEBUG 1


extern float toBW(int bytes, float sec);


/* Helper function to round up to a power of 2.
 */
static inline int nextPow2(int n)
{
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

__global__ void exclusive_scan_kernel (int length, int* in_array, int* next_chunk_sum) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
   
    int threadIndex = threadIdx.x;

    extern __shared__ int temp[];

    // //Load input into the shared memory
    temp[threadIndex] = in_array[index];
    //temp[threadIndex+1] = in_array[index+1];

    int offset = 1, active = 2;
    
    //Up-sweep
    for(int d = length/2; d>0; d=d/2) {
        if(threadIndex % active == active-1){

            // __syncthreads();             

            temp[threadIndex] += temp[threadIndex-offset];

            offset*=2;
            active*=2;
        }
    }

   
    __syncthreads();

    
    if(threadIndex == length-1) {
        next_chunk_sum[index] = temp[length-1];
        temp[length-1] = 0;
    }
    
    
    //Down-sweep

    active = length;
    offset = length/2;
    
     for(int d = 1; d<length; d*=2) {
         __syncthreads();

         if(threadIndex % active == active-1) {
             int cur = threadIndex;
             int next = threadIndex-offset;

             int t = temp[cur];
             temp[cur] += temp[next];
             temp[next] = t;
         }

         offset /= 2;
         active /= 2;
     }
     
    __syncthreads();
    in_array[index] = temp[threadIndex];
}

void exclusive_scan(int* device_data, int length, int* device_next_chunk_sum)
{
    /* TODO
     * Fill in this function with your exclusive scan implementation.
     * You are passed the locations of the data in device memory
     * The data are initialized to the inputs.  Your code should
     * do an in-place scan, generating the results in the same array.
     * This is host code -- you will need to declare one or more CUDA
     * kernels (with the __global__ decorator) in order to actually run code
     * in parallel on the GPU.
     * Note you are given the real length of the array, but may assume that
     * both the data array is sized to accommodate the next
     * power of 2 larger than the input.
     */
    int pow_length = nextPow2(length);
    int chunk_size = TPB;
    int num_block;
    int threads_per_block = TPB;
    
    num_block = pow_length/chunk_size;

    printf("num block %d", num_block);

    exclusive_scan_kernel<<<num_block, threads_per_block, threads_per_block*sizeof(int)>>>(chunk_size, device_data, device_next_chunk_sum);
    cudaThreadSynchronize();
    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
    }

    return;
}

/* This function is a wrapper around the code you will write - it copies the
 * input to the GPU and times the invocation of the exclusive_scan() function
 * above. You should not modify it.
 */
double cudaScan(int* inarray, int* end, int* resultarray)
{
    //GPU pointer for input array
    int* device_data;
    //GPU pointer for holding upsweep sum of intermediate blocks/scanned upsweep sums
    int* device_inter_sum_array;
    
    // We round the array size up to a power of 2, but elements after
    // the end of the original input are left uninitialized and not checked
    // for correctness.
    // You may have an easier time in your implementation if you assume the
    // array's length is a power of 2, but this will result in extra work on
    // non-power-of-2 inputs.
    int rounded_length = nextPow2(end - inarray);

    //Allocate GPU memory for input array
    cudaMalloc((void **)&device_data, sizeof(int) * rounded_length);
    //Allocate GPU memory for holding upsweep reduced sums/scanned sums
    cudaMalloc((void **)&device_inter_sum_array, sizeof(int) * rounded_length);
    
    //Copy input array and next chunk temp array to GPU
    cudaMemcpy(device_data, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);
    
    //Initialise upsweep sum array and scanned sum array
    cudaMemset(device_inter_sum_array,0,rounded_length*sizeof(int));

    double startTime = CycleTimer::currentSeconds();

    //Perform exclusive scan on input array
    exclusive_scan(device_data, end - inarray, device_inter_sum_array);

    // Wait for any work left over to be completed.
    cudaThreadSynchronize();

    //Perform exclusive scan on upsweep sum array
    exclusive_scan(device_inter_sum_array, end - inarray, nullptr);

    //Perform vector sum of scanned input and scanned sum array


    double endTime = CycleTimer::currentSeconds();
    double overallDuration = endTime - startTime;

    //Transfer input and next chunk sum array from GPU to CPU
    cudaMemcpy(resultarray, device_data, (end - inarray) * sizeof(int),
               cudaMemcpyDeviceToHost);
    
    #ifdef DEBUG
        int* inter_sum_array;
        inter_sum_array = new int[rounded_length];
        cudaMemcpy(inter_sum_array, device_inter_sum_array, rounded_length * sizeof(int),
                cudaMemcpyDeviceToHost);
        
        printf("/*DEBUG - RESULT ARRAY*/ ");
        for(int idx = 0; idx < rounded_length; idx++){
            printf("resultarray[%idx]=%d\n",idx,resultarray[idx]);
        }
        
        printf("/*DEBUG - UPSWEEP SCANNED SUM ARRAY*/ ");
        for(int idx = 0; idx < rounded_length; idx++){
            printf("inter_sum_array[%idx]=%d\n",idx,inter_sum_array[idx]);
        }
        delete[] inter_sum_array;
    #endif

    return overallDuration;
}

/* Wrapper around the Thrust library's exclusive scan function
 * As above, copies the input onto the GPU and times only the execution
 * of the scan itself.
 * You are not expected to produce competitive performance to the
 * Thrust version.
 */
double cudaScanThrust(int* inarray, int* end, int* resultarray) {

    int length = end - inarray;
    thrust::device_ptr<int> d_input = thrust::device_malloc<int>(length);
    thrust::device_ptr<int> d_output = thrust::device_malloc<int>(length);

    cudaMemcpy(d_input.get(), inarray, length * sizeof(int),
               cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    thrust::exclusive_scan(d_input, d_input + length, d_output);

    cudaThreadSynchronize();
    double endTime = CycleTimer::currentSeconds();

    cudaMemcpy(resultarray, d_output.get(), length * sizeof(int),
               cudaMemcpyDeviceToHost);
    thrust::device_free(d_input);
    thrust::device_free(d_output);
    double overallDuration = endTime - startTime;
    return overallDuration;
}



int find_peaks(int *device_input, int length, int *device_output) {
    /* TODO:
     * Finds all elements in the list that are greater than the elements before and after,
     * storing the index of the element into device_result.
     * Returns the number of peak elements found.
     * By definition, neither element 0 nor element length-1 is a peak.
     *
     * Your task is to implement this function. You will probably want to
     * make use of one or more calls to exclusive_scan(), as well as
     * additional CUDA kernel launches.
     * Note: As in the scan code, we ensure that allocated arrays are a power
     * of 2 in size, so you can use your exclusive_scan function with them if
     * it requires that. However, you must ensure that the results of
     * find_peaks are correct given the original length.
     */

    cudaScan(device_input, device_input+length, device_output);
    return 0;
}



/* Timing wrapper around find_peaks. You should not modify this function.
 */
double cudaFindPeaks(int *input, int length, int *output, int *output_length) {
    int *device_input;
    int *device_output;
    int rounded_length = nextPow2(length);
    cudaMalloc((void **)&device_input, rounded_length * sizeof(int));
    cudaMalloc((void **)&device_output, rounded_length * sizeof(int));
    cudaMemcpy(device_input, input, length * sizeof(int),
               cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    int result = find_peaks(device_input, length, device_output);

    cudaThreadSynchronize();
    double endTime = CycleTimer::currentSeconds();

    *output_length = result;

    cudaMemcpy(output, device_output, length * sizeof(int),
               cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);

    return endTime - startTime;
}


void printCudaInfo()
{
    // for fun, just print out some stats on the machine

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++)
    {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
}
