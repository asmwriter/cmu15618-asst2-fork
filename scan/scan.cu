#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include "CycleTimer.h"

#define TPB 1024
//#define DEBUG 1
//#define FIND_PEAK_DEBUG

#ifdef DEBUG
#define cudaCheckError(ans) cudaAssert((ans), __FILE__, __LINE__);
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
{
fprintf(stderr, "CUDA Error: %s at %s:%d\n",
cudaGetErrorString(code), file, line);
if (abort) exit(code);
}
}
#else
#define cudaCheckError(ans) ans
#endif

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

__global__ void scalar_vector_sum_kernel (int* scanned_array, int* incr) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int scalar = incr[blockIdx.x];
    scanned_array[index] += scalar;
}

void scalar_vector_sum (int* scanned_array, int* incr, int length){
    int pow_length = nextPow2(length);
    int chunk_size = TPB;
    int num_block;
    int threads_per_block = TPB;
    
    num_block = pow_length/chunk_size;

    printf("num blocks in sum kernel: %d\n", num_block);

    scalar_vector_sum_kernel<<<num_block,TPB,TPB*sizeof(int)>>>(scanned_array, incr);
    return;
}

__global__ void exclusive_scan_kernel (int length, int* in_array, int* next_chunk_sum) {

    int base_index = blockIdx.x * 2 * blockDim.x;
   
    int threadIndex = threadIdx.x;

    extern __shared__ int temp[];

    // //Load input into the shared memory
    temp[2*threadIndex] = in_array[base_index + (2*threadIndex)];
    temp[2*threadIndex+1] = in_array[base_index + (2*threadIndex)+1];

    int offset = 1, active = 2;
    
    //Up-sweep
    for(int d = blockDim.x; d>0; d=d/2) {
        __syncthreads();             
        if(threadIdx.x < d){
            int current = offset*(2*threadIdx.x + 1) -1;
            int next = offset*(2*threadIdx.x + 2) -1;
            temp[next] += temp[current];
        }
        offset*=2;
    }

   
    __syncthreads();

    
    if(threadIndex == 0) {
        if(next_chunk_sum){
            next_chunk_sum[blockIdx.x] = temp[length-1];
        }
        temp[length-1] = 0;
    }
    
    
    //Down-sweep
    
    for(int d = 1; d<length; d*=2) {
         offset /=2;
	 __syncthreads();

         if(threadIndex <d) {
             int cur = offset*(2*threadIdx.x + 1) -1;
             int next = offset*(2*threadIdx.x + 2) -1;
             int t = temp[cur];
             temp[cur] = temp[next];
             temp[next] += t;
         }

     }
     
    __syncthreads();
    in_array[base_index + (2*threadIndex)] = temp[2*threadIndex] ;
    in_array[base_index + (2*threadIndex) + 1] = temp[2*threadIndex+1] ;
    
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
    int chunk_size;
    if(pow_length < 2048){
        chunk_size = pow_length;
    }
    else{
        chunk_size = TPB;
    }
    
    int threads_per_block = chunk_size/2;
    
    int num_block = pow_length/chunk_size;

    printf("num blocks: %d\n", num_block);

    exclusive_scan_kernel<<<num_block, threads_per_block, chunk_size*sizeof(int)>>>(chunk_size, device_data, device_next_chunk_sum);

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
    //Debug GPU pointer
    int* device_debug_inter_sum_array;
    
    // We round the array size up to a power of 2, but elements after
    // the end of the original input are left uninitialized and not checked
    // for correctness.
    // You may have an easier time in your implementation if you assume the
    // array's length is a power of 2, but this will result in extra work on
    // non-power-of-2 inputs.
    int rounded_length = nextPow2(end - inarray);
    printf("rounded length:%d\n",rounded_length);

    //Allocate GPU memory for input array
    cudaCheckError(
    cudaMalloc((void **)&device_data, sizeof(int) * rounded_length)
    );
    //Allocate GPU memory for holding upsweep reduced sums/scanned sums
    cudaCheckError(
    cudaMalloc((void **)&device_inter_sum_array, sizeof(int) * rounded_length)
    );
    //Allocate GPU memory for debug array
    cudaCheckError(
    cudaMalloc((void **)&device_debug_inter_sum_array, sizeof(int) * rounded_length)
    );
    //Copy input array and next chunk temp array to GPU
    cudaCheckError(
    cudaMemcpy(device_data, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice)
    );
    //Initialise upsweep sum array and scanned sum array
    cudaCheckError(
    cudaMemset(device_inter_sum_array,0,rounded_length*sizeof(int))
    );
    double startTime = CycleTimer::currentSeconds();

    printf("Computing exclusive scan of input array\n");

    //Perform exclusive scan on input array
    exclusive_scan(device_data, end - inarray, device_inter_sum_array);

    printf("Completed computation of exclusive scan of input array\n");

    // Wait for any work left over to be completed.
    cudaCheckError(
    cudaThreadSynchronize()
    );

    #ifdef DEBUG
        printf("/*DEBUG - INPUT ARRAY*/ \n");
        for(int idx = 0; idx < rounded_length; idx++){
            printf("inarray[%d]=%d\n",idx,inarray[idx]);
        }
        int* inter_sum_array;
        inter_sum_array = new int[rounded_length];
        cudaCheckError(
        cudaMemcpy(inter_sum_array, device_inter_sum_array, rounded_length * sizeof(int),
                cudaMemcpyDeviceToHost)
        );
        printf("/*DEBUG - UPSWEEP SUM ARRAY*/ \n");
        for(int idx = 0; idx < rounded_length; idx++){
            printf("inter_sum_array[%d]=%d\n",idx,inter_sum_array[idx]);
        }
    #endif

    
    printf("Computing exclusive scan of intermediate sum array\n");

    //Perform exclusive scan on upsweep sum array
    //exclusive_scan(device_inter_sum_array, TPB, nullptr);
    if(rounded_length >= 2048){
        exclusive_scan_kernel<<<1, (rounded_length/2048), (rounded_length/1024)*sizeof(int)>>>((rounded_length/TPB), device_inter_sum_array, nullptr);
    }  
    else{
        exclusive_scan_kernel<<<1, (rounded_length/2), (rounded_length)*sizeof(int)>>>(rounded_length, device_inter_sum_array, nullptr);
    }

    printf("Completed computation of exclusive scan of intermediate sum array\n");

    // Wait for any work left over to be completed.
    cudaCheckError(cudaThreadSynchronize());

     

    //Transfer input and next chunk sum array from GPU to CPU
    cudaCheckError(
    cudaMemcpy(resultarray, device_data, (end - inarray) * sizeof(int),
               cudaMemcpyDeviceToHost)
    );


    
    #ifdef DEBUG
        cudaCheckError(
        cudaMemcpy(inter_sum_array, device_inter_sum_array, rounded_length * sizeof(int),
                cudaMemcpyDeviceToHost)
        );

        printf("DEBUG - RESULT ARRAY \n");
        for(int idx = 0; idx < rounded_length; idx++){
            printf("resultarray[%d]=%d\n",idx,resultarray[idx]);
        }
        
        printf("DEBUG - UPSWEEP SCANNED SUM ARRAY\n");
        for(int idx = 0; idx < rounded_length; idx++){
            printf("inter_sweep_sum_array[%d]=%d\n",idx,inter_sum_array[idx]);
        }

        delete[] inter_sum_array;
    #endif
    
    
    printf("Launching scalar vector sum kernel\n");

    //Perform vector sum of scanned input and scanned sum array
    scalar_vector_sum(device_data, device_inter_sum_array, rounded_length);

    printf("Finished computing scalar vector sum kernel\n");

    // Wait for any work left over to be completed.
    cudaCheckError(cudaThreadSynchronize());

    cudaCheckError(
    cudaMemcpy(resultarray, device_data, (end - inarray) * sizeof(int),
               cudaMemcpyDeviceToHost)
    );
    
    #ifdef DEBUG
        printf("/*DEBUG - RESULT ARRAY*/ \n");
        for(int idx = 0; idx < rounded_length; idx++){
            printf("resultarray[%d]=%d\n",idx,resultarray[idx]);
        }
    #endif
    
    double endTime = CycleTimer::currentSeconds();
    double overallDuration = endTime - startTime;

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

__global__ void device_find_peaks (int* in_array, int* out_array, int length) {
    out_array[0] = 0;
    out_array[length-1] = 0;
        for(int i = 1; i< length; i++){
        if(in_array[i] > in_array[i-1] && in_array[i] > in_array[i+1]){
            out_array[i] = 1;
        }
        else{
            out_array[i] = 0;
        }
    }
    return;
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
    // if(num_threads > length) {
    //     num_threads = length;
    // }
    // else{
    //     num_threads = length / maxthreads;
    // }
    // int num_threads = length / maxthreads;
    int num_threads = 1, num_blocks = 1;
    int rounded_length = nextPow2(length);
    int* device_peak_mask_output;
    //Allocate auxiliary array to hold peak indices
    cudaMalloc((void **)&device_peak_mask_output, rounded_length * sizeof(int));
    cudaCheckError(
        cudaMemset(device_peak_mask_output,0,rounded_length*sizeof(int))
    );

    //CUDA kernel launch to find and mask peak indices
    device_find_peaks<<<num_blocks, num_threads>>>(device_input, device_peak_mask_output, length);
    
    cudaCheckError(cudaThreadSynchronize());

    int* peak_mask_output, *peak_mask_scanned, *peak_indices;
    peak_mask_output = new int[rounded_length];
    peak_mask_scanned = new int[rounded_length];
    peak_indices = new int[rounded_length];
    cudaCheckError(
        cudaMemcpy(peak_mask_output, device_peak_mask_output, rounded_length * sizeof(int),
            cudaMemcpyDeviceToHost)
    );
    #ifdef FIND_PEAK_DEBUG
        printf("/*DEBUG - device_peak_mask_output ARRAY*/ \n");
        for(int idx = 0; idx < rounded_length; idx++){
            printf("peak_mask_output[%d]=%d\n",idx,peak_mask_output[idx]);
        }
    #endif
    
    //Exclusive scan of peak indices
    cudaScan(peak_mask_output, peak_mask_output+length, peak_mask_scanned);

    cudaCheckError(cudaThreadSynchronize());
    
    #ifdef FIND_PEAK_DEBUG
        printf("/*DEBUG - peak_mask_scanned ARRAY*/ \n");
        for(int idx = 0; idx < length; idx++){
            printf("peak_mask_scanned[%d]=%d\n",idx,peak_mask_scanned[idx]);
        }
    #endif
    delete[] peak_mask_output;
    delete[] peak_mask_scanned;
    delete[] peak_indices;
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
