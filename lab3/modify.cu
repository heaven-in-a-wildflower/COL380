#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <algorithm>

// CUDA Error checking macro
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define START_TIMER(start) CHECK_CUDA(cudaEventRecord(start));
#define STOP_TIMER(start, stop, elapsed) \
    CHECK_CUDA(cudaEventRecord(stop)); \
    CHECK_CUDA(cudaEventSynchronize(stop)); \
    CHECK_CUDA(cudaEventElapsedTime(&elapsed, start, stop));

using namespace std;

// Maximum number of matrices we can process in parallel
#define MAX_CONCURRENT_MATRICES 1024

// Persistent CUDA resources with multiple streams
struct CudaResources {
    cudaStream_t streams[MAX_CONCURRENT_MATRICES] = {nullptr};
    int* d_freq[MAX_CONCURRENT_MATRICES] = {nullptr};
    int* d_blockSums[MAX_CONCURRENT_MATRICES] = {nullptr};
    int* d_prefix[MAX_CONCURRENT_MATRICES] = {nullptr};
    int* d_input[MAX_CONCURRENT_MATRICES] = {nullptr};   // 2D input allocated with pitch
    int* d_output[MAX_CONCURRENT_MATRICES] = {nullptr};  // 2D output allocated with pitch
    size_t inputPitch[MAX_CONCURRENT_MATRICES] = {0};
    size_t outputPitch[MAX_CONCURRENT_MATRICES] = {0};
    size_t currentAllocRows = 0;
    size_t currentAllocCols = 0;
    int maxRange = 0;
    int numAllocated = 0;
    bool initialized = false;
};

static CudaResources g_resources;

// Calculate the memory needed for processing given matrices
size_t calculateRequiredMemory(int maxRows, int maxCols, int maxRange, int numMatrices) {
    size_t memoryNeeded = 0;
    
    for (int i = 0; i < numMatrices; i++) {
        // Frequency array: (maxRange + 1) * sizeof(int)
        memoryNeeded += (maxRange + 1) * sizeof(int);
        
        // Prefix array: (maxRange + 1) * sizeof(int)
        memoryNeeded += (maxRange + 1) * sizeof(int);
        
        // Block sums: For each block in the scan
        int blockSize = 256;
        int blockCount = (maxRange + 1 + blockSize - 1) / blockSize;
        memoryNeeded += blockCount * sizeof(int);
        
        // Input and output matrices (estimate with padding)
        size_t matrixSize = maxRows * maxCols * sizeof(int);
        memoryNeeded += 2 * matrixSize * 1.2; // Factor 1.2 for padding in pitched memory
    }
    
    return memoryNeeded;
}

// Get optimal number of concurrent matrices based on available GPU memory
int getOptimalConcurrentMatrices(int maxRows, int maxCols, int maxRange, int requestedMatrices) {
    // Get device properties
    cudaDeviceProp deviceProp;
    CHECK_CUDA(cudaGetDeviceProperties(&deviceProp, 0));
    
    // Total available memory (with 10% margin for safety)
    size_t availableMemory = deviceProp.totalGlobalMem * 0.9;
    
    // Calculate memory required per matrix
    size_t memoryPerMatrix = calculateRequiredMemory(maxRows, maxCols, maxRange, 1);
    
    // Calculate optimal number of concurrent matrices
    int optimalMatrices = availableMemory / memoryPerMatrix;
    
    // Limit between 1 and requested number (or MAX_CONCURRENT_MATRICES)
    optimalMatrices = max(1, min(optimalMatrices, min(requestedMatrices, MAX_CONCURRENT_MATRICES)));
    
    cout << "Memory per matrix: " << memoryPerMatrix / (1024 * 1024) << " MB" << endl;
    cout << "Available memory (90%): " << availableMemory / (1024 * 1024) << " MB" << endl;
    cout << "Optimal concurrent matrices: " << optimalMatrices << endl;
    
    return optimalMatrices;
}


// Free CUDA resources
void freeCudaResources(int numMatrices) {
    numMatrices = min(numMatrices, MAX_CONCURRENT_MATRICES);
    
    for (int i = 0; i < numMatrices; i++) {
        if (g_resources.streams[i]) {
            CHECK_CUDA(cudaStreamSynchronize(g_resources.streams[i]));
            CHECK_CUDA(cudaStreamDestroy(g_resources.streams[i]));
            g_resources.streams[i] = nullptr;
        }
        
        if (g_resources.d_freq[i]) {
            CHECK_CUDA(cudaFree(g_resources.d_freq[i]));
            g_resources.d_freq[i] = nullptr;
        }
        
        if (g_resources.d_prefix[i]) {
            CHECK_CUDA(cudaFree(g_resources.d_prefix[i]));
            g_resources.d_prefix[i] = nullptr;
        }
        
        if (g_resources.d_blockSums[i]) {
            CHECK_CUDA(cudaFree(g_resources.d_blockSums[i]));
            g_resources.d_blockSums[i] = nullptr;
        }
        
        if (g_resources.d_input[i]) {
            CHECK_CUDA(cudaFree(g_resources.d_input[i]));
            g_resources.d_input[i] = nullptr;
        }
        
        if (g_resources.d_output[i]) {
            CHECK_CUDA(cudaFree(g_resources.d_output[i]));
            g_resources.d_output[i] = nullptr;
        }
    }
    
    g_resources.initialized = false;
    g_resources.numAllocated = 0;
}

// Initialize CUDA resources using pitched memory for 2D matrices
void initCudaResources(int maxRows, int maxCols, int maxRange, int numMatrices) {
    // Check if we need to reallocate resources (if dimensions or range have changed)
    bool needReallocation = !g_resources.initialized || 
                           maxRows > g_resources.currentAllocRows || 
                           maxCols > g_resources.currentAllocCols ||
                           maxRange > g_resources.maxRange ||
                           numMatrices > g_resources.numAllocated;
    
    if (needReallocation) {
        // If already initialized, free existing resources first
        if (g_resources.initialized) {
            freeCudaResources(g_resources.numAllocated);
        }
        
        numMatrices = min(numMatrices, MAX_CONCURRENT_MATRICES);
        
        // Determine optimal number of concurrent matrices based on memory
        numMatrices = getOptimalConcurrentMatrices(maxRows, maxCols, maxRange, numMatrices);
        
        for (int i = 0; i < numMatrices; i++) {
            // Create stream for each matrix
            CHECK_CUDA(cudaStreamCreate(&g_resources.streams[i]));
            
            // Allocate memory for each matrix
            CHECK_CUDA(cudaMalloc(&g_resources.d_freq[i], (maxRange + 1) * sizeof(int)));
            CHECK_CUDA(cudaMalloc(&g_resources.d_prefix[i], (maxRange + 1) * sizeof(int)));
            
            int blockSize = 256;
            int blockCount = (maxRange + 1 + blockSize - 1) / blockSize;
            CHECK_CUDA(cudaMalloc(&g_resources.d_blockSums[i], blockCount * sizeof(int)));
            
            // Allocate pitched memory for input and output matrices
            CHECK_CUDA(cudaMallocPitch(&g_resources.d_input[i], &g_resources.inputPitch[i], 
                            maxCols * sizeof(int), maxRows));
            CHECK_CUDA(cudaMallocPitch(&g_resources.d_output[i], &g_resources.outputPitch[i], 
                            maxCols * sizeof(int), maxRows));
            
            // Warm-up: initialize frequency array
            CHECK_CUDA(cudaMemsetAsync(g_resources.d_freq[i], 0, (maxRange + 1) * sizeof(int), g_resources.streams[i]));
        }
        
        // Synchronize all streams before proceeding
        for (int i = 0; i < numMatrices; i++) {
            CHECK_CUDA(cudaStreamSynchronize(g_resources.streams[i]));
        }
        
        g_resources.currentAllocRows = maxRows;
        g_resources.currentAllocCols = maxCols;
        g_resources.maxRange = maxRange;
        g_resources.numAllocated = numMatrices;
        g_resources.initialized = true;
        
        cout << "Allocated resources for " << numMatrices << " concurrent matrices" << endl;
    } else {
        cout << "Reusing existing resources for " << g_resources.numAllocated << " concurrent matrices" << endl;
    }
}

// Kernel to count frequencies using pitched 2D access
__global__ void countFrequencyKernel2D(const int* __restrict__ input, size_t pitch, 
                                        int rows, int cols, int* __restrict__ freq, int range) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && col < cols) {
        // Compute pointer to the row using pitch (in bytes)
        const int* rowPtr = (const int*)((const char*)input + row * pitch);
        int val = rowPtr[col];
        if (val >= 1 && val <= range) {
            atomicAdd(&freq[val], 1);
        }
    }
}

// Phase 1: Local scan within blocks
__global__ void localScanKernel(int* d_freq, int* d_blockSums, int size) {
    __shared__ int temp[1024];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    temp[tid] = (gid < size) ? d_freq[gid] : 0;
    __syncthreads();

    for (int offset = 1; offset < blockDim.x; offset *= 2) {
        int val = (tid >= offset) ? temp[tid - offset] : 0;
        __syncthreads();
        if (tid >= offset) temp[tid] += val;
        __syncthreads();
    }

    if (gid < size) d_freq[gid] = temp[tid];
    if (tid == blockDim.x - 1) d_blockSums[blockIdx.x] = temp[tid];
}

// Phase 2: Propagate block sums
__global__ void addBlockSumsKernel(int* d_freq, int* d_blockSums, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size && blockIdx.x > 0) {
        d_freq[tid] += d_blockSums[blockIdx.x - 1];
    }
}

// Generate modified matrix directly into a pitched 2D layout
__global__ void generateModMatrix2D(const int* __restrict__ d_prefix, int* __restrict__ d_modMatrix, 
    size_t pitch, int rows, int cols, int range) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        // Use 64-bit index to avoid overflow
        long long flatIdx = (long long)row * cols + col;
        int left = 1, right = range;

        while (left <= right) {
            int mid = (left + right) / 2;
            long long startIdx = (mid == 1) ? 0 : (long long)d_prefix[mid - 1];
            long long endIdx = (long long)d_prefix[mid];

            if (flatIdx >= startIdx && flatIdx < endIdx) {
                // Compute pointer for output row using pitch (in bytes)
                int* rowPtr = (int*)((char*)d_modMatrix + row * pitch);
                rowPtr[col] = mid;
                break;
            } else if (flatIdx < startIdx) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
    }
}

// Process a single matrix with the given stream index
void processMatrix(int streamIdx, vector<vector<int>>& matrix, int range, vector<vector<int>>& modMatrix) {
    int rows = matrix.size();
    int cols = matrix[0].size();
    int freqSize = range + 1;
    
    // Copy the matrix to GPU using cudaMemcpy2D
    for (int i = 0; i < rows; i++) {
        CHECK_CUDA(cudaMemcpy2DAsync((char*)g_resources.d_input[streamIdx] + i * g_resources.inputPitch[streamIdx], 
                          g_resources.inputPitch[streamIdx],
                          matrix[i].data(), cols * sizeof(int),
                          cols * sizeof(int), 1, 
                          cudaMemcpyHostToDevice, g_resources.streams[streamIdx]));
    }
    
    // Initialize frequency counter
    CHECK_CUDA(cudaMemsetAsync(g_resources.d_freq[streamIdx], 0, freqSize * sizeof(int), g_resources.streams[streamIdx]));

    // Count frequencies using 2D kernel
    dim3 blockDim(16, 16);
    dim3 gridDim((cols + blockDim.x - 1) / blockDim.x,
                 (rows + blockDim.y - 1) / blockDim.y);
    countFrequencyKernel2D<<<gridDim, blockDim, 0, g_resources.streams[streamIdx]>>>(
        g_resources.d_input[streamIdx], g_resources.inputPitch[streamIdx], 
        rows, cols, g_resources.d_freq[streamIdx], range);
    CHECK_CUDA(cudaGetLastError());
    
    // Local scan on frequency array
    int blockSize = 256;
    int blockCount = (freqSize + blockSize - 1) / blockSize;
    localScanKernel<<<blockCount, blockSize, 0, g_resources.streams[streamIdx]>>>(
        g_resources.d_freq[streamIdx], g_resources.d_blockSums[streamIdx], freqSize);
    CHECK_CUDA(cudaGetLastError());
    
    // Process block sums
    vector<int> h_blockSums(blockCount);
    CHECK_CUDA(cudaMemcpyAsync(h_blockSums.data(), g_resources.d_blockSums[streamIdx],
                    blockCount * sizeof(int), cudaMemcpyDeviceToHost, g_resources.streams[streamIdx]));
    
    // Need to synchronize here to make sure h_blockSums is ready for processing
    CHECK_CUDA(cudaStreamSynchronize(g_resources.streams[streamIdx]));
    
    for (int i = 1; i < blockCount; i++) {
        h_blockSums[i] += h_blockSums[i - 1];
    }
    
    CHECK_CUDA(cudaMemcpyAsync(g_resources.d_blockSums[streamIdx], h_blockSums.data(),
                    blockCount * sizeof(int), cudaMemcpyHostToDevice, g_resources.streams[streamIdx]));
    
    addBlockSumsKernel<<<blockCount, blockSize, 0, g_resources.streams[streamIdx]>>>(
        g_resources.d_freq[streamIdx], g_resources.d_blockSums[streamIdx], freqSize);
    CHECK_CUDA(cudaGetLastError());
    
    // Generate modified matrix using 2D kernel
    dim3 outBlockDim(16, 16);
    dim3 outGridDim((cols + outBlockDim.x - 1) / outBlockDim.x,
                   (rows + outBlockDim.y - 1) / outBlockDim.y);
    generateModMatrix2D<<<outGridDim, outBlockDim, 0, g_resources.streams[streamIdx]>>>(
        g_resources.d_freq[streamIdx], g_resources.d_output[streamIdx], 
        g_resources.outputPitch[streamIdx], rows, cols, range);
    CHECK_CUDA(cudaGetLastError());
    
    // Copy the output back using cudaMemcpy2D
    modMatrix.resize(rows, vector<int>(cols));
    for (int i = 0; i < rows; i++) {
        CHECK_CUDA(cudaMemcpy2DAsync(modMatrix[i].data(), cols * sizeof(int),
                         (char*)g_resources.d_output[streamIdx] + i * g_resources.outputPitch[streamIdx], 
                         g_resources.outputPitch[streamIdx],
                         cols * sizeof(int), 1, 
                         cudaMemcpyDeviceToHost, g_resources.streams[streamIdx]));
    }
}

vector<vector<vector<int>>> modify(vector<vector<vector<int>>>& matrices, vector<int>& ranges) {
    int numMatrices = matrices.size();
    vector<vector<vector<int>>> modifiedMatrices(numMatrices);
    
    cudaEvent_t start, stop, epic_start, epic_stop;
    float elapsed = 0;
    float epic_elapsed = 0;

    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventCreate(&epic_start));
    CHECK_CUDA(cudaEventCreate(&epic_stop));

    START_TIMER(epic_start);
    START_TIMER(start);

    // Determine maximum dimensions for allocation
    int maxRows = 0, maxCols = 0, maxRange = 0;
    for (int m = 0; m < numMatrices; m++) {
        maxRows = max(maxRows, (int)matrices[m].size());
        maxCols = max(maxCols, (int)matrices[m][0].size());
        maxRange = max(maxRange, ranges[m]);
    }

    // Calculate optimal number of concurrent matrices
    int concurrentMatrices = getOptimalConcurrentMatrices(maxRows, maxCols, maxRange, numMatrices);
    
    // Initialize persistent resources (will reuse if already allocated)
    initCudaResources(maxRows, maxCols, maxRange, concurrentMatrices);

    // Use the number of matrices that were actually allocated
    concurrentMatrices = g_resources.numAllocated;

    STOP_TIMER(start, stop, elapsed);
    cout << "Resource Init time= " << elapsed << "ms" << endl;

    START_TIMER(start);
    
    // Process matrices in batches
    for (int batchStart = 0; batchStart < numMatrices; batchStart += concurrentMatrices) {
        int batchSize = min(concurrentMatrices, numMatrices - batchStart);
        
        // Launch processing for each matrix in the batch
        for (int i = 0; i < batchSize; i++) {
            int matrixIdx = batchStart + i;
            processMatrix(i, matrices[matrixIdx], ranges[matrixIdx], modifiedMatrices[matrixIdx]);
        }
        
        // Synchronize all streams for this batch before moving to next batch
        for (int i = 0; i < batchSize; i++) {
            CHECK_CUDA(cudaStreamSynchronize(g_resources.streams[i]));
        }
    }
    
    STOP_TIMER(start, stop, elapsed);
    cout << "Matrix processing time= " << elapsed << "ms" << endl;
    
    STOP_TIMER(epic_start, epic_stop, epic_elapsed);
    cout << "Epic time= " << epic_elapsed << "ms" << endl;

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaEventDestroy(epic_start));
    CHECK_CUDA(cudaEventDestroy(epic_stop));
    
    // Note: We're no longer freeing resources here to allow reuse
    // Resources will persist until program exit or explicit freeCudaResources() call
    
    return modifiedMatrices;
}