#pragma once
#include <vector>
#include <string>
#include <cstdint>
#include <cublas_v2.h>
#include <cuda_runtime.h>
using namespace std;

using ll=uint64_t;

class SparseBlockMatrix {
public:
    int height, width, k;

    // Position of each block (row_start, col_start)
    std::vector<std::pair<int, int>> block_positions;

    // Values of blocks: block_values[i] is a k x k matrix
    std::vector<std::vector<std::vector<ll>>> block_values;

    SparseBlockMatrix() = default;
    SparseBlockMatrix(int h, int w, int bk);

    void addBlock(int row_start, int col_start, const std::vector<std::vector<ll>>& values);
    static SparseBlockMatrix fromFile(const std::string& filepath, int k);
    
    SparseBlockMatrix* multiplyWith(const SparseBlockMatrix& other) const;
    
    void writeToFile(const std::string& filepath) const;
    void print() const;
    double getDensity() const;
    void printStats() const;

    static cublasHandle_t cublas_handle;
    static bool initialized;
    static void init_cuda();
    static void shutdown_cuda();

};

vector<ll> flatten_matrix(const vector<vector<ll>> &mat, int m);
vector<vector<ll>> unflatten_matrix(const vector<ll> &flat, int m);
bool isEqual(const SparseBlockMatrix& A, const SparseBlockMatrix& B);
SparseBlockMatrix* multiply_sequence(const std::vector<SparseBlockMatrix>& mats);
// struct BlockData;
// __global__ void multiplyBlocksKernel(
//     const BlockData* blocks1, int numBlocks1,
//     const BlockData* blocks2, int numBlocks2,
//     BlockData* resultBlocks, int* resultCount,
//     int height, int width, int k)