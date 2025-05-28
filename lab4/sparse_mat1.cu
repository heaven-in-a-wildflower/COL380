#include "sparse_mat.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <cstdint>
#include <vector>
#include <map>
#include <utility>
#include <cstdlib>
#include <ctime>
#include <random>
#include <algorithm>
#include <omp.h>
#include <chrono>
#include "sparse_mat.h"
#include <unordered_map>
#include <unordered_set>
#include <tuple>
#include <cassert>
#include <cuda_runtime.h>
#include <iomanip>
#include <cublas_v2.h>

using namespace std;
using namespace std::chrono;

using ll=uint64_t;

#define CUDA_CHECK(err) \
  if ((err) != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
#define CUBLAS_CHECK(err) \
  if ((err) != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS error");

SparseBlockMatrix::SparseBlockMatrix(int h, int w, int bk) : height(h), width(w), k(bk) {}

void SparseBlockMatrix::addBlock(int row_start, int col_start, const std::vector<std::vector<ll>>& values) {
    // Check if dimensions are correct
    if (values.size() != k || values[0].size() != k) {
        throw std::runtime_error("Block dimensions do not match k");
    }
    
    // // Check if block position is valid
    // if (row_start < 0 || row_start + k > height || col_start < 0 || col_start + k > width) {
    //     throw std::runtime_error("Block position out of bounds");
    // }
    
    // Check if the block is all zeros
    bool all_zeros = true;
    for (int i = 0; i < k && all_zeros; i++) {
        for (int j = 0; j < k && all_zeros; j++) {
            if (values[i][j] != 0) {
                all_zeros = false;
            }
        }
    }
    
    // Don't add if all zeros
    if (all_zeros) return;
    
    // Add the block
    block_positions.push_back({row_start, col_start});
    block_values.push_back(values);
}

// Get the density (percentage of non-zero blocks)
double SparseBlockMatrix::getDensity() const {
    int total_possible_blocks = ((height + k - 1) / k) * ((width + k - 1) / k);
    return 100.0 * block_positions.size() / total_possible_blocks;
}

// Print matrix statistics
void SparseBlockMatrix::printStats() const {
    std::cout << "Matrix dimensions: " << height << "x" << width << " (k=" << k << ")" << std::endl;
    std::cout << "Number of blocks: " << block_positions.size() << std::endl;
    std::cout << "Matrix density: " << std::fixed << std::setprecision(2) << getDensity() << "%" << std::endl;
}

cublasHandle_t SparseBlockMatrix::cublas_handle;
bool SparseBlockMatrix::initialized = false;

void SparseBlockMatrix::init_cuda() {
    if (!initialized) {
        CUDA_CHECK(cudaSetDevice(0)); // optional if multiple GPUs
        CUBLAS_CHECK(cublasCreate(&cublas_handle));
        initialized = true;
    }
}

void SparseBlockMatrix::shutdown_cuda() {
    if (initialized) {
        CUBLAS_CHECK(cublasDestroy(cublas_handle));
        initialized = false;
    }
}

SparseBlockMatrix* SparseBlockMatrix::multiplyWith(const SparseBlockMatrix& B) const {
    
    if (!initialized) throw std::runtime_error("CUDA not initialized. Call init_cuda() first.");
    if (width  != B.height) throw std::runtime_error("Dimension mismatch");
    if (k != B.k)          throw std::runtime_error("Block‐size mismatch");

    int M = height, N = B.width, K = width;
    // 1) Build host‐dense A[M][K], B[K][N]
    std::vector<double> hA(M * K, 0.0), hB(K * N, 0.0);
    // scatter A’s blocks
    for (size_t bi = 0; bi < block_positions.size(); ++bi) {
        int br = block_positions[bi].first;
        int bc = block_positions[bi].second;
        const auto& blk = block_values[bi];
        for (int i = 0; i < k; ++i)
            for (int j = 0; j < k; ++j)
                hA[(br + i) * K + (bc + j)] = double(blk[i][j]);
    }
    // scatter B’s blocks
    for (size_t bi = 0; bi < B.block_positions.size(); ++bi) {
        int br = B.block_positions[bi].first;
        int bc = B.block_positions[bi].second;
        const auto& blk = B.block_values[bi];
        for (int i = 0; i < k; ++i)
            for (int j = 0; j < k; ++j)
                hB[(br + i) * N + (bc + j)] = double(blk[i][j]);
    }

    // 2) Allocate device buffers
    double *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, sizeof(double)*M*K));
    CUDA_CHECK(cudaMalloc(&dB, sizeof(double)*K*N));
    CUDA_CHECK(cudaMalloc(&dC, sizeof(double)*M*N));

    // 3) Copy host → device
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), sizeof(double)*M*K, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), sizeof(double)*K*N, cudaMemcpyHostToDevice));
    // zero‐out C
    CUDA_CHECK(cudaMemset(dC, 0, sizeof(double)*M*N));

    // // 4) Launch cuBLAS Dgemm: C = A·B
    // cublasHandle_t handle;
    // CUBLAS_CHECK(cublasCreate(&handle));
    const double alpha = 1.0, beta = 0.0;
    // Note: column‐major vs row‐major. We'll pretend row‐major by swapping A/B and transposing:
    CUBLAS_CHECK(cublasDgemm(
        cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        dB, N,        // B: (N×K)ᵀ stored as (K×N) so we pass leading dim=N
        dA, K,        // A: (K×M)ᵀ so leading dim=K
        &beta,
        dC, N));      // C: (N×M)ᵀ so leading dim=N

    // 5) Copy result back
    std::vector<double> hC(M * N);
    CUDA_CHECK(cudaMemcpy(hC.data(), dC, sizeof(double)*M*N, cudaMemcpyDeviceToHost));

    // 6) Re‐sparsify in k×k grid
    auto* C = new SparseBlockMatrix(M, N, k);
    for (int br = 0; br < M; br += k) {
      for (int bc = 0; bc < N; bc += k) {
        // build one k×k block
        std::vector<std::vector<ll>> blk(k, std::vector<ll>(k));
        bool nonzero = false;
        for (int i = 0; i < k; ++i) {
          for (int j = 0; j < k; ++j) {
            ll v = ll(std::llround(hC[(br+i)*N + (bc+j)]));
            blk[i][j] = v;
            if (v != 0) nonzero = true;
          }
        }
        if (nonzero) {
          C->block_positions.emplace_back(br, bc);
          C->block_values.push_back(std::move(blk));
        }
      }
    }

    // cleanup
    // cublasDestroy(handle);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return C;
}

void SparseBlockMatrix::print() const {
    std::cout << height << " " << width << "\n";
    std::cout << block_positions.size() << "\n";

    for (size_t i = 0; i < block_positions.size(); ++i) {
        const auto& pos = block_positions[i];
        std::cout << pos.first << " " << pos.second << "\n";

        const auto& block = block_values[i];
        for (const auto& row_vals : block) {
            for (size_t j = 0; j < row_vals.size(); ++j) {
                std::cout << row_vals[j];
                if (j < row_vals.size() - 1) std::cout << " ";
            }
            std::cout << "\n";
        }
    }
}

void SparseBlockMatrix::writeToFile(const std::string& filepath) const {
    std::ofstream out_file(filepath);
    if (!out_file.is_open()) {
        std::cerr << "Error opening file for writing: " << filepath << "\n";
        return;
    }

    out_file << height << " " << width << "\n";
    out_file << block_positions.size() << "\n";

    for (size_t i = 0; i < block_positions.size(); ++i) {
        const auto& pos = block_positions[i];
        out_file << pos.first << " " << pos.second << "\n";

        const auto& block = block_values[i];
        for (const auto& row_vals : block) {
            for (size_t j = 0; j < row_vals.size(); ++j) {
                out_file << row_vals[j];
                if (j < row_vals.size() - 1) out_file << " ";
            }
            out_file << "\n";
        }
    }
    out_file.close();
}

SparseBlockMatrix SparseBlockMatrix::fromFile(const std::string& filepath, int k) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filepath);
    }

    int h, w, num_blocks;
    file >> h >> w >> num_blocks;

    SparseBlockMatrix matrix(h, w, k);

    // Read block positions
    for (int i = 0; i < num_blocks; ++i) {
        int r, c;
        file >> r >> c;
        matrix.block_positions.emplace_back(r, c);
    
        std::vector<std::vector<ll>> block(k, std::vector<ll>(k));
        for (int r = 0; r < k; ++r) {
            for (int c = 0; c < k; ++c) {
                file >> block[r][c];
            }
        }
        matrix.block_values.push_back(block);
    }

    return matrix;
}

template <typename T>
void print_vector(const vector<T> &vec)
{
    for (const auto &v : vec)
        cout << v << " ";
    cout << endl;
}

void print_matrix(const vector<vector<int>> &matrix)
{
    for (const auto &row : matrix)
    {
        for (const auto &elem : row)
            cout << elem << " ";
        cout << endl;
    }
}

// -------------------------------------------------------------------
// Main multiplication driver.

double total_push_time = 0.0;
double block_mult_time = 0.0;
double non_zero_check_time = 0.0;


constexpr int CACHE_LINE_SIZE = 64;
constexpr int INT_SIZE = 8;
constexpr int INTS_PER_CACHE_LINE = CACHE_LINE_SIZE / INT_SIZE;

// Convert 2D matrix to flattened 1D array
vector<ll> flatten_matrix(const vector<vector<ll>> &mat, int m) 
{
    vector<ll> flat(m * m);
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < m; j++)
        {
            flat[i * m + j] = mat[i][j];
        }
    }
    return flat;
};

// Convert flattened 1D array back to 2D matrix
vector<vector<ll>> unflatten_matrix(const vector<ll> &flat, int m) 
{
    vector<vector<ll>> mat(m, vector<ll>(m));
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < m; j++)
        {
            mat[i][j] = flat[i * m + j];
        }
    }
    return mat;
};


// SparseBlockMatrix* multiply_sequence(const std::vector<SparseBlockMatrix>& mats) {
//     if (mats.empty()) throw std::invalid_argument("Empty matrix list for multiplication");
    
//     // Create a copy of the first matrix
//     SparseBlockMatrix* result = new SparseBlockMatrix(mats[0]);
    
//     for (size_t i = 1; i < mats.size(); ++i) {
//         // Store the current result pointer
//         SparseBlockMatrix* temp = result;
        
//         // Multiply with the next matrix, which gives us a new pointer
//         result = result->multiplyWith(mats[i]);
        
//         // Delete the previous result to avoid memory leaks
//         delete temp;
        
//         // Synchronize CUDA operations
//         cudaDeviceSynchronize();
//     }
    
//     return result;
// }

// #include <omp.h>

SparseBlockMatrix* multiply_sequence(const std::vector<SparseBlockMatrix>& mats) {
    if (mats.empty()) throw std::invalid_argument("Empty matrix list for multiplication");
    
    if (mats.size() == 1) {
        return new SparseBlockMatrix(mats[0]);
    }

    size_t mid = mats.size() / 2;
    std::vector<SparseBlockMatrix> left(mats.begin(), mats.begin() + mid);
    std::vector<SparseBlockMatrix> right(mats.begin() + mid, mats.end());

    SparseBlockMatrix *left_result = nullptr, *right_result = nullptr;

#pragma omp parallel
#pragma omp single nowait
    {
#pragma omp task shared(left_result)
        {
            left_result = multiply_sequence(left);
        }
#pragma omp task shared(right_result)
        {
            right_result = multiply_sequence(right);
        }
#pragma omp taskwait
    }

    SparseBlockMatrix* result = left_result->multiplyWith(*right_result);

    delete left_result;
    delete right_result;

    cudaDeviceSynchronize();

    return result;
}

int seed=42;
std::mt19937 gen(seed);
ll randomValue(int min, int max) {
    std::uniform_int_distribution<ll> dist(min, max);
    return dist(gen);
}

std::vector<std::vector<ll>> generateRandomBlock(int k, int min_val = 1, int max_val = 10) {
    std::vector<std::vector<ll>> block(k, std::vector<ll>(k));
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            block[i][j] = randomValue(min_val, max_val);
        }
    }
    return block;
}

// Generate a large sparse block matrix with blocks only at k-aligned positions
SparseBlockMatrix generateLargeSparseBlockMatrix(int height, int width, int k, double density) {
    SparseBlockMatrix matrix(height, width, k);
    
    // Calculate the number of blocks in each dimension
    int num_blocks_height = (height + k - 1) / k;
    int num_blocks_width = (width + k - 1) / k;
    int total_possible_blocks = num_blocks_height * num_blocks_width;
    
    // Calculate how many blocks to generate based on density
    int num_blocks = static_cast<int>(density * total_possible_blocks);
    
    // Randomly select positions for blocks
    std::vector<std::pair<int, int>> potential_positions;
    for (int i = 0; i < num_blocks_height; i++) {
        for (int j = 0; j < num_blocks_width; j++) {
            potential_positions.push_back({i * k, j * k});
        }
    }
    
    // Shuffle the potential positions
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(potential_positions.begin(), potential_positions.end(), g);
    
    // Take the first 'num_blocks' positions
    for (int i = 0; i < num_blocks; i++) {
        int row_start = potential_positions[i].first;
        int col_start = potential_positions[i].second;
        
        // Generate a random block
        auto block = generateRandomBlock(k);
        
        // Add the block to the matrix
        matrix.addBlock(row_start, col_start, block);
    }
    
    return matrix;
}

// Function to profile matrix multiplication
void profileMatrixMultiplication(int height_A, int width_A, int height_B, int width_B, int k, double density_A, double density_B) {
    std::cout << "Profiling matrix multiplication with:" << std::endl;
    std::cout << "Matrix A: " << height_A << "x" << width_A << " (density: " << density_A * 100 << "%)" << std::endl;
    std::cout << "Matrix B: " << height_B << "x" << width_B << " (density: " << density_B * 100 << "%)" << std::endl;
    std::cout << "Block size: " << k << "x" << k << std::endl;
    
    // Generate matrices
    auto start_gen = std::chrono::high_resolution_clock::now();
    SparseBlockMatrix A = generateLargeSparseBlockMatrix(height_A, width_A, k, density_A);
    SparseBlockMatrix B = generateLargeSparseBlockMatrix(height_B, width_B, k, density_B);
    auto end_gen = std::chrono::high_resolution_clock::now();
    auto duration_gen = std::chrono::duration_cast<std::chrono::milliseconds>(end_gen - start_gen).count();
    
    std::cout << "Matrix generation time: " << duration_gen << " ms" << std::endl;

    // Multiply matrices and measure time
    auto start_mult = std::chrono::high_resolution_clock::now();
    SparseBlockMatrix* C = A.multiplyWith(B);
    auto end_mult = std::chrono::high_resolution_clock::now();
    auto duration_mult = std::chrono::duration_cast<std::chrono::milliseconds>(end_mult - start_mult).count();
    
    std::cout << "\nMatrix C (result) statistics:" << std::endl;
    C->printStats();
    
    std::cout << "\nMultiplication time: " << duration_mult << " ms" << std::endl;
    // Clean up
    delete C;
}

int sanity() {
    // Matrix A: 3 rows × 5 cols (block size k=2)
    SparseBlockMatrix A(5, 5, 3);
    A.addBlock(0, 0, {{1,2,3},{2,1,3},{3,4,1}});
    A.addBlock(0, 3, {{2,3,0},{2,1,0},{4,2,0}});
    A.addBlock(3, 0, {{1,2,5},{5,1,2},{0,0,0}});
    A.addBlock(3, 3, {{1,3,0},{4,1,0},{0,0,0}});

    SparseBlockMatrix* C = A.multiplyWith(A);
    
    C->print();
}

// Main function with different profiling scenarios
// int main() {
//     sanity();
//     int dim=1024*4;
//     std::cout << "=== Sparse Block Matrix Multiplication Profiler ===" << std::endl;    
//     profileMatrixMultiplication(dim, dim, dim, dim, 16, 0.1, 0.1);
//     std::cout << "\n---------------------------------------------------\n" << std::endl;
//     profileMatrixMultiplication(dim, dim, dim, dim, 32, 0.1, 0.1);
//     std::cout << "\n---------------------------------------------------\n" << std::endl;
//     return 0;
// }