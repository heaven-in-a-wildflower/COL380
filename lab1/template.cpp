#include <iostream>
#include <vector>
#include <map>
#include <utility>
#include <cstdlib>
#include <ctime>
#include <random>
#include <algorithm>
#include <omp.h>
#include <chrono>
#include "check.h"
#include <immintrin.h> // For AVX intrinsics

using namespace std;
using namespace std::chrono;

#pragma GCC optimize("O3")
#pragma GCC optimize("unroll-loops")
#pragma GCC optimize("tree-vectorize")
#pragma GCC target("avx2,fma,sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,aes,pclmul,rdrnd,bmi,bmi2,lzcnt,f16c")

using namespace std;

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

// Efficient pair encoding into a single integer for hash set
inline uint32_t encode_pair(int x, int y, int l)
{
    return static_cast<uint32_t>(x * l + y);
}

// Decode pair from integer (for debugging if needed)
inline pair<int, int> decode_pair(uint32_t code, int l)
{
    return {code / l, code % l};
}

map<pair<int, int>, vector<vector<int>>> generate_matrix(int n, int m, int b)
{
    map<pair<int, int>, vector<vector<int>>> matrix_map;
    int l = n / m;

    // Build the list of possible positions.
    auto start = chrono::high_resolution_clock::now();

    // Allocate memory for result
    pair<int, int> *all_posns = new pair<int, int>[b];

    // Use flat array for faster set operations
    uint32_t max_positions = l * l;
    if (b > max_positions)
    {
        throw runtime_error("Requested more unique pairs than possible combinations");
    }

    // Create a pool of all possible positions
    uint32_t *position_pool = new uint32_t[max_positions];
    for (int x = 0; x < l; ++x)
    {
        for (int y = 0; y < l; ++y)
        {
            position_pool[x * l + y] = encode_pair(x, y, l);
        }
    }

    // --- Xorshift RNG setup ---
    // Define a simple xorshift32 engine.
    struct Xorshift32
    {
        uint32_t state;
        Xorshift32(uint32_t seed) : state(seed) {}
        uint32_t next()
        {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            return state;
        }
    };

    random_device rd;
    // Global xorshift engine seeded with random_device.
    Xorshift32 gen(rd());

    // Fisher-Yates shuffle to select b unique positions
    for (uint32_t i = 0; i < b; ++i)
    {
        // Generate a random index in the range [i, max_positions - 1]
        uint32_t j = i + (gen.next() % (max_positions - i));

        // Swap elements in the pool.
        uint32_t temp = position_pool[i];
        position_pool[i] = position_pool[j];
        position_pool[j] = temp;

        // Decode and store the selected position.
        all_posns[i] = decode_pair(position_pool[i], l);
    }

    // Cleanup the pool array.
    delete[] position_pool;

    auto end = chrono::high_resolution_clock::now();
    // cout << "Time taken for unique pair generation = "
    //      << chrono::duration_cast<chrono::microseconds>(end - start).count()
    //      << " [us]" << endl;

    // --- Parallel matrix creation using OpenMP tasks ---
    start = chrono::high_resolution_clock::now();

    int num_cores = omp_get_num_procs();
    omp_set_num_threads(num_cores);
#pragma omp parallel
    {
#pragma omp single nowait
        {
            for (size_t idx = 0; idx < b; ++idx)
            {
                auto pos = all_posns[idx];
#pragma omp task firstprivate(pos, m) shared(matrix_map) if (black_box())
                {
                    // Each task creates its own local xorshift engine.
                    // The seed is derived from a combination of random_device and the position.
                    Xorshift32 local_gen(rd() ^ (pos.first * 10007 + pos.second));

                    vector<vector<int>> matrix(m, vector<int>(m));
                    for (int i = 0; i < m; i++)
                    {
                        for (int j = 0; j < m; j++)
                        {
                            // Generate random number in [0, 256]
                            matrix[i][j] = local_gen.next() % 257;
                        }
                    }
#pragma omp critical
                    {
                        matrix_map.emplace(pos, std::move(matrix));
                    }
                }
            }
#pragma omp taskwait
        }
    }

    end = chrono::high_resolution_clock::now();
    // cout << "Time taken for matrix creation = "
    //      << chrono::duration_cast<chrono::microseconds>(end - start).count()
    //      << " [us]" << endl;

    delete[] all_posns;
    return matrix_map;
}
// -------------------------------------------------------------------
// Main multiplication driver.

double total_push_time = 0.0;
double block_mult_time = 0.0;
double non_zero_check_time = 0.0;

vector<float> matmul(map<pair<int, int>, vector<vector<int>>> &blocks, int n, int m, int k)
{
    // Cache line parameters
    constexpr int CACHE_LINE_SIZE = 64;
    constexpr int INT_SIZE = 4;
    constexpr int INTS_PER_CACHE_LINE = CACHE_LINE_SIZE / INT_SIZE;
    constexpr int AVX_INT_COUNT = 8; // __m256i can hold 8 integers

    // Convert 2D matrix to flattened 1D array
    auto flatten_matrix = [m](const vector<vector<int>> &mat) -> vector<int>
    {
        vector<int> flat(m * m);
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
    auto unflatten_matrix = [m](const vector<int> &flat) -> vector<vector<int>>
    {
        vector<vector<int>> mat(m, vector<int>(m));
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < m; j++)
            {
                mat[i][j] = flat[i * m + j];
            }
        }
        return mat;
    };

    auto remove_5_multiples = [=](map<pair<int, int>, vector<int>> &flat_blocks, int m)
    {
        vector<pair<int, int>> positions;
        for (const auto &[key, _] : flat_blocks)
        {
            positions.emplace_back(key);
        }

        vector<pair<int, int>> to_remove;
        int num_cores = omp_get_num_procs();
        omp_set_num_threads(num_cores);

#pragma omp parallel
        {
#pragma omp single
            {
                for (const auto &p : positions)
                {
#pragma omp task if (black_box())
                    {
                        int thread_id = omp_get_thread_num();
                        bool is_zero = true;
                        vector<int> &flat_mat = flat_blocks[p];

                        for (auto &cell : flat_mat)
                        {
                            if (cell % 5 == 0)
                            {
                                cell = 0;
                            }
                            else
                            {
                                is_zero = false;
                            }
                        }

                        if (is_zero)
                        {
#pragma omp critical
                            to_remove.emplace_back(p);
                        }
                    }
                }
#pragma omp taskwait
            }
        }

        for (const auto &p : to_remove)
        {
            flat_blocks.erase(p);
        }
    };

    // AVX-optimized multiply two matrices (without updating Pi)
    auto multiply_blocks = [=](const vector<int> &flat_a, const vector<int> &flat_b) -> vector<int>
    {
        vector<int> flat_result(m * m, 0);

        // Pre-transpose matrix B for better cache locality
        vector<int> b_transposed(m * m);
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < m; j++)
            {
                b_transposed[j * m + i] = flat_b[i * m + j];
            }
        }

        // Align b_transposed for better AVX performance if needed
        vector<int> b_aligned = b_transposed;
        if (m % AVX_INT_COUNT != 0)
        {
            b_aligned.resize((m + AVX_INT_COUNT - 1) / AVX_INT_COUNT * AVX_INT_COUNT * m, 0);
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < m; j++)
                {
                    b_aligned[i * ((m + AVX_INT_COUNT - 1) / AVX_INT_COUNT * AVX_INT_COUNT) + j] = b_transposed[i * m + j];
                }
            }
        }
        int num_cores = omp_get_num_procs();
        omp_set_num_threads(num_cores);
#pragma omp parallel
        {
#pragma omp single nowait
            {
                // Process by blocks to improve cache utilization
                const int block_size = min(m, INTS_PER_CACHE_LINE);

                for (int i = 0; i < m; i += block_size)
                {
                    for (int j = 0; j < m; j += block_size)
                    {
                        // Create a task for each block
#pragma omp task firstprivate(i, j) if (black_box())
                        {
                            // Process the block
                            for (int ii = i; ii < min(i + block_size, m); ii++)
                            {
                                for (int jj = j; jj < min(j + block_size, m); jj++)
                                {
                                    int sum = 0;
                                    // SIMD processing with AVX
                                    __m256i sum_vec = _mm256_setzero_si256();

                                    // Process 8 elements at a time with AVX
                                    for (int k = 0; k < m - (m % AVX_INT_COUNT); k += AVX_INT_COUNT)
                                    {
                                        __m256i a_vec = _mm256_loadu_si256((__m256i *)&flat_a[ii * m + k]);
                                        __m256i b_vec = _mm256_loadu_si256((__m256i *)&b_aligned[jj * ((m + AVX_INT_COUNT - 1) / AVX_INT_COUNT * AVX_INT_COUNT) + k]);
                                        __m256i mult = _mm256_mullo_epi32(a_vec, b_vec);
                                        sum_vec = _mm256_add_epi32(sum_vec, mult);
                                    }

                                    // Extract sum from vector
                                    int sum_array[AVX_INT_COUNT];
                                    _mm256_storeu_si256((__m256i *)sum_array, sum_vec);
                                    for (int k = 0; k < AVX_INT_COUNT; k++)
                                    {
                                        sum += sum_array[k];
                                    }

                                    // Process remaining elements
                                    for (int k = m - (m % AVX_INT_COUNT); k < m; k++)
                                    {
                                        sum += flat_a[ii * m + k] * b_transposed[jj * m + k];
                                    }

                                    flat_result[ii * m + jj] = sum;
                                }
                            }
                        }
                    }
                }
#pragma omp taskwait
            }
        }
        return flat_result;
    };

    // AVX-optimized multiply with Pi computation
    auto multiply_blocks_and_compute_Pi = [=](const vector<int> &flat_a,
                                              const vector<int> &flat_b,
                                              const int pos_a_x,
                                              vector<int> &Pi) -> vector<int>
    {
        vector<int> flat_result(m * m, 0);

        // Pre-transpose matrix B for better cache locality
        vector<int> b_transposed(m * m);
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < m; j++)
            {
                b_transposed[j * m + i] = flat_b[i * m + j];
            }
        }

        // Align b_transposed for better AVX performance if needed
        vector<int> b_aligned = b_transposed;
        if (m % AVX_INT_COUNT != 0)
        {
            b_aligned.resize((m + AVX_INT_COUNT - 1) / AVX_INT_COUNT * AVX_INT_COUNT * m, 0);
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < m; j++)
                {
                    b_aligned[i * ((m + AVX_INT_COUNT - 1) / AVX_INT_COUNT * AVX_INT_COUNT) + j] = b_transposed[i * m + j];
                }
            }
        }

        // Padded local_Pi to avoid false sharing
        // Each thread's count is aligned to cache line boundary
        const int padded_size = ((m + INTS_PER_CACHE_LINE - 1) / INTS_PER_CACHE_LINE) * INTS_PER_CACHE_LINE;
        vector<int> local_Pi(padded_size * omp_get_max_threads(), 0);
        int num_cores = omp_get_num_procs();
        omp_set_num_threads(num_cores);
#pragma omp parallel
        {
            int thread_id = omp_get_thread_num();

#pragma omp single nowait
            {
                for (int i = 0; i < m; i++)
                {
#pragma omp task firstprivate(i, thread_id) shared(flat_result, local_Pi) if (black_box())
                    {
                        int row_offset = thread_id * padded_size;
                        int rowCount = 0;

                        for (int j = 0; j < m; j++)
                        {
                            int sum = 0;
                            int count = 0;

                            // AVX for sum calculation
                            __m256i sum_vec = _mm256_setzero_si256();
                            __m256i count_vec = _mm256_setzero_si256();
                            __m256i zero_vec = _mm256_setzero_si256();

                            // Process 8 elements at a time
                            for (int k = 0; k < m - (m % AVX_INT_COUNT); k += AVX_INT_COUNT)
                            {
                                __m256i a_vec = _mm256_loadu_si256((__m256i *)&flat_a[i * m + k]);
                                __m256i b_vec = _mm256_loadu_si256((__m256i *)&b_aligned[j * ((m + AVX_INT_COUNT - 1) / AVX_INT_COUNT * AVX_INT_COUNT) + k]);

                                __m256i mult = _mm256_mullo_epi32(a_vec, b_vec);
                                sum_vec = _mm256_add_epi32(sum_vec, mult);

                                // Count non-zero products
                                __m256i nonzero_mask = _mm256_cmpgt_epi32(mult, zero_vec);
                                count_vec = _mm256_add_epi32(count_vec,
                                                             _mm256_and_si256(nonzero_mask, _mm256_set1_epi32(1)));
                            }

                            // Extract sum and count from vectors
                            int sum_array[AVX_INT_COUNT];
                            int count_array[AVX_INT_COUNT];
                            _mm256_storeu_si256((__m256i *)sum_array, sum_vec);
                            _mm256_storeu_si256((__m256i *)count_array, count_vec);

                            for (int k = 0; k < AVX_INT_COUNT; k++)
                            {
                                sum += sum_array[k];
                                count += count_array[k];
                            }

                            // Handle remaining elements
                            for (int k = m - (m % AVX_INT_COUNT); k < m; k++)
                            {
                                int prod = flat_a[i * m + k] * b_transposed[j * m + k];
                                sum += prod;
                                if (prod != 0)
                                    count++;
                            }

                            flat_result[i * m + j] = sum;
                            rowCount += count;
                        }
                        local_Pi[row_offset + i] = rowCount;
                    }
                }
#pragma omp taskwait
            }
        }

        // Reduce the local_Pi values to the global Pi
        for (int t = 0; t < omp_get_max_threads(); t++)
        {
            for (int i = 0; i < m; i++)
            {
                if (local_Pi[t * padded_size + i] > 0)
                {
#pragma omp atomic
                    Pi[m * pos_a_x + i] += local_Pi[t * padded_size + i];
                }
            }
        }

        return flat_result;
    };

    auto multiply_matrices = [&](const map<pair<int, int>, vector<int>> &A_flat,
                                 const map<pair<int, int>, vector<int>> &B_flat,
                                 int k, int m, vector<int> &Pi) -> map<pair<int, int>, vector<int>>
    {
        using BlockPos = pair<int, int>;
        using FlatMatrix = vector<int>;

        // Pre-process B into column-oriented structure
        map<int, vector<pair<int, const FlatMatrix *>>> B_columns;
        for (const auto &entry : B_flat)
        {
            const auto &pos = entry.first;
            const auto &matrix = entry.second;
            B_columns[pos.first].push_back({pos.second, &matrix});
        }

        map<BlockPos, FlatMatrix> result_map;
        int num_cores = omp_get_num_procs();
        omp_set_num_threads(num_cores);
#pragma omp parallel
        {
#pragma omp single nowait
            {
                for (const auto &[pos_a, matrix_a] : A_flat)
                {
                    auto it = B_columns.find(pos_a.second);
                    if (it != B_columns.end())
                    {
                        for (const auto &[col_b, matrix_b_ptr] : it->second)
                        {
#pragma omp task firstprivate(pos_a, col_b, matrix_a, matrix_b_ptr) if (black_box())
                            {
                                auto result_pos = make_pair(pos_a.first, col_b);
                                FlatMatrix mult_result;

                                if (k == 2)
                                {
                                    auto mult_start = chrono::high_resolution_clock::now();
                                    mult_result = multiply_blocks_and_compute_Pi(matrix_a, *matrix_b_ptr, pos_a.first, Pi);
                                    auto mult_end = chrono::high_resolution_clock::now();
// #pragma omp critical
//                                     {
//                                         block_mult_time += chrono::duration<double>(mult_end - mult_start).count();
//                                     }
                                }
                                else
                                {
                                    mult_result = multiply_blocks(matrix_a, *matrix_b_ptr);
                                }

                                // Check if the computed block is nonzero
                                auto non_zero_start = chrono::high_resolution_clock::now();
                                bool nonzero = false;
                                for (const auto &val : mult_result)
                                {
                                    if (val != 0)
                                    {
                                        nonzero = true;
                                        break;
                                    }
                                }
                                auto non_zero_end = chrono::high_resolution_clock::now();
// #pragma omp critical
//                                 {
//                                     non_zero_check_time += chrono::duration<double>(non_zero_end - non_zero_start).count();
//                                 }

                                if (nonzero)
                                {
#pragma omp critical
                                    {
                                        auto push_start = chrono::high_resolution_clock::now();

                                        auto [it, inserted] = result_map.insert({result_pos, mult_result});
                                        if (!inserted)
                                        {
                                            // If there's a collision, add the matrices
                                            auto &existing = it->second;
                                            for (size_t i = 0; i < m * m; i++)
                                            {
                                                existing[i] += mult_result[i];
                                            }
                                        }
                                        auto push_end = chrono::high_resolution_clock::now();
                                        // total_push_time += chrono::duration<double>(push_end - push_start).count();
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

#pragma omp taskwait
        return result_map;
    };

    auto pre_start = chrono::high_resolution_clock::now();

    // Convert all blocks to flattened format
    map<pair<int, int>, vector<int>> flat_blocks;
    for (const auto &[pos, mat] : blocks)
    {
        flat_blocks[pos] = flatten_matrix(mat);
    }

    remove_5_multiples(flat_blocks, m);
    auto pre_end = chrono::high_resolution_clock::now();
    // cout << "Pre-proc time= "
    //      << chrono::duration_cast<chrono::microseconds>(pre_end - pre_start).count()
    //      << " [us]" << endl;

    // Global arrays for row statistics.
    vector<int> Pi(n, 0);
    vector<int> Bi(n, 0);

    // If exponentiation is needed (k > 2), use square–and–multiply style.
    int exp = k;
    if (exp > 2)
    {
        auto result = flat_blocks;
        auto base = flat_blocks;
        exp -= 1;

        while (exp > 0)
        {
            if (exp % 2 == 1)
            {
                result = multiply_matrices(result, base, k, m, Pi);
            }
            base = multiply_matrices(base, base, k, m, Pi);
            exp /= 2;
        }
        flat_blocks = result;
    }

    if (k == 2)
    {
        auto start = chrono::high_resolution_clock::now();
        flat_blocks = multiply_matrices(flat_blocks, flat_blocks, k, m, Pi);
        auto end = chrono::high_resolution_clock::now();
        // cout << "Time taken by multiply_matrices = "
        //      << chrono::duration_cast<chrono::microseconds>(end - start).count()
        //      << " [us]" << endl;
        // cout << "Time taken for block mult operations: " << block_mult_time * 1e6 << " us" << endl;
        // cout << "Time taken for push_back operations: " << total_push_time * 1e6 << " us" << endl;
        // cout << "Time taken for checking non_zero: " << non_zero_check_time * 1e6 << " us" << endl;

        vector<float> row_statistics(n, 0.0f);

        // Update Bi using AVX
        __m256i increment = _mm256_set1_epi32(m);
        for (const auto &entry : flat_blocks)
        {
            int base = m * entry.first.first;
            int i = 0;

            // Process 8 elements at a time using AVX
            for (; i + 7 < m; i += 8)
            {
                __m256i current = _mm256_loadu_si256((__m256i *)&Bi[base + i]);
                current = _mm256_add_epi32(current, increment);
                _mm256_storeu_si256((__m256i *)&Bi[base + i], current);
            }

            // Handle remaining elements
            for (; i < m; i++)
            {
                Bi[base + i] += m;
            }
        }

        // Calculate row_statistics using AVX
        int i = 0;
        __m256 zeros = _mm256_setzero_ps();

        // Process 8 elements at a time
        for (; i + 7 < n; i += 8)
        {
            __m256i bi_values = _mm256_loadu_si256((__m256i *)&Bi[i]);
            __m256i pi_values = _mm256_loadu_si256((__m256i *)&Pi[i]);

            // Convert to float for division
            __m256 bi_float = _mm256_cvtepi32_ps(bi_values);
            __m256 pi_float = _mm256_cvtepi32_ps(pi_values);

            // Perform division, handling division by zero
            __m256 mask = _mm256_cmp_ps(bi_float, zeros, _CMP_NEQ_OQ);
            __m256 result = _mm256_div_ps(pi_float, bi_float);
            result = _mm256_and_ps(result, mask);

            _mm256_storeu_ps(&row_statistics[i], result);
        }

        // Handle remaining elements
        for (; i < n; i++)
        {
            if (Bi[i] != 0)
                row_statistics[i] = static_cast<float>(Pi[i]) / Bi[i];
        }

        // Convert flat_blocks back to blocks before returning
        blocks.clear();
        for (const auto &[pos, flat_mat] : flat_blocks)
        {
            blocks[pos] = unflatten_matrix(flat_mat);
        }
        return row_statistics;
    }

    // Convert flat_blocks back to blocks before returning
    blocks.clear();
    for (const auto &[pos, flat_mat] : flat_blocks)
    {
        blocks[pos] = unflatten_matrix(flat_mat);
    }

    return vector<float>();
}