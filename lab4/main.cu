#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <algorithm>
#include <chrono>
#include <omp.h>
#include <utility>
#include <set>
#include <mpi.h>
#include <cstdint>
#include "sparse_mat.h"
#include "timing_macros.h"

using namespace std;
using namespace std::chrono;

using ll=uint64_t;

struct MatrixMetadata
{
    int height;
    int width;
    std::vector<std::pair<int, int>> block_positions;
};

// Function to serialize a sparse block matrix for MPI transmission
void serializeMatrix(const SparseBlockMatrix* matrix, vector<int>& serialized_positions, 
                     vector<ll>& serialized_values) {
    serialized_positions.clear();
    serialized_values.clear();
    
    for (size_t i = 0; i < matrix->block_positions.size(); i++) {
        serialized_positions.push_back(matrix->block_positions[i].first);
        serialized_positions.push_back(matrix->block_positions[i].second);
        
        // Flatten the block values
        for (int r = 0; r < matrix->k; r++) {
            for (int c = 0; c < matrix->k; c++) {
                serialized_values.push_back(matrix->block_values[i][r][c]);
            }
        }
    }
}

// Function to deserialize a sparse block matrix received via MPI
SparseBlockMatrix* deserializeMatrix(int height, int width, int k, int block_count,
                                   const vector<int>& positions, const vector<ll>& values) {
    SparseBlockMatrix* matrix = new SparseBlockMatrix(height, width, k);
    
    for (int i = 0; i < block_count; i++) {
        int row_start = positions[i * 2];
        int col_start = positions[i * 2 + 1];
        
        // Reconstruct block values
        vector<vector<ll>> block_vals(k, vector<ll>(k));
        for (int r = 0; r < k; r++) {
            for (int c = 0; c < k; c++) {
                block_vals[r][c] = values[i * k * k + r * k + c];
            }
        }
        
        matrix->addBlock(row_start, col_start, block_vals);
    }
    
    return matrix;
}

int main(int argc, char **argv)
{
    int rank, size;
START_TIMER(mpi_init)
    MPI_Init(&argc, &argv);
    SparseBlockMatrix::init_cuda();  // initialize CUDA & cuBLAS *only once*
END_TIMER(mpi_init)

START_TIMER(read_matrices)
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int num_threads = 8;
    omp_set_num_threads(num_threads);
    string folder = argv[1];
    int N = 0, k = 0;
    vector<SparseBlockMatrix> matrices;
    
    if (rank == 0)
    {
        ifstream size_file(folder + "/size");
        if (!size_file.is_open())
        {
            cerr << "Failed to open size file in folder: " << folder << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        size_file >> N >> k;
        size_file.close();
    }
    
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&k, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Initialize GPU for this node (assuming one GPU per node)
    // In a real implementation, you would include CUDA headers and initialize CUDA here
    // cudaSetDevice(rank % 4); // Assuming 4 GPUs in total
    
    int matrices_per_proc = N / size;
    int remainder = N % size;
    int start_idx = rank * matrices_per_proc + min(rank, remainder);
    int end_idx = start_idx + matrices_per_proc + (rank < remainder ? 1 : 0);
    
    vector<vector<SparseBlockMatrix>> local_matrices(num_threads);
    vector<MatrixMetadata> local_metadata;
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        vector<SparseBlockMatrix> &my_vec = local_matrices[tid];
        vector<MatrixMetadata> my_meta;
        #pragma omp for schedule(static)
        for (int i = start_idx + 1; i <= end_idx; ++i)
        {
            string filename = folder + "/matrix" + to_string(i);
            try
            {
                SparseBlockMatrix mat = SparseBlockMatrix::fromFile(filename, k);
                my_vec.push_back(mat);
                MatrixMetadata meta;
                meta.height = mat.height;
                meta.width = mat.width;
                for (const auto &pos : mat.block_positions)
                {
                    meta.block_positions.emplace_back(pos);
                }
                my_meta.push_back(meta);
            }
            catch (const exception &e)
            {
                #pragma omp critical
                {
                    cerr << "Error reading matrix " << i << ": " << e.what() << endl;
                }
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
        #pragma omp critical
        {
            local_metadata.insert(local_metadata.end(), my_meta.begin(), my_meta.end());
        }
    }
    
    for (auto &vec : local_matrices)
    {
        matrices.insert(matrices.end(), vec.begin(), vec.end());
    }
END_TIMER(read_matrices)

START_TIMER(local_multiplication)
    // Each node computes its local product using GPU
    SparseBlockMatrix* local_product = nullptr;
    bool empty_flag = matrices.empty();
    if (!empty_flag) {
        // Replace with GPU version
        local_product = multiply_sequence(matrices);
    } else {
        local_product = new SparseBlockMatrix(0, 0, k);
    }
END_TIMER(local_multiplication)

START_TIMER(hierarchical_reduction)
    // Step 1: Set up the hierarchical reduction
    // Level 1: Nodes (0,1) and (2,3) reduce within pairs
    // Level 2: Nodes 0 and 2 reduce to final result at node 0
    
    int matrix_height = local_product->height;
    int matrix_width = local_product->width;
    int block_count = local_product->block_positions.size();
    
    // Define partner for first level reduction
    int partner = -1;
    int is_sender = false;
    
    // For a 4-node setup (can be generalized)
    if (size >= 2) {
        // Determine pairs: 0-1, 2-3, etc.
        if (rank % 2 == 1) {
            // Odd ranks send to even partners in first level
            partner = rank - 1;
            is_sender = true;
        } else if (rank % 2 == 0 && rank + 1 < size) {
            // Even ranks with an odd partner receive
            partner = rank + 1;
            is_sender = false;
        }
    }
    
    // Level 1 reduction
    if (partner != -1) {
        if (is_sender) {
            // Sender: Serialize and send matrix
            vector<int> positions;
            vector<ll> values;
            serializeMatrix(local_product, positions, values);
            
            // Send matrix metadata
            MPI_Send(&matrix_height, 1, MPI_INT, partner, 0, MPI_COMM_WORLD);
            MPI_Send(&matrix_width, 1, MPI_INT, partner, 1, MPI_COMM_WORLD);
            MPI_Send(&block_count, 1, MPI_INT, partner, 2, MPI_COMM_WORLD);
            
            // Send positions and values if there are any blocks
            if (block_count > 0) {
                MPI_Send(positions.data(), block_count * 2, MPI_INT, partner, 3, MPI_COMM_WORLD);
                MPI_Send(values.data(), block_count * k * k, MPI_UINT64_T, partner, 4, MPI_COMM_WORLD);
            }
            
            // Sender nodes are done after sending
            delete local_product;
            local_product = nullptr;
        } else {
            // Receiver: Get partner's matrix and multiply with local
            int partner_height, partner_width, partner_blocks;
            MPI_Recv(&partner_height, 1, MPI_INT, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&partner_width, 1, MPI_INT, partner, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&partner_blocks, 1, MPI_INT, partner, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            SparseBlockMatrix* partner_matrix = nullptr;
            
            if (partner_blocks > 0) {
                // Receive and deserialize partner's matrix
                vector<int> partner_positions(partner_blocks * 2);
                vector<ll> partner_values(partner_blocks * k * k);
                
                MPI_Recv(partner_positions.data(), partner_blocks * 2, MPI_INT, partner, 3, 
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(partner_values.data(), partner_blocks * k * k, MPI_UINT64_T, partner, 4, 
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                partner_matrix = deserializeMatrix(partner_height, partner_width, k, 
                                                 partner_blocks, partner_positions, partner_values);
            } else {
                partner_matrix = new SparseBlockMatrix(partner_height, partner_width, k);
            }
            
            // Multiply local and partner matrices using GPU
            vector<SparseBlockMatrix> to_multiply = {*local_product, *partner_matrix};
            delete local_product;
            delete partner_matrix;
            local_product = multiply_sequence(to_multiply);
        }
    }
    
    // Level 2 reduction - only needed with 4 or more nodes
    // Now nodes 0 and 2 (and potentially others for larger node counts) have partial products
    if (size >= 4) {
        partner = -1;
        is_sender = false;
        
        // For a 4-node setup (can be extended for larger node counts)
        if (rank == 2) {
            partner = 0;
            is_sender = true;
        } else if (rank == 0) {
            partner = 2;
            is_sender = false;
        }
        
        if (partner != -1) {
            if (is_sender && local_product != nullptr) {
                // Node 2 sends to Node 0
                matrix_height = local_product->height;
                matrix_width = local_product->width;
                block_count = local_product->block_positions.size();
                
                vector<int> positions;
                vector<ll> values;
                serializeMatrix(local_product, positions, values);
                
                // Send matrix metadata
                MPI_Send(&matrix_height, 1, MPI_INT, partner, 5, MPI_COMM_WORLD);
                MPI_Send(&matrix_width, 1, MPI_INT, partner, 6, MPI_COMM_WORLD);
                MPI_Send(&block_count, 1, MPI_INT, partner, 7, MPI_COMM_WORLD);
                
                // Send positions and values if there are any blocks
                if (block_count > 0) {
                    MPI_Send(positions.data(), block_count * 2, MPI_INT, partner, 8, MPI_COMM_WORLD);
                    MPI_Send(values.data(), block_count * k * k, MPI_UINT64_T, partner, 9, MPI_COMM_WORLD);
                }
                
                delete local_product;
                local_product = nullptr;
            } else if (!is_sender && local_product != nullptr) {
                // Node 0 receives from Node 2 and computes final product
                int partner_height, partner_width, partner_blocks;
                MPI_Recv(&partner_height, 1, MPI_INT, partner, 5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(&partner_width, 1, MPI_INT, partner, 6, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(&partner_blocks, 1, MPI_INT, partner, 7, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                SparseBlockMatrix* partner_matrix = nullptr;
                
                if (partner_blocks > 0) {
                    // Receive and deserialize partner's matrix
                    vector<int> partner_positions(partner_blocks * 2);
                    vector<ll> partner_values(partner_blocks * k * k);
                    
                    MPI_Recv(partner_positions.data(), partner_blocks * 2, MPI_INT, partner, 8, 
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Recv(partner_values.data(), partner_blocks * k * k, MPI_UINT64_T, partner, 9, 
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    
                    partner_matrix = deserializeMatrix(partner_height, partner_width, k, 
                                                     partner_blocks, partner_positions, partner_values);
                } else {
                    partner_matrix = new SparseBlockMatrix(partner_height, partner_width, k);
                }
                
                // Multiply local and partner matrices using GPU for final result
                vector<SparseBlockMatrix> to_multiply = {*local_product, *partner_matrix};
                delete local_product;
                delete partner_matrix;
                local_product = multiply_sequence(to_multiply);
            }
        }
    }
END_TIMER(hierarchical_reduction)

START_TIMER(final_output)
    // Output final result from node 0
    if (rank == 0 && local_product != nullptr) {
        cout << "Final result dimensions: " << local_product->height << "x" 
             << local_product->width << " with " << local_product->block_positions.size() 
             << " non-zero blocks" << endl;
        
        // Write the final result to a file
        local_product->writeToFile(folder + "/matrix");
    }
    
    // Clean up resources
    if (local_product != nullptr) {
        delete local_product;
    }
END_TIMER(final_output)

    SparseBlockMatrix::shutdown_cuda();
    MPI_Finalize();
    return 0;
}

