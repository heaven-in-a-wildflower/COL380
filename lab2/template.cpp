#include "template.hpp"
#include <algorithm>
#include <unordered_map>
#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <mpi.h>
#include <chrono>

#define MAX_VERTEX_ID 1000000 // Adjust based on your data
#define MAX_COLORS 50

using namespace std;
using namespace std::chrono;

void init_mpi(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
}

void end_mpi()
{
    MPI_Finalize();
}

vector<vector<int>> degree_cen(vector<pair<int, int>> &partial_edge_list,
                               map<int, int> &partial_vector_color,
                               int k)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    chrono::time_point<high_resolution_clock> start, end;

    // Step 1: Preparing local color data with contiguous memory layout
    start = chrono::high_resolution_clock::now();

    // Pre-allocate to avoid reallocation
    const int color_count = partial_vector_color.size();

    // cout << "color_count=" << color_count << endl;

    int *local_vertices = new int[color_count];
    int *local_colors = new int[color_count];

    // Convert map to arrays with a single traversal
    int idx = 0;
    for (const auto &pair : partial_vector_color)
    {
        local_vertices[idx] = pair.first;
        local_colors[idx] = pair.second;
        ++idx;
    }

    // Gather color counts from all processes
    // //start = chrono::high_resolution_clock::now();
    int *color_counts = new int[size];
    MPI_Gather(&color_count, 1, MPI_INT, color_counts, 1, MPI_INT, 0, MPI_COMM_WORLD);
    end = chrono::high_resolution_clock::now();

    cout << rank << " :: " << ";color_count gathering time= "
         << chrono::duration_cast<chrono::microseconds>(end - start).count()
         << " [us]" << endl;

    // Calculate displacements for gathering variable-length data
    start = chrono::high_resolution_clock::now();
    int *displacements = new int[size](); // Initialize to zero
    if (rank == 0)
    {
        for (int i = 1; i < size; i++)
        {
            displacements[i] = displacements[i - 1] + color_counts[i - 1];
        }
    }

    // Prepare buffers for gathering all vertices and colors
    int *all_vertices = nullptr;
    int *all_colors = nullptr;
    int total_colors = 0;

    if (rank == 0)
    {
        total_colors = displacements[size - 1] + color_counts[size - 1];
        all_vertices = new int[total_colors];
        all_colors = new int[total_colors];
    }
    end = chrono::high_resolution_clock::now();

    cout << rank << " :: " << ";all_colors buffer prep time= "
         << chrono::duration_cast<chrono::microseconds>(end - start).count()
         << " [us]" << endl;

    // Gather vertices and colors to process 0
    start = chrono::high_resolution_clock::now();
    MPI_Gatherv(local_vertices, color_count, MPI_INT,
                all_vertices, color_counts, displacements,
                MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Gatherv(local_colors, color_count, MPI_INT,
                all_colors, color_counts, displacements,
                MPI_INT, 0, MPI_COMM_WORLD);
    end = chrono::high_resolution_clock::now();

    cout << rank << " :: " << ";all_colors and all_vertices gather time= "
         << chrono::duration_cast<chrono::microseconds>(end - start).count()
         << " [us]" << endl;

    // Free temporary arrays no longer needed
    delete[] local_vertices;
    delete[] local_colors;

    // Build global color map with better data structure
    start = chrono::high_resolution_clock::now();

    // Use a flat array-based hash table for O(1) lookups with better cache locality
    int *global_color_map = nullptr;

    if (rank == 0)
    {
        global_color_map = new int[MAX_VERTEX_ID];
        // memset(global_color_map, -1, MAX_VERTEX_ID * sizeof(int)); // -1 indicates no color
        for (int i = 0; i < MAX_VERTEX_ID; i++)
        {
            global_color_map[i] = -1; // -1 indicates no color
        }

        // Fill the map with gathered data
        for (int i = 0; i < total_colors; i++)
        {
            int vertex = all_vertices[i];
            int color = all_colors[i];
            if (vertex < MAX_VERTEX_ID)
            {
                global_color_map[vertex] = color;
            }
        }

        // Free arrays no longer needed
        delete[] all_vertices;
        delete[] all_colors;
    }
    end = chrono::high_resolution_clock::now();

    if (rank == 0)
        cout << rank << " :: " << ";global color map building time= "
             << chrono::duration_cast<chrono::microseconds>(end - start).count()
             << " [us]" << endl;

    // Step 2: Find vertices needing color information
    start = chrono::high_resolution_clock::now();

    // Use a bitset for faster lookups with better cache locality
    bool *vertex_needs_color = new bool[MAX_VERTEX_ID](); // Initialize to false

    // Mark vertices in edges
    int max_vertex_id = 0;
    for (const auto &edge : partial_edge_list)
    {
        vertex_needs_color[edge.first] = true;
        vertex_needs_color[edge.second] = true;
        max_vertex_id = std::max(max_vertex_id, std::max(edge.first, edge.second));
    }

    // Remove vertices that already have color info locally
    for (const auto &pair : partial_vector_color)
    {
        if (pair.first < MAX_VERTEX_ID)
        {
            vertex_needs_color[pair.first] = false;
        }
    }

    // Convert to packed array for MPI
    int request_count = 0;
    for (int i = 0; i <= max_vertex_id; i++)
    {
        if (vertex_needs_color[i])
        {
            request_count++;
        }
    }

    int *request_vertices = new int[request_count];
    int req_idx = 0;
    for (int i = 0; i <= max_vertex_id; i++)
    {
        if (vertex_needs_color[i])
        {
            request_vertices[req_idx++] = i;
        }
    }

    delete[] vertex_needs_color;

    end = chrono::high_resolution_clock::now();
    cout << rank << " :: " << ";vertices with missing color detection time= "
         << chrono::duration_cast<chrono::microseconds>(end - start).count()
         << " [us]" << endl;

    // Step 3: Request colors for unknown vertices
    start = chrono::high_resolution_clock::now();

    // Gather request counts to process 0
    MPI_Gather(&request_count, 1, MPI_INT, color_counts, 1, MPI_INT, 0, MPI_COMM_WORLD);

    end = chrono::high_resolution_clock::now();
    cout << rank << " :: " << ";requesting colors for vertices without color time= "
         << chrono::duration_cast<chrono::microseconds>(end - start).count()
         << " [us]" << endl;

    // Calculate displacements for gathering requests
    start = chrono::high_resolution_clock::now();

    if (rank == 0)
    {
        for (int i = 1; i < size; i++)
        {
            displacements[i] = displacements[i - 1] + color_counts[i - 1];
        }
    }

    // Prepare buffer for gathering all requests
    int *all_requests = nullptr;
    int total_requests = 0;

    if (rank == 0)
    {
        total_requests = displacements[size - 1] + color_counts[size - 1];
        all_requests = new int[total_requests];
    }

    // end = chrono::high_resolution_clock::now();

    // Gather all requests to process 0
    // start = chrono::high_resolution_clock::now();

    MPI_Gatherv(request_vertices, request_count, MPI_INT,
                all_requests, color_counts, displacements,
                MPI_INT, 0, MPI_COMM_WORLD);

    delete[] request_vertices; // No longer needed

    end = chrono::high_resolution_clock::now();
    cout << rank << " :: " << ";all_requests gathering time= "
         << chrono::duration_cast<chrono::microseconds>(end - start).count()
         << " [us]" << endl;

    // Process 0 prepares responses in flat arrays
    start = chrono::high_resolution_clock::now();

    int *response_sizes = new int[size](); // Initialize to 0
    int **vertex_responses = new int *[size];
    int **color_responses = new int *[size];

    if (rank == 0)
    {
        // First pass: count responses per process
        for (int i = 0; i < size; i++)
        {
            int start_idx = displacements[i];
            int end_idx = (i < size - 1) ? displacements[i + 1] : total_requests;

            for (int j = start_idx; j < end_idx; j++)
            {
                int vertex = all_requests[j];
                if (vertex < MAX_VERTEX_ID && global_color_map[vertex] != -1)
                {
                    response_sizes[i]++;
                }
            }

            // Allocate response arrays
            vertex_responses[i] = new int[response_sizes[i]];
            color_responses[i] = new int[response_sizes[i]];
        }

        // Second pass: fill response arrays
        int *current_indices = new int[size](); // Track current position in each response array

        for (int i = 0; i < size; i++)
        {
            int start_idx = displacements[i];
            int end_idx = (i < size - 1) ? displacements[i + 1] : total_requests;

            for (int j = start_idx; j < end_idx; j++)
            {
                int vertex = all_requests[j];
                if (vertex < MAX_VERTEX_ID && global_color_map[vertex] != -1)
                {
                    int idx = current_indices[i]++;
                    vertex_responses[i][idx] = vertex;
                    color_responses[i][idx] = global_color_map[vertex];
                }
            }
        }

        delete[] current_indices;
        delete[] all_requests; // No longer needed
    }

    end = chrono::high_resolution_clock::now();
    if (rank == 0)
        cout << rank << " :: " << ";response preparation time= "
             << chrono::duration_cast<chrono::microseconds>(end - start).count()
             << " [us]" << endl;

    // Broadcast response sizes
    int response_size = 0;
    start = chrono::high_resolution_clock::now();

    if (rank == 0)
    {
        for (int i = 1; i < size; i++)
        {
            MPI_Send(&response_sizes[i], 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
        response_size = response_sizes[0];
    }
    else
    {
        MPI_Recv(&response_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    end = chrono::high_resolution_clock::now();
    if (rank == 0)
        cout << rank << " :: " << ";response size broadcast time= "
             << chrono::duration_cast<chrono::microseconds>(end - start).count()
             << " [us]" << endl;

    // Receive color information
    int *received_vertices = new int[response_size];
    int *received_colors = new int[response_size];

    start = chrono::high_resolution_clock::now();
    /*
    if (rank == 0)
    {
        memcpy(received_vertices, vertex_responses[0], response_size * sizeof(int));
        memcpy(received_colors, color_responses[0], response_size * sizeof(int));
    }
    */
    if (rank == 0)
    {
        for (int i = 0; i < response_size; i++)
        {
            received_vertices[i] = vertex_responses[0][i];
            received_colors[i] = color_responses[0][i];
        }
    }
    else if (response_size > 0)
    {
        MPI_Recv(received_vertices, response_size, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(received_colors, response_size, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // end = chrono::high_resolution_clock::now();

    // Send responses to other processes
    // start = chrono::high_resolution_clock::now();

    if (rank == 0)
    {
        for (int i = 1; i < size; i++)
        {
            if (response_sizes[i] > 0)
            {
                MPI_Send(vertex_responses[i], response_sizes[i], MPI_INT, i, 1, MPI_COMM_WORLD);
                MPI_Send(color_responses[i], response_sizes[i], MPI_INT, i, 2, MPI_COMM_WORLD);
            }

            // Free memory after sending
            delete[] vertex_responses[i];
            delete[] color_responses[i];
        }

        // Free rank 0's arrays too
        delete[] vertex_responses[0];
        delete[] color_responses[0];
        delete[] vertex_responses;
        delete[] color_responses;
        delete[] response_sizes;

        // Free global color map
        delete[] global_color_map;
    }

    end = chrono::high_resolution_clock::now();
    cout << rank << " :: " << ";misc time= "
         << chrono::duration_cast<chrono::microseconds>(end - start).count()
         << " [us]" << endl;

    // Update local color map with received information
    start = chrono::high_resolution_clock::now();

    for (int i = 0; i < response_size; i++)
    {
        partial_vector_color[received_vertices[i]] = received_colors[i];
    }

    delete[] received_vertices;
    delete[] received_colors;

    end = chrono::high_resolution_clock::now();
    cout << rank << " :: " << ";partial_vector_color updation time= "
         << chrono::duration_cast<chrono::microseconds>(end - start).count()
         << " [us]" << endl;

    // Step 4: Create optimized adjacency list using arrays
    start = chrono::high_resolution_clock::now();

    // First pass: count neighbors per vertex
    int max_vertex = 0;
    for (const auto &edge : partial_edge_list)
    {
        max_vertex = std::max(max_vertex, std::max(edge.first, edge.second));
    }

    // Allocate count array
    int *neighbor_counts = new int[max_vertex + 1]();

    // Count neighbors
    for (const auto &edge : partial_edge_list)
    {
        neighbor_counts[edge.first]++;
        neighbor_counts[edge.second]++;
    }

    // Create offset array for CSR representation
    int *offsets = new int[max_vertex + 2]();
    for (int i = 1; i <= max_vertex + 1; i++)
    {
        offsets[i] = offsets[i - 1] + neighbor_counts[i - 1];
    }

    // Allocate adjacency array
    int total_edges = offsets[max_vertex + 1];
    int *adjacency = new int[total_edges];

    // Reset counts to use as current indices
    // memset(neighbor_counts, 0, (max_vertex + 1) * sizeof(int));
    for (int i = 0; i <= max_vertex; i++)
    {
        neighbor_counts[i] = 0;
    }

    // Fill adjacency list
    for (const auto &edge : partial_edge_list)
    {
        int v1 = edge.first;
        int v2 = edge.second;

        adjacency[offsets[v1] + neighbor_counts[v1]++] = v2;
        adjacency[offsets[v2] + neighbor_counts[v2]++] = v1;
    }

    delete[] neighbor_counts; // No longer needed

    end = chrono::high_resolution_clock::now();
    cout << rank << " :: " << ";adjacency list creation time= "
         << chrono::duration_cast<chrono::microseconds>(end - start).count()
         << " [us]" << endl;

    // Step 5: Find distinct colors and create color lookup
    start = chrono::high_resolution_clock::now();

    // Create flat array for color lookup
    int *color_lookup = new int[max_vertex + 1];
    for (int i = 0; i <= max_vertex; i++)
    {
        color_lookup[i] = -1; // -1 indicates no color
    }

    // Fill color lookup
    for (const auto &pair : partial_vector_color)
    {
        if (pair.first <= max_vertex)
        {
            color_lookup[pair.first] = pair.second;
        }
    }

    int max_color = -1;
    for (int i = 0; i <= max_vertex; i++)
    {
        if (color_lookup[i] > max_color)
        {
            max_color = color_lookup[i];
        }
    }

    cout << "max_color=" << max_color << endl;

    // After each process has identified its local distinct colors

    // First, each process collects its distinct colors
    set<int> local_distinct_colors_;
    for (const auto &pair_ : partial_vector_color)
    {
        if (pair_.second != -1)
        { // Skip uncolored vertices
            local_distinct_colors_.insert(pair_.second);
        }
    }

    cout << rank << " :: Local distinct colors: " << local_distinct_colors_.size() << endl;

    vector<int> global_colors_;

    // Convert the set to a vector for MPI communication
    vector<int> local_colors_(local_distinct_colors_.begin(), local_distinct_colors_.end());

    // Get counts from all processes to rank 0
    int local_count_ = local_colors_.size();
    vector<int> color_counts_;
    if (rank == 0)
    {
        color_counts_.resize(size);
    }
    MPI_Gather(&local_count_, 1, MPI_INT, color_counts_.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Only rank 0 needs to process the gathered data
    if (rank == 0)
    {
        // Calculate displacements
        vector<int> displacements_(size);
        displacements_[0] = 0;
        for (int i_ = 1; i_ < size; i_++)
        {
            displacements_[i_] = displacements_[i_ - 1] + color_counts_[i_ - 1];
        }

        // Allocate buffer for all colors
        int total_colors_ = displacements_[size - 1] + color_counts_[size - 1];
        vector<int> all_local_colors_(total_colors_);

        // Gather all local colors to rank 0
        MPI_Gatherv(local_colors_.data(), local_count_, MPI_INT,
                    all_local_colors_.data(), color_counts_.data(), displacements_.data(),
                    MPI_INT, 0, MPI_COMM_WORLD);

        // Find unique colors among all processes
        set<int> global_distinct_colors_(all_local_colors_.begin(), all_local_colors_.end());

        // Only rank 0 prints the total
        cout << "Total distinct colors across all processes: "
             << global_distinct_colors_.size() << endl;

        // Optional: store the set of global colors for further processing
        global_colors_ = vector<int>(global_distinct_colors_.begin(), global_distinct_colors_.end());
        std::sort(global_colors_.begin(), global_colors_.end());
    }
    else
    {
        // Non-root processes just send their data
        MPI_Gatherv(local_colors_.data(), local_count_, MPI_INT,
                    nullptr, nullptr, nullptr,
                    MPI_INT, 0, MPI_COMM_WORLD);
    }

    end = chrono::high_resolution_clock::now();
    cout << rank << " :: " << ";distinct color finding time= "
         << chrono::duration_cast<chrono::microseconds>(end - start).count()
         << " [us]" << endl;

    // Step 6: Optimize local centrality computation
    start = chrono::high_resolution_clock::now();

    // Structure to hold centrality data
    struct CentralityEntry
    {
        int vertex;
        int count;
    };

    // Pre-allocate vectors for each color
    vector<vector<CentralityEntry>> local_centrality(max_color + 1);
    for (int i = 0; i <= max_color; i++)
    {
        // Estimate size to avoid frequent reallocations
        local_centrality[i].reserve((max_vertex + 1) / 10); // Assume ~10% of vertices have this color
    }

    // Process each vertex - using vector instead of unordered_map
    for (int vertex = 0; vertex <= max_vertex; vertex++)
    {
        if (offsets[vertex] == offsets[vertex + 1])
            continue; // Skip vertices with no neighbors

        // Create counter for colors - using vector instead of unordered_map
        vector<int> color_counts(max_color + 1, 0);

        // Count neighbors by color
        for (int i = offsets[vertex]; i < offsets[vertex + 1]; i++)
        {
            int neighbor = adjacency[i];
            if (neighbor <= max_vertex && color_lookup[neighbor] != -1)
            {
                color_counts[color_lookup[neighbor]]++;
            }
        }

        // Store non-zero counts
        for (int color = 0; color <= max_color; color++)
        {
            if (color_counts[color] > 0)
            {
                local_centrality[color].push_back({vertex, color_counts[color]});
            }
        }
    }

    delete[] adjacency;
    delete[] offsets;
    delete[] color_lookup;
    end = chrono::high_resolution_clock::now();
    cout << rank << " :: " << ";local_centrality computation time= "
         << chrono::duration_cast<chrono::microseconds>(end - start).count()
         << " [us]" << endl;

    // Step 7: Prepare centrality data for sending with optimization
    start = chrono::high_resolution_clock::now();

    // Count total entries to preallocate
    int total_entries = 0;
    for (const auto &color_entries : local_centrality)
    {
        total_entries += color_entries.size();
    }

    // Preallocate centrality data array
    int *centrality_data = new int[total_entries * 3];
    int cent_idx = 0;

    // Fill centrality data
    for (int color = 0; color <= max_color; color++)
    {
        for (const auto &entry : local_centrality[color])
        {
            centrality_data[cent_idx++] = color;
            centrality_data[cent_idx++] = entry.vertex;
            centrality_data[cent_idx++] = entry.count;
        }
    }

    // Clear the local_centrality structure since we no longer need it
    for (auto &vec : local_centrality)
    {
        vector<CentralityEntry>().swap(vec); // Force memory deallocation
    }
    local_centrality.clear();

    end = chrono::high_resolution_clock::now();
    cout << rank << " :: " << ";centrality_data filling time= "
         << chrono::duration_cast<chrono::microseconds>(end - start).count()
         << " [us]" << endl;

    // Step 8: Gather centrality data
    start = chrono::high_resolution_clock::now();

    int centrality_count = cent_idx; // Actual number of entries * 3

    // Gather counts
    MPI_Gather(&centrality_count, 1, MPI_INT, color_counts, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculate displacements for gathering
    if (rank == 0)
    {
        displacements[0] = 0;
        for (int i = 1; i < size; i++)
        {
            displacements[i] = displacements[i - 1] + color_counts[i - 1];
        }
    }

    // Allocate buffer for all centrality data
    int *all_centrality_data = nullptr;
    if (rank == 0)
    {
        int total_centrality = displacements[size - 1] + color_counts[size - 1];
        all_centrality_data = new int[total_centrality];
    }

    // Gather all centrality data
    MPI_Gatherv(centrality_data, centrality_count, MPI_INT,
                all_centrality_data, color_counts, displacements,
                MPI_INT, 0, MPI_COMM_WORLD);

    delete[] centrality_data; // No longer needed

    end = chrono::high_resolution_clock::now();
    cout << rank << " :: " << ";all_centrality_data gathering time= "
         << chrono::duration_cast<chrono::microseconds>(end - start).count()
         << " [us]" << endl;

    // Step 9: Process 0 aggregates and finds top k
    vector<vector<int>> result;

    if (rank == 0)
    {
        start = chrono::high_resolution_clock::now();

        // Define constants for known constraints to avoid dynamic allocations

        // First pass: determine max color and max vertex
        int max_color = -1;
        int max_vertex = -1;
        for (int i = 0; i < displacements[size - 1] + color_counts[size - 1]; i += 3)
        {
            max_color = max(max_color, all_centrality_data[i]);
            max_vertex = max(max_vertex, all_centrality_data[i + 1]);
        }

        // Sanity check limits
        max_color = min(max_color, MAX_COLORS - 1);
        max_vertex = min(max_vertex, MAX_VERTEX_ID - 1);

        // Create flat arrays instead of vectors of vectors
        // Use a single allocation for all color-vertex centrality values
        int *color_centrality = new int[(max_color + 1) * (max_vertex + 1)](); // Initialize to zero

        // Aggregate the counts - direct indexing into flat array
        for (int i = 0; i < displacements[size - 1] + color_counts[size - 1]; i += 3)
        {
            int color = all_centrality_data[i];
            int vertex = all_centrality_data[i + 1];
            int count = all_centrality_data[i + 2];

            if (color <= max_color && vertex <= max_vertex)
            {
                color_centrality[color * (max_vertex + 1) + vertex] += count;
            }
        }

        delete[] all_centrality_data; // No longer needed

        end = chrono::high_resolution_clock::now();
        cout << rank << " :: " << ";global_centrality filling time= "
             << chrono::duration_cast<chrono::microseconds>(end - start).count()
             << " [us]" << endl;

        // Find top k vertices for each color using optimized approach
        start = chrono::high_resolution_clock::now();

        // Create result vectors once
        result.clear();
        result.reserve(max_color + 1);

        // Create a reusable array for vertex scores to avoid repeated allocations
        int *vertex_scores = new int[max_vertex + 1];
        int *vertex_indices = new int[max_vertex + 1];

        // Process each color
        for (int color : global_colors_)
        {
            if (color > max_color)
                continue; // Skip if color is out of bounds

            // Create a vector to store the top-k vertices for this color
            vector<int> top_k;
            top_k.reserve(k);

            // Copy scores for this color to the working array and initialize indices
            int score_count = 0;
            for (int vertex = 0; vertex <= max_vertex; vertex++)
            {
                int score = color_centrality[color * (max_vertex + 1) + vertex];
                if (score > 0)
                { // Only consider positive scores
                    vertex_scores[score_count] = score;
                    vertex_indices[score_count] = vertex;
                    score_count++;
                }
            }

            // Use partial sort to find the top k elements - much faster than sorting the entire array
            if (score_count > k)
            {
                // Sort only the first k elements (partial sort)
                for (int i = 0; i < k; i++)
                {
                    int max_idx = i;
                    for (int j = i + 1; j < score_count; j++)
                    {
                        // Check score first, then vertex ID for ties
                        if (vertex_scores[j] > vertex_scores[max_idx] ||
                            (vertex_scores[j] == vertex_scores[max_idx] &&
                             vertex_indices[j] < vertex_indices[max_idx]))
                        {
                            max_idx = j;
                        }
                    }
                    // Swap to position i
                    if (max_idx != i)
                    {
                        std::swap(vertex_scores[i], vertex_scores[max_idx]);
                        std::swap(vertex_indices[i], vertex_indices[max_idx]);
                    }
                }

                // Take only the top k vertices
                for (int i = 0; i < k; i++)
                {
                    top_k.push_back(vertex_indices[i]);
                }
            }
            else
            {
                // If we have fewer than k vertices with non-zero scores
                // First, sort what we have
                for (int i = 0; i < score_count; i++)
                {
                    int max_idx = i;
                    for (int j = i + 1; j < score_count; j++)
                    {
                        if (vertex_scores[j] > vertex_scores[max_idx] ||
                            (vertex_scores[j] == vertex_scores[max_idx] &&
                             vertex_indices[j] < vertex_indices[max_idx]))
                        {
                            max_idx = j;
                        }
                    }
                    // Swap to position i
                    if (max_idx != i)
                    {
                        std::swap(vertex_scores[i], vertex_scores[max_idx]);
                        std::swap(vertex_indices[i], vertex_indices[max_idx]);
                    }
                }

                // Add all vertices with non-zero scores
                for (int i = 0; i < score_count; i++)
                {
                    top_k.push_back(vertex_indices[i]);
                }

                // Create a bit array to track included vertices
                bool *included = new bool[max_vertex + 1]();
                for (int vertex : top_k)
                {
                    included[vertex] = true;
                }

                // Fill remaining slots with vertices having zero centrality
                int vertex = 0;
                while (top_k.size() < k && vertex <= max_vertex)
                {
                    if (!included[vertex])
                    {
                        top_k.push_back(vertex);
                    }
                    vertex++;
                }

                // Continue beyond max_vertex if needed
                while (top_k.size() < k)
                {
                    if (vertex <= max_vertex)
                    {
                        if (!included[vertex])
                        {
                            top_k.push_back(vertex);
                        }
                    }
                    else
                    {
                        top_k.push_back(vertex);
                    }
                    vertex++;
                }

                delete[] included;
            }

            // Add the final vector to the results
            result.push_back(std::move(top_k));
        }

        delete[] vertex_scores;
        delete[] vertex_indices;
        delete[] color_centrality;

        end = chrono::high_resolution_clock::now();
        cout << rank << " :: " << ";finding top k time= "
             << chrono::duration_cast<chrono::microseconds>(end - start).count()
             << " [us]" << endl;
    }

    // Cleanup remaining arrays
    delete[] color_counts;
    delete[] displacements;

    return result;
}