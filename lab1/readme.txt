Name: Aneeket Yadav
Entry no: 2022CS11116

Optimisations performed-
a.Random matrix generation
1. Xorshift random number generation
2. Fisher-Yates shuffle to select b unique positions
3. Efficient pair encoding into a single integer for hash set

b. Matrix multiplication
1. Pre-processing matrix B in C = A X B for better cache locality.
2. Flattened matrix so that more elements can be stored in the same cache block.
3. No false sharing
4. AVX instructions for SIMD processing
5. Pre-processing const map<pair<int, int>, vector<int>> &B_flat to map<int, vector<pair<int, const vector<int>*>>> B_columns. This decreases the time complexity of iterating over two maps during matrix multiplication.
6. Binary exponentiation for k>2

Optimisations 4 and 5 were quite critical in improving performance.
As for parallel scaling, it is generally observed that performance scales from 2 to 8 cores, but degrades from then on, with most time being taken for n=32 cores. This may have to do with the fact that a single sector contains 24 processors, thus migrating a process to another sector would be detrimental in a NUMA architecture.
