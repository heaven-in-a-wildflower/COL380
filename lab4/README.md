# Lab 4: Distributed Sparse Matrix Multiplication

## Problem statement

Compute the product of a sequence of sparse, `k x k` block matrices. The matrices are supplied as files in a test folder, with dimensions compatible for multiplication. The implementation distributes the matrices across MPI ranks, computes local products in a divide-and-conquer sequence with OpenMP and CUDA/cuBLAS, combines partial products through a hierarchical MPI reduction, and writes the final result to `matrix`.

## Files

- `main.cu` - MPI/OpenMP entry point. Reads the matrix dimensions and distributes `matrix1` onward across ranks, initializes CUDA, performs local and hierarchical products, times major stages, and writes rank 0's final result.
- `sparse_mat.h` - Declares `SparseBlockMatrix`, sparse block storage and file I/O, CUDA/cuBLAS initialization, and matrix multiplication helpers.
- `sparse_mat1.cu` - Implements block-matrix loading and I/O, dense CUDA matrix multiplication through cuBLAS, re-sparsification of the result, and divide-and-conquer sequence multiplication.
- `makefile` - Builds `a4` with NVCC and MPI/OpenMP support. It also provides `run` and `clean` targets.
- `gen.py` - Generates a test folder containing `size` and sequential `matrix1`, `matrix2`, ... files with dimensions compatible for multiplication.
- `timing_macros.h` - Defines macros used to report initialization, input, multiplication, reduction, and output timings.
- `test_folder/` - Sample generated matrix data; `test_folder/test_folder/size` defines `N` and `k`, and `matrix1` through `matrixN` contain the inputs.
- `report.pdf` - Lab report describing implementation and performance results.
- `metadata.yml` - Submission metadata.
- `.vscode/` - Local VS Code C/C++ configuration.
