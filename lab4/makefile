MPICXX = mpicxx
NVCC = nvcc

# Flags
CXXFLAGS = -std=c++17 -O3 -march=native -funroll-loops -ftree-vectorize -fopenmp
NVCCFLAGS = -std=c++14 -O3 -Xcompiler="-march=native -funroll-loops -ftree-vectorize -fopenmp"
CUDA_LDFLAGS = -lcudart -lcublas

# Find MPI include path using mpicxx
MPI_INCLUDE = $(shell mpicxx --showme:incdirs)
MPI_LIBDIRS = $(shell mpicxx --showme:libdirs)
MPI_LIBS = $(shell mpicxx --showme:libs)

# Object files
OBJS = sparse_mat.o main.o

# Targets
all: a4

# Compilation rules
main.o: main.cu
	$(NVCC) $(NVCCFLAGS) -I$(MPI_INCLUDE) -c $< -o $@

sparse_mat.o: sparse_mat1.cu
	$(NVCC) $(NVCCFLAGS) -I$(MPI_INCLUDE) -c $< -o $@

a4: $(OBJS)
	$(NVCC) $(NVCCFLAGS) $(OBJS) -o $@ $(CUDA_LDFLAGS) -Xcompiler="-fopenmp" -L$(MPI_LIBDIRS) -lmpi -lmpi_cxx

# Run the program
run: clean all
	mpirun -np 4 ./a4 test_folder/test_folder

# Clean up
clean:
	rm -f *.o main

# # Compilers
# MPICXX = mpicxx
# NVCC = nvcc

# # Flags
# CXXFLAGS = -std=c++17 -O3 -march=native -funroll-loops -ftree-vectorize -fopenmp
# NVCCFLAGS = -std=c++14 -O3 -Xcompiler="-march=native -funroll-loops -ftree-vectorize -fopenmp"
# CUDA_LDFLAGS = -lcudart

# # Find MPI include path using mpicxx
# MPI_INCLUDE = $(shell mpicxx --showme:incdirs)
# MPI_LIBDIRS = $(shell mpicxx --showme:libdirs)
# MPI_LIBS = $(shell mpicxx --showme:libs)

# # Object files
# OBJS = sparse_mat.o

# # Targets
# all: sparse_mat_exec

# sparse_mat.o: sparse_mat1.cu
# 	$(NVCC) $(NVCCFLAGS) -I$(MPI_INCLUDE) -c $< -o $@

# sparse_mat_exec: $(OBJS)
# 	$(NVCC) $(NVCCFLAGS) $(OBJS) -o $@ $(CUDA_LDFLAGS) -Xcompiler="-fopenmp" -L$(MPI_LIBDIRS) -lmpi -lmpi_cxx

# # Run the executable
# run: sparse_mat_exec
# 	./sparse_mat_exec

# # Clean up build files
# clean:
# 	rm -f *.o sparse_mat_exec
