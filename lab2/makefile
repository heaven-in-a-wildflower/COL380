# Compiler and flags
MPICXX = mpic++
CXXFLAGS = -O3 -march=native -funroll-loops -ftree-vectorize -Wall -Wextra -pedantic -MMD -MP
PES = 4

# Executable name
TARGET = check

# Source files
SRCS = check.cpp template6.cpp
OBJS = $(SRCS:.cpp=.o)

# Default rule
all: $(TARGET)

# Compile the program
$(TARGET): $(OBJS)
	$(MPICXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

# Rule for compiling each .cpp to .o
%.o: %.cpp
	$(MPICXX) $(CXXFLAGS) -c $< -o $@

# Dependency tracking for headers
-include $(OBJS:.o=.d)

run: $(TARGET)
	@echo "Running MPI program..."
	mpirun -np $(PES) ./$(TARGET) testcase_3/testcase mhaskex.txt 10 > mhaskex_log.txt
	@echo "Sorting output..."
	sort --numeric-sort --stable mhaskex_log.txt | tee mhaskex_log_sorted.txt
	@echo "Comparing..."
	diff testcase_3/testcase/akshad.txt mhaskex.txt

run1: $(TARGET)
	@echo "Running MPI program..."
	mpirun -np $(PES) ./$(TARGET) test/test/testcase1 mhaske1.txt 10 > mhaske1_log.txt
	@echo "Sorting output..."
	sort --numeric-sort --stable mhaske1_log.txt | tee mhaske1_log_sorted.txt
	@echo "Comparing..."
	diff test/test/output1.txt mhaske1.txt

run2: $(TARGET)
	@echo "Running MPI program..."
	mpirun -np $(PES) ./$(TARGET) test/test/testcase2 mhaske2.txt 10 > mhaske2_log.txt
	@echo "Sorting output..."
	sort --numeric-sort --stable mhaske2_log.txt | tee mhaske2_log_sorted.txt
	@echo "Comparing..."
	diff test/test/output2.txt mhaske2.txt

# Clean generated files
clean:
	rm -f $(TARGET) mhaske1_log.txt mhaske1_log_sorted.txt mhaske2_log.txt mhaske2_log_sorted.txt
