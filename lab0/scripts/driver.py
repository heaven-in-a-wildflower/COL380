import numpy as np, os, time
import matplotlib.pyplot as plt
import json
import subprocess
import csv 

def write_perf_csv_header(output_file):
    with open(output_file, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["type", "dim","readMatrix", "createMatrix", "writeMatrix", "matrixMultiplyIJK","matrixMultiplyIKJ","matrixMultiplyJIK","matrixMultiplyJKI","matrixMultiplyKIJ","matrixMultiplyKJI"])

def execute_test_case(type, number_row1, number_col1, number_col2, path_input, path_output):
    # Generate random matrices
    mtx_A = np.random.random(size = (number_row1, number_col1)) * 1e2 # dtype = float64
    mtx_B = np.random.random(size = (number_col1, number_col2)) * 1e2 # dtype = float64

    # Matrix multiplication
    mtx_C = (mtx_A @ mtx_B).flatten() # dtpye = float64

    # Create directories if does not exist
    if not os.path.exists(path_input):
        os.makedirs(path_input)
    if not os.path.exists(path_output):
        os.makedirs(path_output)

    # Store the input matrices
    with open(f"{path_input}/mtx_A.bin", "wb") as fp:
        fp.write(mtx_A.tobytes())
    with open(f"{path_input}/mtx_B.bin", "wb") as fp:
        fp.write(mtx_B.tobytes())
    
    # Compile the students code
    os.system("make")
    # running the code compiled with -g or -pg flags takes approximately the same time as running the code compiled without these flags.

    # Execute the students code    
    command1 = f"time sudo taskset -c 0 perf stat -e cycles,instructions,cache-references,cache-misses ./main {type} {number_row1} {number_col1} {number_col2} {path_input} {path_output}"
    
    command2 = f"taskset -c 0 sudo perf record -F 4000 --call-graph dwarf ./main {type} {number_row1} {number_col1} {number_col2} {path_input} {path_output}"

    command3 = "sudo perf report --stdio --sort symbol"

    command4 = f"taskset -c 0 ./main {type} {number_row1} {number_col1} {number_col2} {path_input} {path_output}"

    command5 = "taskset -c 0 gprof -p main gmon.out"

    # Gather cache stats
    with open("cache_stats.txt", "a") as file:
        file.write(f"\ntype: {type}, dim: {number_row1}\n")
    with open("cache_stats.txt", "a") as file:
        subprocess.run(command1, shell=True, stdout=file, stderr=file)
    os.system("python3 cache_stats.py")

    # Gather perf data
    with open("perf.txt", "w") as file:
        file.write(f"\ntype: {type}, dim: {number_row1}\n")

    time_start = time.perf_counter()
    os.system(command2)
    time_duration = time.perf_counter() - time_start # in seconds
    print("time taken: ", time_duration)
    
    with open("perf.txt", "a") as file:
        subprocess.run(command3, shell=True, stdout=file, stderr=file)
        os.system("python3 perf_stats.py")

    # Gather gprof data
    # os.system(command4)
    with open("gprof.txt", "a") as file:
        file.write(f"\ntype: {type}, dim: {number_row1}\n")
    with open("gprof.txt", "a") as file:
        subprocess.run(command5, shell=True, stdout=file, stderr=file)
    os.system("python3 gprof_stats.py") 

    # Get the students output matrix
    with open(f"{path_output}/mtx_C.bin", "rb") as fp:
        student_result = np.frombuffer(fp.read(), dtype=mtx_C.dtype)
    
    # Check if the result matrix dimensions match
    if mtx_C.shape != student_result.shape:
        print("The result matrix shape didn't match")
        return False, time_duration
    
    # Check if the student's result is close to the numpy's result within a tolerance
    result = np.allclose(mtx_C, student_result, rtol=1e-10, atol=1e-12)

    return result, time_duration

if __name__ == "__main__":

    # l = np.array([1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000])
    # dims = np.array([5000,4000])
    dims = np.array([5000,4000,3000,2000,1000])

    types = [0,1,2,3,4,5]
    # types = [0]
    # type_labels = ["IJK", "IKJ", "JIK", "JKI", "KIJ", "KJI"]
    # type_labels = ["IJK"]

    time_dict = {t: [] for t in types}
    # l_p = []

    with open("cache_stats.txt", "w") as file:
        pass

    with open("gprof.txt", "w") as file:
        pass

    with open("python_times.csv", "w") as csvfile:
        pass

    write_perf_csv_header("perf_times_percent.csv")

    for dim in dims:
        for t in types:
            print("running for n = ", dim, ", type = ", t)
            result, time_duration = execute_test_case(t, dim, dim, dim, "./input_path/", "./output_path/")
            time_dict[t].append(time_duration)

            if result:
                print("Test Case passed")
            else:
                print("Test Case failed")
            print("\n")

            # Write the performance times to a CSV file
            with open("python_times.csv", "a", newline="") as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow([t, dim, round(time_duration,4)]) 