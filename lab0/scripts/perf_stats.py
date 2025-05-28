import re
import csv

# Input and output file paths
input_file = "perf.txt"
output_file = "perf_times_percent.csv"

# Functions to search for
functions = ["readMatrix", "createMatrix", "writeMatrix", "matrixMultiplyIJK",
            "matrixMultiplyIKJ", "matrixMultiplyJIK", "matrixMultiplyJKI",
            "matrixMultiplyKIJ", "matrixMultiplyKJI"]

# Patterns to match
pattern1 = re.compile(r"\|--([0-9]+\.[0-9]+)%--(\w+)")  # Format: |--98.13%--func_name
pattern2 = re.compile(r"([0-9]+\.[0-9]+)%\s+[0-9]+\.[0-9]+%\s+\[.]\s+(\w+)")  # Format: 0.29% 0.00% [.] func_name

def parse_perf_file(input_file):
    # Initialize results dictionary with 0.0 for all functions
    results = {func: 0.0 for func in functions}
    
    with open(input_file, "r") as file:
        lines = file.readlines()
        
        # Extract type and dim from the first line
        type_dim_match = re.match(r"type:\s*(\d+),\s*dim:\s*(\d+)", lines[1])
        if not type_dim_match:
            raise ValueError("Type and dim information not found in the file.")
            
        type_value = int(type_dim_match.group(1))
        dim_value = int(type_dim_match.group(2))
        
        # Search for function times
        for line in lines:
            # Check for pattern1 (|--98.13%--func_name)
            match1 = pattern1.search(line)
            if match1:
                percent, func_name = float(match1.group(1)), match1.group(2)
                if func_name in functions:
                    results[func_name] = percent
                    
            # Check for pattern2 (0.29% [.] func_name)
            match2 = pattern2.search(line)
            if match2:
                percent, func_name = float(match2.group(1)), match2.group(2)
                if func_name in functions:
                    results[func_name] = percent
                    
    return type_value, dim_value, results

def write_to_csv(output_file, type_value, dim_value, results):
    with open(output_file, "a", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write data
        csvwriter.writerow([
            type_value, 
            dim_value, 
            results["readMatrix"],
            results["createMatrix"],
            results["writeMatrix"],
            results["matrixMultiplyIJK"],
            results["matrixMultiplyIKJ"],
            results["matrixMultiplyJIK"],
            results["matrixMultiplyJKI"],
            results["matrixMultiplyKIJ"],
            results["matrixMultiplyKJI"]
        ])

# Parse the perf.txt file
type_value, dim_value, results = parse_perf_file(input_file)

# Write results to CSV
write_to_csv(output_file, type_value, dim_value, results)

# Part 2

# Input file paths
perf_file = "perf_times_percent.csv"
cache_file = "cache_stats.csv"

# Output file path
output_file = "perf_times.csv"

def multiply_rows(perf_file, cache_file, output_file):
    # Read the cache stats file to get the multiplier values
    with open(cache_file, "r") as cache_csv:
        cache_reader = csv.reader(cache_csv)
        cache_data = list(cache_reader)
    
    # Extract the second last column from cache_data
    cache_multipliers = [float(row[-2])/100 for row in cache_data[1:]]  # Skip the header
    
    # Read the performance times percent file and multiply
    with open(perf_file, "r") as perf_csv, open(output_file, "w", newline="") as output_csv:
        perf_reader = csv.reader(perf_csv)
        perf_writer = csv.writer(output_csv)
        
        # Process the rows
        for i, row in enumerate(perf_reader):
            if i == 0:  # Write header directly
                perf_writer.writerow(row)
            else:
                # Multiply relevant columns by the cache multiplier
                type_dim = row[:2]  # Keep first two columns as is
                perf_values = [float(val) for val in row[2:]]
                multiplier = cache_multipliers[i - 1]  # Match row index
                updated_values = [round(val * multiplier,4) for val in perf_values]
                
                # Write the updated row to the output file
                perf_writer.writerow(type_dim + updated_values)

# Call the function
multiply_rows(perf_file, cache_file, output_file)

