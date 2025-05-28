import csv
import re

def process_gprof_file(input_file, output_time_csv, output_percent_csv):
    # Specify the functions of interest
    functions = ["readMatrix", "createMatrix", "writeMatrix", 
                 "matrixMultiplyIJK", "matrixMultiplyIKJ", 
                 "matrixMultiplyJIK", "matrixMultiplyJKI", 
                 "matrixMultiplyKIJ", "matrixMultiplyKJI"]
    
    data = {}
    current_type = None
    current_dim = None

    # Read the input file
    with open(input_file, "r") as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            # Check for a new section: "type: x, dim: y"
            section_match = re.match(r"type: (\d+), dim: (\d+)", line)
            if section_match:
                current_type = int(section_match.group(1))
                current_dim = int(section_match.group(2))
                data[(current_type, current_dim)] = []
            elif current_type is not None and current_dim is not None:
                # Extract function information: "% cumulative self ... name"
                func_match = re.match(
                    r"^\s*\d+\.\d+\s+\d+\.\d+\s+\d+\.\d+\s+\d+\s+[\d\.]+\s+[\d\.]+\s+([\w:<>~]+)",
                    line,
                )
                if func_match:
                    func_name = func_match.group(1).strip()
                    # Check if "ms/call" is in the previous line
                    if i > 0 and "ms/call" in lines[i - 1]:
                        func_time = float(line.split()[4]) / 1000  # Convert ms to seconds
                    elif i > 0 and "Ks/call" in lines[i - 1]:
                        func_time = float(line.split()[4]) * 1000  
                    else:
                        func_time = float(line.split()[4])
                    
                    func_percent = float(line.split()[0])  # Extract % time
                    data[(current_type, current_dim)].append((func_name, func_time, func_percent))

    # Write the "time" CSV
    with open(output_time_csv, "w", newline="") as time_csv:
        writer = csv.writer(time_csv)
        # Write the header
        time_header = ["type", "dim"] + functions
        writer.writerow(time_header)
        # Write the data rows
        for (t, d), funcs in data.items():
            row = [t, d]
            func_dict = {func[0]: func[1] for func in funcs}  # Map function name to cumulative time
            for func in functions:
                row.append(func_dict.get(func, 0.0))  # Default to 0.0 if missing
            writer.writerow(row)

    # Write the "%time" CSV
    with open(output_percent_csv, "w", newline="") as percent_csv:
        writer = csv.writer(percent_csv)
        # Write the header
        percent_header = ["type", "dim"] + functions
        writer.writerow(percent_header)
        # Write the data rows
        for (t, d), funcs in data.items():
            row = [t, d]
            func_dict = {func[0]: func[2] for func in funcs}  # Map function name to % time
            for func in functions:
                row.append(func_dict.get(func, 0.0))  # Default to 0.0 if missing
            writer.writerow(row)

# Usage
input_file = "gprof.txt"  # Replace with the path to your input file
output_time_csv = "gprof_times.csv"  # Replace with the desired time CSV output file
output_percent_csv = "gprof_times_percent.csv"  # Replace with the desired percent CSV output file

process_gprof_file(input_file, output_time_csv, output_percent_csv)
