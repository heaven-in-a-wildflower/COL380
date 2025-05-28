import os
import random

def generate_graph(vertices, edges, colors, num_files):
    # Ensure testcase directory exists
    os.makedirs("testcase_3", exist_ok=True)
    
    # Assign a random color (0-9) to each vertex
    colors_used = set()
    vertex_colors = {}
    for v in range(vertices):
        rc = random.randint(0, colors - 1)
        vertex_colors[v] = rc
        colors_used.add(rc)

    print(f"Colors used: {len(colors_used)}")
    
    # Generate unique edges
    edge_set = set()
    while len(edge_set) < edges:
        u, v = random.sample(range(vertices), 2)
        if (u, v) not in edge_set and (v, u) not in edge_set:
            edge_set.add((u, v))
    
    edges_list = list(edge_set)
    
    # Split data into 40 files
    vertices_per_file = vertices // num_files
    edges_per_file = edges // num_files
    
    for i in range(num_files):
        print(f"Generating graph_{i}.txt")
        file_path = os.path.join("testcase_3", f"graph_{i}.txt")
        with open(file_path, "w") as f:
            f.write(f"{vertices_per_file} {edges_per_file}\n")
            
            # Write vertex-color pairs
            start_vertex = i * vertices_per_file
            end_vertex = start_vertex + vertices_per_file
            for v in range(start_vertex, end_vertex):
                f.write(f"{v} {vertex_colors[v]}\n")
            
            # Write edges
            start_edge = i * edges_per_file
            end_edge = start_edge + edges_per_file
            for u, v in edges_list[start_edge:end_edge]:
                f.write(f"{u} {v}\n")

import subprocess

def files_are_different(file1, file2):
    cmd = f"diff --brief <(tail -n +2 {file1}) <(tail -n +2 {file2})"
    result = subprocess.run(cmd, shell=True, executable='/bin/bash', stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return result.returncode == 1  # Returns True if files are different, False if identical


# Parameters
total_vertices = 100000
total_edges = 20000000
color_range = 50
num_files = 40

print("------------generating graph------------")
generate_graph(total_vertices, total_edges, color_range, num_files)
print("------------graph generated------------")

# # run the executable check :  mpirun -np 4 ./check TestCase1 output.txt 10

# # run the executable check :  mpirun -np 4 ./check TestCase2 output.txt 10
# os.system("mpirun -np 4 ./checkakshad testcase akshad.txt 10")
# print("------------akshad.txt generated------------")
# os.system("mpirun -np 4 ./checkumesh testcase umesh.txt 10")
# print("------------umesh.txt generated------------")
# os.system("mpirun -np 4 ./checkrathi testcase rathi.txt 10")
# print("------------rathi.txt generated------------")
# os.system("mpirun -np 4 ./checkvraj testcase vraj.txt 10")
# print("------------vraj.txt generated------------")

# file1 = "akshad.txt"
# file2 = "umesh.txt"
# file3 = "rathi.txt"
# file4 = "vraj.txt"

# any_different = False

# if files_are_different(file1, file2):
#     any_different = True
#     print("--------aksahd.txt and umesh.txt are different.----------")
# if files_are_different(file1, file3):
#     any_different = True
#     print("--------aksahd.txt and rathi.txt are different.----------")
# if files_are_different(file2, file3):
#     any_different = True
#     print("--------umesh.txt and rathi.txt are different.----------")
# if files_are_different(file1, file4):
#     any_different = True
#     print("--------aksahd.txt and vraj.txt are different.----------")
# if files_are_different(file2, file4):
#     any_different = True
#     print("--------umesh.txt and vraj.txt are different.----------")
# if files_are_different(file3, file4):
#     any_different = True
#     print("--------rathi.txt and vraj.txt are different.----------")

# if not any_different:
#     print("All files are identical.")
#     #delete the testcase folder
#     os.system("rm -r testcase")