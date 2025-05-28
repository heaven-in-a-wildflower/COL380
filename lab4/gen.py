import os
import random
import argparse
import numpy as np

def generate_block(k, populate_chance):
    """Generate a kxk block with values (some zeros based on populate_chance)."""
    block = np.zeros((k, k), dtype=int)
    for i in range(k):
        for j in range(k):
            if random.random() < populate_chance:
                block[i][j] = random.randint(1, 100)
    return block

def generate_matrix_file(path, height, width, k, density):
    num_block_rows = height // k
    num_block_cols = width // k
    possible_blocks = [(i * k, j * k) for i in range(num_block_rows) for j in range(num_block_cols)]
    num_blocks = int(len(possible_blocks) * density)
    chosen_blocks = random.sample(possible_blocks, num_blocks)

    with open(path, 'w') as f:
        f.write(f"{height} {width}\n")
        f.write(f"{num_blocks}\n")
        for row, col in chosen_blocks:
            f.write(f"{row} {col}\n")

        for row, col in chosen_blocks:
            block = generate_block(k, populate_chance=0.9)
            for r in range(k):
                f.write(" ".join(str(val) for val in block[r]) + "\n")
            # f.write("\n")

def generate_folder(folder_path, N, k, min_dim, max_dim, density):
    os.makedirs(folder_path, exist_ok=True)

    with open(os.path.join(folder_path, "size"), 'w') as f:
        f.write(f"{N} {k}\n")

    # Generate the dimensions such that they allow matrix multiplication
    dimensions = []
    first_height = random.randint(min_dim, max_dim) // k * k
    dimensions.append(first_height)

    for _ in range(N):
        next_dim = random.randint(min_dim, max_dim) // k * k
        if next_dim == 0:
            next_dim = k
        dimensions.append(next_dim)

    # dimensions[i] = height of matrix i
    # dimensions[i+1] = width of matrix i
    for i in range(N):
        height = dimensions[i]
        width = dimensions[i + 1]
        generate_matrix_file(os.path.join(folder_path, f"matrix{i + 1}"), height, width, k, density)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate sparse kxk block matrices.")
    parser.add_argument("--folder", type=str, default="F", help="Folder to store matrix files")
    parser.add_argument("--num_matrices", type=int, default=5, help="Number of matrices to generate")
    parser.add_argument("--block_size", type=int, default=2, help="Block size k")
    parser.add_argument("--min_dim", type=int, default=16, help="Minimum dimension for matrix height/width")
    parser.add_argument("--max_dim", type=int, default=16, help="Maximum dimension for matrix height/width")
    parser.add_argument("--density", type=float, default=0.2, help="Approximate % of blocks to populate (0 to 1)")

    args = parser.parse_args()
    assert args.min_dim <= args.max_dim
    assert args.min_dim >= args.block_size

    generate_folder(args.folder, args.num_matrices, args.block_size, args.min_dim, args.max_dim, args.density)
    print(f"Generated {args.num_matrices} matrices in '{args.folder}' that are compatible for multiplication.")
