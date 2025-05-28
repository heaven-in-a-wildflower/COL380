import sys
import binascii

def print_bin_file(file_path):
    try:
        with open(file_path, 'rb') as file:
            data = file.read()
            hex_data = binascii.hexlify(data)
            
            # Print in a formatted way
            for i in range(0, len(hex_data), 16):
                hex_chunk = hex_data[i:i+16]
                ascii_chunk = ''.join([chr(b) if 32 <= b <= 126 else '.' for b in hex_chunk])
                formatted_line = ' '.join([hex_chunk[j:j+2].decode() for j in range(0, len(hex_chunk), 2)])
                print(f"{formatted_line:<48} {ascii_chunk}")
    except Exception as e:
        print(f"Error reading the file: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python print_bin.py <path_to_bin_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    print_bin_file(file_path)
