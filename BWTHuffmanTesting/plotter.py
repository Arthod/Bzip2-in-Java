import os


BMH_path = "cantrbryBWT/"
BMH_sizes = {}

huffman_path = "cantrbryPureHuffman/"
huffman_sizes = {}

raw_path = "cantrbry/"
raw_sizes = {}

files = os.listdir(raw_path)

for file_name in files:
    BMH_sizes[file_name] = os.path.getsize(BMH_path + file_name)
    huffman_sizes[file_name] = os.path.getsize(huffman_path + file_name)
    raw_sizes[file_name] = os.path.getsize(raw_path + file_name)


# Sort raw file names by size
file_names_sorted = sorted(raw_sizes, key=raw_sizes.get)

# Plot it

import matplotlib.pyplot as plt
x = [i for i in range(len(file_names_sorted))]
plt.xticks(x, file_names_sorted)
plt.plot(x, [BMH_sizes[file_name] for file_name in file_names_sorted], "-", marker='o', label="BWT-MTF-Huffman")
plt.plot(x, [huffman_sizes[file_name] for file_name in file_names_sorted], "-", marker='o', label="Pure Huffman")
plt.plot(x, [raw_sizes[file_name] for file_name in file_names_sorted], "-", marker='o', label="Raw")
plt.legend()

plt.show()