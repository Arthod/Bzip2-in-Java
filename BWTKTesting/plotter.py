import os


path = "cantrbry"
size = 5

encoded_sizes = [{} for _ in range(size)]
raw_sizes = {}

files = os.listdir(path)

for i in range(size):
    for file_name in files:
        print(path + str(i+1) + "/" + file_name, os.path.getsize(path + str(i+1) + "/" + file_name))
        encoded_sizes[i][file_name] = os.path.getsize(path + str(i+1) + "/" + file_name)
for file_name in files:
    raw_sizes[file_name] = os.path.getsize(path + "/" + file_name)
print(encoded_sizes)

# Sort raw file names by size
file_names_sorted = sorted(raw_sizes, key=raw_sizes.get)

# Plot it

import matplotlib.pyplot as plt
x = [i for i in range(len(file_names_sorted))]
plt.xticks(x, file_names_sorted)
plt.plot(x, [raw_sizes[file_name] for file_name in file_names_sorted], "-", marker='o', label="Raw")
for i in range(size):
    print("doing")
    plt.plot(x, [encoded_sizes[i][file_name] for file_name in file_names_sorted], "-", marker='o', label="Encoded" + str(i + 1))
plt.legend()

plt.show()