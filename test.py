import os


def get_runs(file_path):
    f = open(file_path, "r")
    arr = []
    for x in f:
        arr.append(int(x.replace("\n", "")))
    return arr

cantrbry_path = "cantrbry/"
files = os.listdir(cantrbry_path)
file_paths = []

cantrbry_runs_path = "cantrbryBWTRuns2/"
other_runs = {}

other_runs_path = "cantrbryBWTRuns/"
cantrbry_runs = {}

for file_name in files:
    file_paths.append(cantrbry_path + file_name)
    other_runs[file_name] = get_runs(other_runs_path + file_name)
    try:
        cantrbry_runs[file_name] = get_runs(cantrbry_runs_path + file_name)
    except:
        pass


# Plot it

import matplotlib.pyplot as plt
for file_name in cantrbry_runs:
    plt.plot([i for i in range(len(cantrbry_runs[file_name]))], cantrbry_runs[file_name], "-", marker="o", label=file_name)

plt.legend()
#plt.yscale("log")
plt.show()


"""
for file_name in other_runs:
    plt.plot([i for i in range(len(other_runs[file_name]))], other_runs[file_name], "-", marker="o", label=file_name)

plt.legend()
plt.show()
"""