
#
# Encoding/Decoding duration measurement using different
# data structures and techniques on human files
#
import glob
from msilib.schema import File
import os
import subprocess
import matplotlib.pyplot as plt
import numpy as np


fig, (ax1, ax2) = plt.subplots(2, sharex=True)
types = {
    "al": "ArrayList",
    "alR": "ArrayListReversed",
    "ll": "LinkedList",
    "ar": "Array",
    "arR": "ArrayReversed",
}
retries = 3
file_paths = glob.glob("text_files/*.txt")
print(file_paths)

for type_id, type in types.items():
    print(type_id)
    X = []
    Y1 = []
    Y2 = []

    for file_path in file_paths:
        encoding_durations = []
        decoding_durations = []
        for _ in range(retries):
            p = subprocess.run(f"java Test test {type_id} -f {file_path}", stdout=subprocess.PIPE)
                    
            out_string = p.stdout.decode("utf-8")

            # Parse output
            encoding_duration, decoding_duration = map(int, out_string.split(" "))

            encoding_durations.append(encoding_duration)
            decoding_durations.append(decoding_duration)
        Y1.append(sum(encoding_durations) // retries)
        Y2.append(sum(decoding_durations) // retries)
        X.append(os.path.getsize(file_path))

    ax1.plot(np.array(X), np.array(Y1), label=type + " encoding", marker="o")
    ax2.plot(np.array(X), np.array(Y2), label=type + " decoding", marker="o")

plt.suptitle("Encoding/Decoding duration measurement using different\ndata structures and techniques on human files")

plt.xlabel("")
plt.ylabel("duration [ns]")


ax1.set_title("Encoding")
ax1.legend(loc='best')
ax1.grid()

ax2.set_title("Decoding")
ax2.legend(loc='best')
ax2.grid()

plt.show()