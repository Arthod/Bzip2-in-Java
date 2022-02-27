
#
# Encoding/Decoding duration measurement using different
# data structures and techniques on randomly generated files 
# of size n in bytes
#

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


X = [int(pow(i, 5)) for i in range(1, 12)]#[pow(10, i) for i in range(1, 4)]    # 27
for type_id, type in types.items():
    print(type_id)
    Y1 = []
    Y2 = []

    for i in range(len(X)):
        n = X[i]
        print(i, n)

        encoding_durations = []
        decoding_durations = []

        for _ in range(retries):
            p = subprocess.run(f"java Test test {type_id} -n {n}", stdout=subprocess.PIPE)
                    
            out_string = p.stdout.decode("utf-8")

            # Parse output
            encoding_duration, decoding_duration = map(int, out_string.split(" "))

            encoding_durations.append(encoding_duration)
            decoding_durations.append(decoding_duration)
        Y1.append(sum(encoding_durations) // retries)
        Y2.append(sum(decoding_durations) // retries)

    ax1.plot(np.array(X), np.array(Y1), label=type + " encoding", marker="o")
    ax2.plot(np.array(X), np.array(Y2), label=type + " decoding", marker="o")

plt.suptitle("Encoding/Decoding duration measurement using different\ndata structures and techniques on randomly generated files\nof size n in bytes")

plt.xlabel("file size n [bytes]")
plt.ylabel("duration [ns]")


ax1.set_title("Encoding")
ax1.legend(loc='best')
ax1.grid()

ax2.set_title("Decoding")
ax2.legend(loc='best')
ax2.grid()

plt.show()