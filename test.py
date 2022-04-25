import os
import matplotlib.pyplot as plt
import subprocess


cantrbry_path = "cantrbry/"
filenames = os.listdir(cantrbry_path)

# Plot raw file sizes
xx = [filename for filename in filenames]
yy = [os.path.getsize(cantrbry_path + filename) for filename in filenames]
plt.plot(xx, yy, "-", marker="o", label="raw")

def get_text_call(filename, trees_improve_iter=3, trees_count=6, block_size=50):
    return ["java", "EncodeDecode", str(filename), str(trees_improve_iter), str(trees_count), str(block_size)]

# Call EncodeDecode
# String filename, int TREES_IMPROVE_ITER, int TREES_COUNT, int BLOCK_SIZE

print(xx)
# Block size
print("Block size")
for i in [30, 50, 100, 200, 300, 500, 1000, 2000]:
    yy = []
    for f in filenames:
        filename = cantrbry_path + f
        
        cmdtext = get_text_call(filename, block_size=i)
        result = subprocess.check_output(cmdtext, stderr=subprocess.STDOUT)
        yy.append(int(result))

    print(i, *yy, sep=' ')
    plt.plot(xx, yy, "-", marker="o", label=f"Block size = {i}")

plt.legend()
plt.ticklabel_format(style='plain', axis='y')
plt.show()

# Trees improve iter
print("Trees improve iter")
for i in range(1, 8+1):
    yy = []
    for f in filenames:
        filename = cantrbry_path + f
        
        cmdtext = get_text_call(filename, trees_improve_iter=i)
        result = subprocess.check_output(cmdtext, stderr=subprocess.STDOUT)
        yy.append(int(result))

    print(i, *yy, sep=' ')
    plt.plot(xx, yy, "-", marker="o", label=f"trees improve iter = {i}")

plt.legend()
plt.ticklabel_format(style='plain', axis='y')
plt.show()

# Tree count
print("Trees count")
for i in range(1, 8+1):
    yy = []
    for f in filenames:
        filename = cantrbry_path + f
        
        cmdtext = get_text_call(filename, trees_count=i)
        result = subprocess.check_output(cmdtext, stderr=subprocess.STDOUT)
        yy.append(int(result))

    print(i, *yy, sep=' ')
    plt.plot(xx, yy, "-", marker="o", label=f"trees_count = {i}")
    

plt.legend()
plt.ticklabel_format(style='plain', axis='y')
plt.show()


"""
for file_name in other_runs:
    plt.plot([i for i in range(len(other_runs[file_name]))], other_runs[file_name], "-", marker="o", label=file_name)

plt.legend()
plt.show()
"""