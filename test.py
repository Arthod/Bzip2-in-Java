import os
import matplotlib.pyplot as plt
import subprocess


cantrbry_path = "mycorpus/"
filenames = os.listdir(cantrbry_path)

# Plot raw file sizes
xx = [filename for filename in filenames]
yy = [os.path.getsize(cantrbry_path + filename) for filename in filenames]
plt.plot(xx, yy, "-", marker="o", label="raw")

def get_text_call(filename, trees_improve_iter=None, trees_count=None, block_size=None, RLE=None, SSF=None):
    X = ["java", "EncodeDecode", str(filename)]
    if (trees_improve_iter is not None):
        X = X + ["-it", str(trees_improve_iter)]
    if (trees_count is not None):
        X = X + ["-tc", str(trees_count)]
    if (block_size is not None):
        X = X + ["-bs", str(block_size)]
    if (RLE is not None):
        X = X + ["-rle"]
    if (SSF is not None):
        X = X + ["-ssf"]

    return X

# Call EncodeDecode
# String filename, int TREES_IMPROVE_ITER, int TREES_COUNT, int BLOCK_SIZE

print(*xx, sep=';')
    

# SSF
print("SSF")
yy = []
for f in filenames:
    filename = cantrbry_path + f
    
    cmdtext = get_text_call(filename, SSF="asd")
    result = subprocess.check_output(cmdtext, stderr=subprocess.STDOUT)
    yy.append(str(result).split(" "))

for i, a in enumerate(yy):
    print(i, *a, sep=";")

# Block size
print("Block size")
for i in [30, 50, 100, 200, 300, 500, 1000, 2000]:
    yy = []
    for f in filenames:
        filename = cantrbry_path + f
        
        cmdtext = get_text_call(filename, block_size=i)
        result = subprocess.check_output(cmdtext, stderr=subprocess.STDOUT)
        yy.append(int(result))

    print(i, *yy, sep=';')
    

# Trees improve iter
print("Trees improve iter")
for i in range(0, 16+1):
    yy = []
    for f in filenames:
        filename = cantrbry_path + f
        
        cmdtext = get_text_call(filename, trees_improve_iter=i)
        result = subprocess.check_output(cmdtext, stderr=subprocess.STDOUT)
        yy.append(int(result))

    print(i, *yy, sep=';')
    

# Tree count
print("Trees count")
for i in range(1, 16+1):
    yy = []
    for f in filenames:
        filename = cantrbry_path + f
        
        cmdtext = get_text_call(filename, trees_count=i)
        result = subprocess.check_output(cmdtext, stderr=subprocess.STDOUT)
        yy.append(int(result))

    print(i, *yy, sep=';')
    

# RLE?
print("RLE")
for rle in ["false", "true"]:
    yy = []
    for f in filenames:
        filename = cantrbry_path + f
        
        cmdtext = get_text_call(filename, RLE=rle)
        result = subprocess.check_output(cmdtext, stderr=subprocess.STDOUT)
        yy.append(int(result))

    print(rle, *yy, sep=';')

"""
for file_name in other_runs:
    plt.plot([i for i in range(len(other_runs[file_name]))], other_runs[file_name], "-", marker="o", label=file_name)

plt.legend()
plt.show()
"""