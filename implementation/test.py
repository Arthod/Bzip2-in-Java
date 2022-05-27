import os
import matplotlib.pyplot as plt
import subprocess


cantrbry_path = "silesia/"
filenames = [filename for filename in os.listdir(cantrbry_path) if filename.lower().count("ignore") == 0]

outfile = open('results.txt', 'w')

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
        X = X + ["-rle", str(RLE)]
    if (SSF is not None):
        X = X + ["-ssf", str(SSF)]

    return X

# Call EncodeDecode
# String filename, int TREES_IMPROVE_ITER, int TREES_COUNT, int BLOCK_SIZE

print(*xx, sep=';')
print(*yy, sep=';')
outfile.write(";".join([str(a) for a in xx]) + "\n")
outfile.write(";".join([str(a) for a in yy]) + "\n")
outfile.flush()
os.fsync(outfile.fileno())


# SSF
print("SSF")
outfile.write("SSF" + "\n")
yy = []
for f in filenames:
    filename = cantrbry_path + f
    
    cmdtext = get_text_call(filename, SSF="true")
    result = subprocess.check_output(cmdtext, stderr=subprocess.STDOUT)
    yy.append(str(result.decode("utf-8")).split(" ")[:-1])

for i, a in enumerate(yy):
    print(i, *a, sep=";")
    outfile.write(";".join([str(b) for b in ([i] + a)]) + "\n")
outfile.flush()
os.fsync(outfile.fileno())
    

# Block size
print("Block size")
outfile.write("Block size" + "\n")
for i in [30, 50, 100, 200, 300, 500, 1000, 2000]:
    yy = []
    for f in filenames:
        filename = cantrbry_path + f
        
        cmdtext = get_text_call(filename, block_size=i)
        result = subprocess.check_output(cmdtext, stderr=subprocess.STDOUT)
        yy.append(int(result))

    print(i, *yy, sep=';')
    outfile.write(";".join([str(a) for a in ([i] + yy)]) + "\n")
    outfile.flush()
    os.fsync(outfile.fileno())

# Trees improve iter
print("Trees improve iter")
for i in range(1, 8):
    yy = []
    for f in filenames:
        filename = cantrbry_path + f
        
        cmdtext = get_text_call(filename, trees_improve_iter=i)
        result = subprocess.check_output(cmdtext, stderr=subprocess.STDOUT)
        yy.append(int(result))

    print(i, *yy, sep=';')
    outfile.write(";".join([str(a) for a in ([i] + yy)]) + "\n")
    outfile.flush()
    os.fsync(outfile.fileno())
    

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
    outfile.write(";".join([str(a) for a in ([i] + yy)]) + "\n")
    outfile.flush()
    os.fsync(outfile.fileno())
    

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
    outfile.write(";".join([str(a) for a in ([rle] + yy)]) + "\n")
    outfile.flush()
    os.fsync(outfile.fileno())
outfile.close()

"""
for file_name in other_runs:
    plt.plot([i for i in range(len(other_runs[file_name]))], other_runs[file_name], "-", marker="o", label=file_name)

plt.legend()
plt.show()
"""