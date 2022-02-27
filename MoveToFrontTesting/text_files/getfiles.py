import subprocess

for i in range(10000):
    try:
        p = subprocess.run(f"wget https://www.gutenberg.org/ebooks/{i}.txt.utf-8")
    except:
        print(f"failed: {i}")

