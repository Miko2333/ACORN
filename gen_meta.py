import random
import os

dataset = "sift1M"
N = 1000000
# N = 2029997
M = 10000

iter = 0
cad = 12

output_file = f"testing_data/{dataset}/metadata_base_{iter}.txt"
# output_file = f"testing_data/paper/metadata_base_{iter}.txt"
num = [random.randint(1, 12) for _ in range(N)]
with open(output_file, "w") as f:
    f.write(" ".join(map(str, num)))
print(f"Generated {output_file} with {N} random integers between 1 and 12.")
output_file = f"testing_data/sift1M/metadata_query_{iter}.txt"
# output_file = f"testing_data/paper/metadata_query_{iter}.txt"
# num = [random.randint(1, 12) for _ in range(M)]
num = []
for i in range(M):
    num.append(min(i // (M // 12) + 1, 12))
with open(output_file, "w") as f:
    f.write(" ".join(map(str, num)))
print(f"Generated {output_file} with {M} random integers between 1 and 12.")

N = 2029997
M = 10000

for iter in range(5, 6):
    output_file = f"testing_data/sift1M/metadata_base_{iter}.txt"
    # output_file = f"testing_data/paper/metadata_base_{iter}.txt"
    num = [random.randint(1, 12) for _ in range(N)]
    with open(output_file, "w") as f:
        f.write(" ".join(map(str, num)))
    print(f"Generated {output_file} with {N} random integers between 1 and 12.")
    output_file = f"testing_data/sift1M/metadata_query_{iter}.txt"
    # output_file = f"testing_data/paper/metadata_query_{iter}.txt"
    # num = [random.randint(1, 12) for _ in range(M)]
    num = []
    for i in range(M):
        num.append(min(i // (M // 12) + 1, 12))
    with open(output_file, "w") as f:
        f.write(" ".join(map(str, num)))
    print(f"Generated {output_file} with {M} random integers between 1 and 12.")