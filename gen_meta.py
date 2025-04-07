import random
import os

N = 1000000
M = 10000

for iter in range(3):
    output_file = f"testing_data/sift1M/metadata_base_{iter}.txt"
    num = [random.randint(1, 12) for _ in range(N)]
    with open(output_file, "w") as f:
        f.write(" ".join(map(str, num)))
    print(f"Generated {output_file} with {N} random integers between 1 and 12.")
    output_file = f"testing_data/sift1M/metadata_query_{iter}.txt"
    num = [random.randint(1, 12) for _ in range(M)]
    with open(output_file, "w") as f:
        f.write(" ".join(map(str, num)))
    print(f"Generated {output_file} with {M} random integers between 1 and 12.")