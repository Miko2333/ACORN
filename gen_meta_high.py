import random
import os
import numpy as np

N = 0
M = 0

dataset = "paper"

if dataset == "sift":
    N = 1000000
    M = 10000
elif dataset == "paper":
    N = 2029997
    M = 10000
elif dataset == "gist":
    N = 1000000
    M = 1000

iter = 4

mod_b = 0
mod_q = 0

cad = 0

output_file = f"testing_data/{dataset}/metadata_base_{iter}.txt"

if mod_b == 0:
    cad = N // 100
    num = [random.randint(1, cad) for _ in range(N)]

with open(output_file, "w") as f:
    f.write(" ".join(map(str, num)))
print(f"Generated {output_file} with {N} random integers between 1 and {cad}.")


output_file = f"testing_data/{dataset}/metadata_query_{iter}.txt"

num = []

if mod_q == 0:
    for i in range(M):
        len = random.randint(cad // 30, cad // 5)
        L = random.randint(1, cad-len+1)
        R = L + len - 1
        num.append((L, R))

with open(output_file, "w") as f:
    for tuple in num:
        f.write(f"{tuple[0]} {tuple[1]}\n")
print(f"Generated {output_file} with {M} random integers between 1 and {cad}.")
