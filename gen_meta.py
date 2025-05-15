import random
import os
import numpy as np

N = 0
M = 0

dataset = "sift"

if dataset == "sift":
    N = 1000000
    M = 10000
elif dataset == "paper":
    N = 2029997
    M = 10000
elif dataset == "gist":
    N = 1000000
    M = 1000

iter = 6

mod_b = 0
mod_q = 2

cad = 0

output_file = f"testing_data/{dataset}/metadata_base_{iter}.txt"

if mod_b == 0:
    cad = 12
    num = [random.randint(1, cad) for _ in range(N)]
elif mod_b == 1:
    cad = 12
    values = np.arange(1, cad+1)  # 数值范围 1 到 12

    # 计算归一化的概率分布
    unnormalized_probs = 1 / values
    probs = unnormalized_probs / unnormalized_probs.sum()

    # 生成随机数
    num = np.random.choice(values, size=N, p=probs)
elif mod_b == 2:
    cad = 1000
    num = [random.randint(1, cad) for _ in range(N)]

with open(output_file, "w") as f:
    f.write(" ".join(map(str, num)))
print(f"Generated {output_file} with {N} random integers between 1 and {cad}.")


output_file = f"testing_data/{dataset}/metadata_query_{iter}.txt"

if mod_q == 0:
    num = [random.randint(1, cad) for _ in range(M)]
elif mod_q == 1:
    num = []
    for i in range(M):
        num.append(min(i // (M // cad) + 1, cad))
elif mod_q == 2:
    num = [1 for _ in range(N)]

with open(output_file, "w") as f:
    f.write(" ".join(map(str, num)))
print(f"Generated {output_file} with {M} random integers between 1 and {cad}.")
