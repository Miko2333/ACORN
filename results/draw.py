import os
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# 参数
algorithms = ["HNSW", "ACORN", "CQ"]
datasets = ["paper"]
i_values = [15, 20, 25, 30, 35, 40, 45, 50]
dataset_queries = {"sift": 10000, "paper": 10000, "gist": 1000}

# 正则表达式
patterns = {
    "recall": re.compile(r"Recall@10\s*=\s*([\d.]+)"),
    "time": re.compile(r"Search done:\s*\[([\d.]+)\s*s\]"),
    "dpq": re.compile(r"average distance computations per query:\s*([\d.]+)")
}

# 结果结构
results = defaultdict(lambda: defaultdict(lambda: {"recall": [], "qps": [], "dpq": []}))

# 数据读取
for algo in algorithms:
    for data in datasets:
        num_queries = dataset_queries[data]
        for i in i_values:
            folder = f"efs={i}"
            recalls, qtimes, dpqs = [], [], []

            for k in range(4, 5):
                filename = os.path.join(folder, f"{algo}_res_{data}_{k}.txt")
                if not os.path.exists(filename):
                    print(f"File not found: {filename}")
                    continue

                with open(filename, "r") as f:
                    content = f.read()
                    recall_match = patterns["recall"].search(content)
                    time_match = patterns["time"].search(content)
                    dpq_match = patterns["dpq"].search(content)

                    if recall_match and time_match and dpq_match:
                        recall = float(recall_match.group(1))
                        seconds = float(time_match.group(1))
                        dpq = float(dpq_match.group(1))
                        # print(recall, seconds, dpq)

                        recalls.append(recall)
                        qtimes.append(seconds)
                        dpqs.append(dpq)
                    else:
                        print(f"Missing data in {filename}")

            if recalls and qtimes and dpqs:
                results[data][algo]["recall"].append(sum(recalls) / len(recalls))
                results[data][algo]["qps"].append(np.log10(dataset_queries[data]/(sum(qtimes) / len(qtimes))))
                results[data][algo]["dpq"].append(sum(dpqs) / len(dpqs))

# 拟合并绘图（翻转坐标：X = QPS/DPQ, Y = Recall）

# 颜色定义
colors = {"HNSW": "red", "ACORN": "blue", "CQ": "green"}

for data in datasets:
    # QPS vs Recall（log 横轴，不再拟合）
    plt.figure(figsize=(10, 6))
    for algo in algorithms:
        x = results[data][algo]["qps"]
        y = results[data][algo]["recall"]
        valid = [(xi, yi) for xi, yi in zip(x, y) if xi > 0]
        if valid:
            x_valid, y_valid = zip(*sorted(valid))  # 确保按x排序
            plt.plot(x_valid, y_valid, marker='o', label=algo, color=colors[algo])
    # plt.xscale("log")
    if data == "gist":
        plt.ylim(0.3, 1)
    else:
        plt.ylim(0.5, 1)
    plt.xlabel("QPS (log scale)")
    plt.ylabel("Recall@10")
    # plt.title(f"QPS vs Recall on {data}")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f"qps_vs_recall_{data}_line.png")
    plt.show()

    # DPQ vs Recall（线性横轴）
    plt.figure(figsize=(10, 6))
    for algo in algorithms:
        x = results[data][algo]["dpq"]
        y = results[data][algo]["recall"]
        valid = [(xi, yi) for xi, yi in zip(x, y) if xi > 0]
        if valid:
            x_valid, y_valid = zip(*sorted(valid))
            plt.plot(x_valid, y_valid, marker='o', label=algo, color=colors[algo])
    if data == "gist":
        plt.ylim(0.3, 1)
    else:
        plt.ylim(0.5, 1)
    plt.xlabel("DPQ")
    plt.ylabel("Recall@10")
    # plt.title(f"DPQ vs Recall on {data}")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f"dpq_vs_recall_{data}_line.png")
    plt.show()

# fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 12), sharex=True)
# fig.suptitle("QPS vs Recall", fontsize=16)

# for idx, data in enumerate(datasets):
#     ax = axes[idx]
#     for algo in algorithms:
#         x = results[data][algo]["qps"]
#         y = results[data][algo]["recall"]
#         valid = [(xi, yi) for xi, yi in zip(x, y) if xi > 0]
#         if valid:
#             x_valid, y_valid = zip(*sorted(valid))
#             ax.plot(x_valid, y_valid, marker='o', label=algo, color=colors[algo])
#     ax.set_title(f"{data}")
#     ax.set_ylim(0.3 if data == "gist" else 0.5, 1)
#     ax.set_ylabel("Recall@10")
#     ax.grid(True, linestyle="--", linewidth=0.5)
#     ax.legend()

# axes[-1].set_xlabel("QPS (log scale)")
# plt.tight_layout(rect=[0, 0, 1, 0.96])
# plt.savefig("combined_qps_vs_recall.png")
# plt.show()

# fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 12), sharex=True)
# fig.suptitle("DPQ vs Recall", fontsize=16)

# for idx, data in enumerate(datasets):
#     ax = axes[idx]
#     for algo in algorithms:
#         x = results[data][algo]["dpq"]
#         y = results[data][algo]["recall"]
#         valid = [(xi, yi) for xi, yi in zip(x, y) if xi > 0]
#         if valid:
#             x_valid, y_valid = zip(*sorted(valid))
#             ax.plot(x_valid, y_valid, marker='o', label=algo, color=colors[algo])
#     ax.set_title(f"{data}")
#     ax.set_ylim(0.3 if data == "gist" else 0.5, 1)
#     ax.set_ylabel("Recall@10")
#     ax.grid(True, linestyle="--", linewidth=0.5)
#     ax.legend()

# axes[-1].set_xlabel("DPQ")
# plt.tight_layout(rect=[0, 0, 1, 0.96])
# plt.savefig("combined_dpq_vs_recall.png")
# plt.show()
