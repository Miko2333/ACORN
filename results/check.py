import re
from collections import defaultdict, Counter

def count_success_fail_x(filename):
    pattern = re.compile(r'^(success|fail|save) on (\d+)$')
    counters = defaultdict(Counter)  # counters['success'][x] 和 counters['fail'][x]

    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            match = pattern.match(line)
            if match:
                status = match.group(1)  # 'success' 或 'fail'
                x = int(match.group(2))
                counters[status][x] += 1

    # 打印统计结果
    for status in ['success', 'fail', 'save']:
        print(f"\n{status.upper()} results:")
        for x, count in sorted(counters[status].items()):
            print(f"  x = {x}: {count} time(s)")

# 使用示例
count_success_fail_x('CQ_res_sift_2.txt')