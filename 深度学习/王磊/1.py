import numpy as np
import matplotlib.pyplot as plt

# 参数设置
n_values = [10, 20, 40, 80, 160, 320]  # 问题规模
p_values = [1, 2, 4, 8, 16, 32, 64, 128]  # 处理器数量
alpha = 0.1  # 通信因子

# 初始化结果存储
results = []

# 计算加速比和效率
for n in n_values:
    for p in p_values:
        if p == 1:
            T_parallel = n  # 单处理器时没有通信开销
        else:
            T_parallel = n / p + alpha * (p - 1)
        S = n / T_parallel  # 加速比
        E = S / p  # 效率
        results.append((n, p, S, E))

# 数据分析和可视化
results = np.array(results)
for n in n_values:
    data = results[results[:, 0] == n]
    plt.plot(data[:, 1], data[:, 2], label=f"Speedup (n={n})")
plt.xscale("log", base=2)
plt.xlabel("处理器数量 (p)")
plt.ylabel("Speedup (S)")
plt.title("Speedup vs Processors for Different Problem Sizes")
plt.legend()
plt.show()

for n in n_values:
    data = results[results[:, 0] == n]
    plt.plot(data[:, 1], data[:, 3], label=f"Efficiency (n={n})")
plt.xscale("log", base=2)
plt.xlabel("Number of Processors (p)")
plt.ylabel("Efficiency (E)")
plt.title("Efficiency vs Processors for Different Problem Sizes")
plt.legend()
plt.show()
