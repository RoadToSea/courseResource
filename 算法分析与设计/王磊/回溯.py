# 定义物品的重量和价值
weights = [10, 7, 8, 4]  # 物品的重量
values = [100, 63, 56, 12]  # 物品的价值
W = 16  # 背包的最大承重


# 回溯法求解0-1背包问题
def knapsack_backtrack(weights, values, W):
    n = len(weights)
    best_value = 0

    def backtrack(i, current_weight, current_value):
        nonlocal best_value
        # 如果所有物品都处理完毕，更新最优解
        if i == n:
            if current_value > best_value:
                best_value = current_value
            return
        # 选择第i个物品
        if current_weight + weights[i] <= W:
            backtrack(i + 1, current_weight + weights[i], current_value + values[i])
        # 不选择第i个物品
        backtrack(i + 1, current_weight, current_value)

    # 从第0个物品开始回溯
    backtrack(0, 0, 0)
    return best_value


# 求解并输出结果
max_value = knapsack_backtrack(weights, values, W)
print(f"回溯算法——最大价值为: {max_value}")
