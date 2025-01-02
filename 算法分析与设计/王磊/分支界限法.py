import heapq


class Node:
    def __init__(self, level, profit, weight, bound):
        self.level = level  # 当前物品的索引
        self.profit = profit  # 当前背包的总价值
        self.weight = weight  # 当前背包的总重量
        self.bound = bound  # 当前节点的上界

    def __lt__(self, other):
        return self.bound > other.bound  # 优先队列需要按照上界大小排序


def knapsack(n, weights, profits, capacity):
    def bound(node, n, weights, profits, capacity):
        if node.weight >= capacity:
            return 0
        profit_bound = node.profit
        j = node.level + 1
        total_weight = node.weight

        while j < n and total_weight + weights[j] <= capacity:
            total_weight += weights[j]
            profit_bound += profits[j]
            j += 1

        if j < n:
            profit_bound += (capacity - total_weight) * (profits[j] / weights[j])

        return profit_bound

    # 初始化
    max_profit = 0
    pq = []

    # 从第一个节点开始
    v = Node(-1, 0, 0, 0)
    v.bound = bound(v, n, weights, profits, capacity)
    heapq.heappush(pq, v)

    while pq:
        v = heapq.heappop(pq)
        if v.bound > max_profit:
            u = Node(v.level + 1, v.profit + profits[v.level + 1], v.weight + weights[v.level + 1], 0)
            if u.weight <= capacity and u.profit > max_profit:
                max_profit = u.profit
            u.bound = bound(u, n, weights, profits, capacity)
            if u.bound > max_profit:
                heapq.heappush(pq, u)
            u = Node(v.level + 1, v.profit, v.weight, 0)
            u.bound = bound(u, n, weights, profits, capacity)
            if u.bound > max_profit:
                heapq.heappush(pq, u)

    return max_profit


# 测试数据
n = 4
weights = [10, 7, 8, 4]
profits = [100, 63, 56, 12]
capacity = 16

max_profit = knapsack(n, weights, profits, capacity)
print(f"分支定界算法——最大价值: {max_profit}")
