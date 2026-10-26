"""Tree topology, growth snapshots, and radii; no L-Py runtime required."""
import numpy as np

def apply_pipe_model(nodes, leaf_radius=0.003, exponent=2.5):
    """
    用 pipe model（da Vinci 法则）重新计算节点半径。

    parent.radius^n = sum(child.radius^n for c in children)

    Args:
        nodes: extract_nodes_from_lstring 的输出（会被原地修改）
        leaf_radius: 叶端（无子节点）的基础半径，越大整棵树越粗
        exponent: pipe model 指数
                  2.0 → 严格截面守恒（最瘦版本，da Vinci 原版）
                  2.49 → Murray's law（流体优化，常用于血管）
                  2.5~3.0 → 实测树木经验值，越大越粗
    """
    n = len(nodes)

    # 构建 children 列表
    children = [[] for _ in range(n)]
    for i, node in enumerate(nodes):
        p = node["parent_idx"]
        if p != -1:
            children[p].append(i)

    # 后序遍历（迭代版，避免递归爆栈）
    visited = [False] * n
    post_order = []
    for start in range(n):
        if visited[start]:
            continue
        stack = [(start, False)]
        while stack:
            i, expanded = stack.pop()
            if expanded:
                post_order.append(i)
                visited[i] = True
            else:
                stack.append((i, True))
                for c in children[i]:
                    if not visited[c]:
                        stack.append((c, False))

    # 自下而上计算 radius
    new_r = [leaf_radius] * n
    for i in post_order:
        if children[i]:
            total = sum(new_r[c] ** exponent for c in children[i])
            new_r[i] = total ** (1.0 / exponent)

    # 写回
    for i, node in enumerate(nodes):
        node["radius"] = float(new_r[i])

    return nodes


def smooth_radius_along_chains(nodes):
    """
    Pipe model 只在分叉点改变半径，导致同一 chain 上节点等粗。
    这个函数沿每条 chain 平滑插值半径。

    具体做法：对每个节点，找它"最近的下游分叉点"（或叶子）和
    "最近的上游分叉点"（或根），沿这段 chain 线性插值半径。
    """
    n = len(nodes)
    children = [[] for _ in range(n)]
    for i, node in enumerate(nodes):
        p = node["parent_idx"]
        if p != -1:
            children[p].append(i)

    # 标记节点类型：分叉点（有 >=2 个孩子）、叶子（无孩子）、链节点（1 个孩子）
    is_junction = [len(children[i]) >= 2 for i in range(n)]
    is_leaf = [len(children[i]) == 0 for i in range(n)]
    is_root = [nodes[i]["parent_idx"] == -1 for i in range(n)]

    # 关键节点 = 分叉点 + 叶子 + 根
    keep_radius = [is_junction[i] or is_leaf[i] or is_root[i] for i in range(n)]

    # 对每个非关键节点，沿 chain 往下找最近的关键节点（end），
    # 往上找最近的关键节点（start），按距离线性插值

    # 先备份关键节点的半径
    key_radius = {i: nodes[i]["radius"] for i in range(n) if keep_radius[i]}

    # 对每个链节点（非关键节点），向上和向下走 chain 找端点
    new_radius = [None] * n
    for i in range(n):
        if keep_radius[i]:
            new_radius[i] = nodes[i]["radius"]
            continue

        # 向上找上游关键节点
        up_node = nodes[i]["parent_idx"]
        up_dist = 0
        cur = i
        while up_node != -1 and not keep_radius[up_node]:
            up_dist += 1
            cur = up_node
            up_node = nodes[up_node]["parent_idx"]
        # 此时 up_node 是上游关键节点（或 -1，到根了还没找到，理论上不会发生）
        if up_node == -1:
            new_radius[i] = nodes[i]["radius"]
            continue
        up_dist += 1

        # 向下找下游关键节点
        # 链节点只有 1 个孩子，所以可以沿 children[i][0] 一直走
        down_node = children[i][0] if children[i] else None
        down_dist = 1
        while down_node is not None and not keep_radius[down_node]:
            if not children[down_node]:
                break
            down_node = children[down_node][0]
            down_dist += 1

        if down_node is None or not keep_radius[down_node]:
            new_radius[i] = nodes[i]["radius"]
            continue

        # 线性插值
        r_up = key_radius[up_node]
        r_down = key_radius[down_node]
        total = up_dist + down_dist
        t = up_dist / total
        new_radius[i] = r_up * (1 - t) + r_down * t

    # 写回
    for i in range(n):
        if new_radius[i] is not None:
            nodes[i]["radius"] = float(new_radius[i])
    return nodes


def snapshot_at_time(nodes, t, pipe_exponent=2.5, leaf_radius=None,
                     smooth_chains=True):
    """
    取出 birth_step <= t 的节点子集，并按 pipe model 重算半径。
    模拟"时间 t 时这棵树长什么样"。

    Args:
        nodes: 完整树的 nodes（每个节点含 xyz, start_xyz, radius, birth_step, parent_idx）
        t: 时间步（整数或浮点都行，按 birth_step <= t 截断）
        pipe_exponent: pipe model 指数，建议从 tree_data['params']['PIPE_EXPONENT'] 取
        leaf_radius: 叶端基础半径，None 则从原 nodes 推断
        smooth_chains: 是否沿 chain 平滑插值（参考你已有的 smooth_radius_along_chains）

    Returns:
        新的 nodes 列表（独立副本，parent_idx 已重新映射）。
        如果 t 太小连根都还没出生，返回空 list。
    """
    if not nodes:
        return []

    # 1. 选出 birth_step <= t 的节点（root 用 birth_step=0 兜底，避免 birth_step=-1 的 F 节点被滤掉）
    keep = []
    for i, nd in enumerate(nodes):
        b = nd.get("birth_step", 0)
        if b < 0:                      # F 节点（base / interpretation 产生的）继承父节点的 birth_step
            p = nd["parent_idx"]
            b = nodes[p].get("birth_step", 0) if p != -1 else 0
        if b <= t:
            keep.append(i)

    if not keep:
        return []

    # 2. 重新映射 parent_idx 到新索引
    old2new = {old: new for new, old in enumerate(keep)}
    new_nodes = []
    for old in keep:
        nd = dict(nodes[old])          # 浅拷贝即可，xyz 等 np.array 不会被改
        nd["xyz"]        = np.array(nd["xyz"], dtype=float).copy()
        nd["start_xyz"]  = np.array(nd["start_xyz"], dtype=float).copy()
        p = nd["parent_idx"]
        # 父若被截掉（理论上不会，因为父的 birth_step 必 <= 子的；但 root 的 parent=-1 要保留）
        nd["parent_idx"] = old2new.get(p, -1) if p != -1 else -1
        new_nodes.append(nd)

    # 3. 估计 leaf_radius：取原 nodes 里所有"无子节点"的半径中位数
    if leaf_radius is None:
        n_full = len(nodes)
        had_child = [False] * n_full
        for nd in nodes:
            p = nd["parent_idx"]
            if p != -1:
                had_child[p] = True
        leaf_rs = [nodes[i]["radius"] for i in range(n_full) if not had_child[i]]
        leaf_radius = float(np.median(leaf_rs)) if leaf_rs else 0.003

    # 4. 用你已有的 pipe model（直接调，不重写）
    new_nodes = apply_pipe_model(new_nodes, leaf_radius=leaf_radius, exponent=pipe_exponent)
    if smooth_chains:
        new_nodes = smooth_radius_along_chains(new_nodes)


    return new_nodes


