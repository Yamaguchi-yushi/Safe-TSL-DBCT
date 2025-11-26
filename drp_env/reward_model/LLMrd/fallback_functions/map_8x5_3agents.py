import numpy as np

FACTOR_NUMBER = 10

def evaluation_func(observation, eps=1e-6):
    import heapq

    B = observation.shape[0]
    N = 40

    # ---------------------------------------------------------
    # 1. 固定スライス
    # ---------------------------------------------------------
    own_prev           = observation[:,   0: 40]
    own_curr           = observation[:,  40: 80]
    goal               = observation[:,  80:120]
    # other_1_prev       = observation[:, 120:160]
    other_1_curr       = observation[:, 160:200]
    # other_2_prev       = observation[:, 200:240]
    other_2_curr       = observation[:, 240:280]
    collision_distance = observation[:, 280:281]
    collision_info     = observation[:, 281:283]
    wait_count         = observation[:, 283:284]
    nodes_flat         = observation[:, 284:364]

    # others_prev = np.concatenate([other_1_prev, other_2_prev], axis=1)  # (B, 80)
    others_curr = np.concatenate([other_1_curr, other_2_curr], axis=1)  # (B, 80)

    tail = observation[0, 364:]
    graph_diameter = float(tail[-1])
    edges_flat = tail[:-1]
    edges_arr = edges_flat.reshape(-1, 3)
    edges = [(int(a), int(b), float(w)) for a, b, w in edges_arr]

    # ---------------------------------------------------------
    # 2. ノード座標
    # ---------------------------------------------------------
    nodes = nodes_flat.reshape(B, N, 2)

    # ---------------------------------------------------------
    # 3. グラフ構築
    # ---------------------------------------------------------
    def build_graph(edge_list):
        g = {i: [] for i in range(N)}
        for a, b, w in edge_list:
            g[a].append((b, w))
            g[b].append((a, w))
        return g

    graph = build_graph(edges)

    # ---------------------------------------------------------
    # 4. Dijkstra
    # ---------------------------------------------------------
    def dijkstra(g, start, goal):
        if start == goal:
            return 0.0
        dist = {v: np.inf for v in g}
        dist[start] = 0.0
        pq = [(0.0, start)]
        while pq:
            d, v = heapq.heappop(pq)
            if v == goal:
                return d
            if d > dist[v]:
                continue
            for nv, w in g[v]:
                nd = d + w
                if nd < dist[nv]:
                    dist[nv] = nd
                    heapq.heappush(pq, (nd, nv))
        return dist[goal]

    # ---------------------------------------------------------
    # 5. エッジ上補間
    # ---------------------------------------------------------
    def estimate_partial_distance(pos_vec, goal_node):
        nz = np.where(pos_vec > 1e-8)[0]
        w  = pos_vec[nz]

        if len(nz) == 1:
            return dijkstra(graph, int(nz[0]), goal_node)

        if len(nz) == 2:
            i, j = int(nz[0]), int(nz[1])
            wi, wj = float(w[0]), float(w[1])
            
            s = wi + wj + eps
            # wi_norm = wi / s
            wj_norm = wj / s

            alpha = wj_norm

            Di = dijkstra(graph, i, goal_node)
            Dj = dijkstra(graph, j, goal_node)

            return (1 - alpha) * Di + alpha * Dj

        pivot = int(np.argmax(pos_vec))
        return dijkstra(graph, pivot, goal_node)

    # ---------------------------------------------------------
    # 6. 座標
    # ---------------------------------------------------------
    # agent_prev_pos = np.sum(own_prev[:, :, None] * nodes, axis=1)
    agent_curr_pos = np.sum(own_curr[:, :, None] * nodes, axis=1)
    # goal_pos       = np.sum(goal[:,     :, None] * nodes, axis=1)

    # others_prev_pos = np.stack([
    #    np.sum(others_prev[:,   0:40, None] * nodes, axis=1),
    #    np.sum(others_prev[:, 40:80, None] * nodes, axis=1)
    # ], axis=1)

    others_curr_pos = np.stack([
        np.sum(others_curr[:,   0:40, None] * nodes, axis=1),
        np.sum(others_curr[:, 40:80, None] * nodes, axis=1)
    ], axis=1)

    # ---------------------------------------------------------
    # 7. ダイクストラ距離
    # ---------------------------------------------------------
    dist_goal      = np.zeros((B, 1))
    dist_goal_prev = np.zeros((B, 1))

    for b in range(B):
        goal_node = int(np.argmax(goal[b]))
        dist_goal[b, 0]      = estimate_partial_distance(own_curr[b], goal_node)
        dist_goal_prev[b, 0] = estimate_partial_distance(own_prev[b], goal_node)

    prog_goal = dist_goal_prev - dist_goal
    at_goal = (dist_goal < eps).astype(float)
    dist_goal_norm = dist_goal / (graph_diameter + eps)

    # ---------------------------------------------------------
    # 9. 分離距離（デバッグ付き）
    # ---------------------------------------------------------
    sep = np.linalg.norm(agent_curr_pos[:, None, :] - others_curr_pos, axis=2)

    min_sep = np.min(sep, axis=1, keepdims=True)
    avg_sep = np.mean(sep, axis=1, keepdims=True)

    min_sep_norm = min_sep / (graph_diameter + eps)
    avg_sep_norm = avg_sep / (graph_diameter + eps)

    # ---------------------------------------------------------
    # 10. 衝突関連
    # ---------------------------------------------------------
    safety_margin   = min_sep / (collision_distance + eps)
    collision_risk  = (min_sep < collision_distance * 2).astype(float)

    wait_norm       = wait_count.astype(float)
    in_collision    = collision_info[:, 0:1]
    others_in_collision = collision_info[:, 1:2]

    # ---------------------------------------------------------
    # 11. 出力
    # ---------------------------------------------------------

    return [
        prog_goal,
        in_collision,
        others_in_collision,
        wait_norm,
        dist_goal_norm,
        min_sep_norm,
        avg_sep_norm,
        safety_margin,
        collision_risk,
        at_goal
    ]


# import numpy as np

# def evaluation_func(observation, eps=1e-6):
#     """
#     DRP（Drone Routing Problem）の観測データ（3エージェント、8x5ノード）から
#     12個の評価指標を計算して返す。
#     """
#     B = observation.shape[0]

#     # ---- 1. 各部分を切り出し ----
#     own_prev = observation[:, 0:40]           # 自分の前ステップ位置 (40d)
#     own_curr = observation[:, 40:80]          # 自分の現在位置 (40d)
#     goal = observation[:, 80:120]             # ゴール位置 (40d)
#     other_agents_prev = observation[:, 120:200]  # 他2エージェントの前ステップ位置 (2×40d)
#     other_agents_curr = observation[:, 200:280]  # 他2エージェントの現在位置 (2×40d)
#     collision_distance = observation[:, 280:281]  # 衝突距離
#     collision_info = observation[:, 281:283]     # [自分衝突, 他者衝突]
#     wait_count = observation[:, 283:284]        # 連続待機カウント
#     nodes_flat = observation[:, 284:364]       # 40ノード×2座標 = 80d

#     # ---- 2. ノード座標を (B, 40, 2) に変形 ----
#     nodes = nodes_flat.reshape(B, 40, 2)

#     # ---- 3. 自エージェント位置計算（ワンホット×ノード座標の積和） ----
#     agent_prev_pos = np.sum(own_prev[:, :, None] * nodes, axis=1)  # (B,2)
#     agent_curr_pos = np.sum(own_curr[:, :, None] * nodes, axis=1)  # (B,2)
#     goal_pos = np.sum(goal[:, :, None] * nodes, axis=1)            # (B,2)

#     # ---- 4. 他エージェント位置計算（2エージェント×前後） ----
#     other_prev_pos = np.stack([
#         np.sum(other_agents_prev[:, 0:40, None] * nodes, axis=1),
#         np.sum(other_agents_prev[:, 40:80, None] * nodes, axis=1)
#     ], axis=1)  # (B,2,2)

#     other_curr_pos = np.stack([
#         np.sum(other_agents_curr[:, 0:40, None] * nodes, axis=1),
#         np.sum(other_agents_curr[:, 40:80, None] * nodes, axis=1)
#     ], axis=1)  # (B,2,2)

#     # ---- 5. ゴール距離と進捗 ----
#     dist_goal = np.linalg.norm(agent_curr_pos - goal_pos, axis=1, keepdims=True)
#     dist_goal_prev = np.linalg.norm(agent_prev_pos - goal_pos, axis=1, keepdims=True)
#     prog_goal = dist_goal_prev - dist_goal
#     at_goal = (dist_goal < eps).astype(float)

#     # ---- 6. 他エージェントとの分離距離 ----
#     sep = np.linalg.norm(agent_curr_pos[:, None, :] - other_curr_pos, axis=2)  # (B,2)
#     min_sep = np.min(sep, axis=1, keepdims=True)
#     avg_sep = np.mean(sep, axis=1, keepdims=True)
#     sep_var = np.var(sep, axis=1, keepdims=True)

#     # ---- 7. 安全余裕・衝突指標 ----
#     safety_margin = min_sep / (collision_distance + eps)
#     collision_count = (sep < collision_distance).sum(axis=1, keepdims=True).astype(float)
#     collision_risk = (min_sep < collision_distance).astype(float)

#     # ---- 8. 待機カウント ----
#     wait_norm = wait_count.astype(float)

#     # ---- 9. 提供衝突情報 ----
#     in_collision = collision_info[:, 0:1]
#     others_in_collision = collision_info[:, 1:2]

#     print(f"   prog_goal: {prog_goal.flatten()}")
#     print(f"   in_collision: {in_collision.flatten()}")
#     print(f"   others_in_collision: {others_in_collision.flatten()}")
#     print(f"   wait_norm: {wait_norm.flatten()}")
#     print(f"   dist_goal: {dist_goal.flatten()}")
#     print(f"   min_sep: {min_sep.flatten()}")
#     print(f"   avg_sep: {avg_sep.flatten()}")
#     print(f"   sep_var: {sep_var.flatten()}")
#     print(f"   safety_margin: {safety_margin.flatten()}")
#     print(f"   collision_count: {collision_count.flatten()}")
#     print(f"   collision_risk: {collision_risk.flatten()}")
#     print(f"   at_goal: {at_goal.flatten()}")


#     # ---- 10. 結果をリストで返す ----
#     return [
#         prog_goal,
#         in_collision,
#         others_in_collision,
#         wait_norm,
#         dist_goal,
#         min_sep,
#         avg_sep,
#         sep_var,
#         safety_margin,
#         collision_count,
#         collision_risk,
#         at_goal
#     ]

# FACTOR_NUMBER = 12