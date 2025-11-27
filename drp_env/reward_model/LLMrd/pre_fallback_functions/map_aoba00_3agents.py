import numpy as np

FACTOR_NUMBER = 10

def evaluation_func(observation, eps=1e-6):
    import heapq

    B = observation.shape[0]
    N = 43   # ★ aoba00 ノード数

    # ---------------------------------------------------------
    # 1. 固定スライス（40→43に調整）
    # ---------------------------------------------------------
    own_prev           = observation[:,   0: 43]
    own_curr           = observation[:,  43: 86]
    goal               = observation[:,  86:129]
    other_1_prev       = observation[:, 129:172]
    other_1_curr       = observation[:, 172:215]
    other_2_prev       = observation[:, 215:258]
    other_2_curr       = observation[:, 258:301]

    collision_distance = observation[:, 301:302]
    collision_info     = observation[:, 302:304]
    wait_count         = observation[:, 304:305]

    nodes_flat         = observation[:, 305:305 + (2 * N)]  # 43×2=86 → 305:391

    # 多エージェント結合
    others_prev = np.concatenate([other_1_prev, other_2_prev], axis=1)  # (B, 86)
    others_curr = np.concatenate([other_1_curr, other_2_curr], axis=1)  # (B, 86)

    # ---------------------------------------------------------
    # 2. edges_flat + graph_diameter（スライス）
    # ---------------------------------------------------------
    tail = observation[0, 305 + 2*N:]   # 391 以降
    graph_diameter = float(tail[-1])
    edges_flat = tail[:-1]

    edges_arr = edges_flat.reshape(-1, 3)
    edges = [(int(a), int(b), float(w)) for a, b, w in edges_arr]

    # ---------------------------------------------------------
    # 3. ノード座標
    # ---------------------------------------------------------
    nodes = nodes_flat.reshape(B, N, 2)

    # ---------------------------------------------------------
    # 4. グラフ構築
    # ---------------------------------------------------------
    def build_graph(edge_list):
        g = {i: [] for i in range(N)}
        for a, b, w in edge_list:
            g[a].append((b, w))
            g[b].append((a, w))
        return g

    graph = build_graph(edges)

    # ---------------------------------------------------------
    # 5. Dijkstra
    # ---------------------------------------------------------
    def dijkstra(start, goal):
        if start == goal:
            return 0.0
        dist = {v: np.inf for v in range(N)}
        dist[start] = 0.0
        pq = [(0.0, start)]
        while pq:
            d, v = heapq.heappop(pq)
            if v == goal:
                return d
            if d > dist[v]:
                continue
            for nv, w in graph[v]:
                nd = d + w
                if nd < dist[nv]:
                    dist[nv] = nd
                    heapq.heappush(pq, (nd, nv))
        return dist[goal]

    # ---------------------------------------------------------
    # 6. ★ エッジ上距離補間（改善 α補間版）
    # ---------------------------------------------------------
    def estimate_partial_distance(pos_vec, goal_node):
        nz = np.where(pos_vec > 1e-8)[0]
        w  = pos_vec[nz]

        # --- ノード上 ---
        if len(nz) == 1:
            return dijkstra(int(nz[0]), goal_node)

        # --- エッジ上 ---
        if len(nz) == 2:
            i, j = int(nz[0]), int(nz[1])
            wi, wj = float(w[0]), float(w[1])

            s = wi + wj + eps
            wi_norm = wi / s
            wj_norm = wj / s

            alpha = wj_norm  # j側の割合

            Di = dijkstra(i, goal_node)
            Dj = dijkstra(j, goal_node)

            # 線形補間（位置補間と整合）
            return (1 - alpha) * Di + alpha * Dj

        # ---（通常は発生しない）fallback ---
        pivot = int(np.argmax(pos_vec))
        return dijkstra(pivot, goal_node)

    # ---------------------------------------------------------
    # 7. 座標（ユークリッド距離用）
    # ---------------------------------------------------------
    agent_curr_pos = np.sum(own_curr[:, :, None] * nodes, axis=1)

    others_curr_pos = np.stack([
        np.sum(others_curr[:,   0:43, None] * nodes, axis=1),
        np.sum(others_curr[:, 43:86, None] * nodes, axis=1)
    ], axis=1)

    # ---------------------------------------------------------
    # 8. ダイクストラ距離（正規化）
    # ---------------------------------------------------------
    dist_goal      = np.zeros((B, 1))
    dist_goal_prev = np.zeros((B, 1))

    for b in range(B):
        goal_node = int(np.argmax(goal[b]))
        dist_goal[b, 0]      = estimate_partial_distance(own_curr[b], goal_node)
        dist_goal_prev[b, 0] = estimate_partial_distance(own_prev[b], goal_node)

    prog_goal = dist_goal_prev - dist_goal
    at_goal = (dist_goal < eps).astype(float)

    # 正規化
    dist_goal_norm = dist_goal / (graph_diameter + eps)

    # ---------------------------------------------------------
    # 9. 分離距離（ユークリッド → 正規化）
    # ---------------------------------------------------------
    sep = np.linalg.norm(agent_curr_pos[:, None, :] - others_curr_pos, axis=2)

    min_sep = np.min(sep, axis=1, keepdims=True)
    avg_sep = np.mean(sep, axis=1, keepdims=True)

    min_sep_norm = min_sep / (graph_diameter + eps)
    avg_sep_norm = avg_sep / (graph_diameter + eps)

    safety_margin   = min_sep / (collision_distance + eps)
    collision_risk  = (min_sep < collision_distance * 2).astype(float)

    wait_norm       = wait_count.astype(float)
    in_collision    = collision_info[:, 0:1]
    others_in_collision = collision_info[:, 1:2]

    # ---------------------------------------------------------
    # 10. 出力
    # ---------------------------------------------------------
    return [
        prog_goal,           # 1
        in_collision,        # 2
        others_in_collision, # 3
        wait_norm,           # 4
        dist_goal_norm,      # 5 ★正規化
        min_sep_norm,        # 6 ★正規化
        avg_sep_norm,        # 7 ★正規化
        safety_margin,       # 8
        collision_risk,      # 9
        at_goal              # 10
    ]




# import numpy as np

# def evaluation_func(observation, eps=1e-6):
#     """
#     DRP（Drone Routing Problem）の観測データ（3エージェント、43ノード）から
#     12個の評価指標を計算して返す。
#     ダイクストラは使わず、ユークリッド距離に基づいて計算する。
#     """
#     B = observation.shape[0]

#     # ---- 1. 各部分を切り出し ----
#     own_prev = observation[:, 0:43]             # 自身の前ステップ位置 (43d)
#     own_curr = observation[:, 43:86]            # 自身の現在位置 (43d)
#     goal = observation[:, 86:129]               # ゴール位置 (43d)
#     other_agents_prev = observation[:, 129:215]   # 他2エージェントの前ステップ位置 (2×43d)
#     other_agents_curr = observation[:, 215:301]   # 他2エージェントの現在位置 (2×43d)
#     collision_distance = observation[:, 301:302]  # 衝突距離
#     collision_info = observation[:, 302:304]      # [自身衝突, 他者衝突]
#     wait_count = observation[:, 304:305]          # 連続待機カウント
#     nodes_flat = observation[:, 305:391]          # ノード座標 (43×2=86d)

#     # ---- 2. ノード座標を (B, 43, 2) に変形 ----
#     nodes = nodes_flat.reshape(B, 43, 2)

#     # ---- 3. 自身の連続位置を計算（ワンホット×座標の積和） ----
#     agent_prev_pos = np.sum(own_prev[:, :, None] * nodes, axis=1)  # (B,2)
#     agent_curr_pos = np.sum(own_curr[:, :, None] * nodes, axis=1)  # (B,2)
#     goal_pos = np.sum(goal[:, :, None] * nodes, axis=1)            # (B,2)

#     # ---- 4. 他エージェント位置計算（2エージェント×前後） ----
#     other_prev_pos = np.stack([
#         np.sum(other_agents_prev[:, 0:43, None] * nodes, axis=1),
#         np.sum(other_agents_prev[:, 43:86, None] * nodes, axis=1)
#     ], axis=1)  # (B,2,2)

#     other_curr_pos = np.stack([
#         np.sum(other_agents_curr[:, 0:43, None] * nodes, axis=1),
#         np.sum(other_agents_curr[:, 43:86, None] * nodes, axis=1)
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

#     # ---- 10. 結果をリストで返す（12個）----
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

# FACTOR_NUMBER = 12  # 評価指標の数（固定）
