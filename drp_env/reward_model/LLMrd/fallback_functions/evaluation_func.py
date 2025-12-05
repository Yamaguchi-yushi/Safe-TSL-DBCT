import numpy as np

FACTOR_NUMBER = 10

def evaluation_func(observation, eps=1e-6):
    """
    汎用 DRP 報酬関数（可変ノード数N, 可変エージェント数A, 可変エッジ数E）
    """
    import heapq

    B = observation.shape[0]

    # ---------------------------------------------------------
    # 1. 末尾から N, A, E, graph_diameter を取得
    # 観測末尾: [..., graph_diameter, N, A, E]
    # ---------------------------------------------------------
    E = int(observation[0, -1])
    A = int(observation[0, -2])
    N = int(observation[0, -3])
    graph_diameter = float(observation[0, -4])

    # ---------------------------------------------------------
    # 2. 可変長スライス用のサイズ計算
    # ---------------------------------------------------------
    # size_own_prev_curr = N * 2       # 前の位置 + 現在の位置
    # size_goal = N
    size_others_curr = (A - 1) * N * 2  # 他エージェントの前+現在位置
    size_nodes_flat = N * 2          # ノード座標 (x, y) * N
    size_edges_flat = E * 3          # エッジ情報 (from, to, distance) * E

    # ---------------------------------------------------------
    # 3. 観測スライス（先頭から順番に読み取る）
    # ---------------------------------------------------------
    idx = 0

    # 自エージェントの位置 (前の位置 + 現在の位置)
    own_prev = observation[:, idx : idx + N]
    idx += N
    own_curr = observation[:, idx : idx + N]
    idx += N

    # ゴール位置
    goal = observation[:, idx : idx + N]
    idx += N

    # 他エージェントの位置
    others_curr_raw = observation[:, idx : idx + size_others_curr]
    idx += size_others_curr

    # 衝突距離
    collision_distance = observation[:, idx : idx + 1]
    idx += 1

    # 衝突情報
    collision_info = observation[:, idx : idx + 2]
    idx += 2

    # 待機カウント
    wait_count = observation[:, idx : idx + 1]
    idx += 1

    # ノード座標
    nodes_flat = observation[:, idx : idx + size_nodes_flat]
    idx += size_nodes_flat

    # エッジ情報
    edges_flat = observation[0, idx : idx + size_edges_flat]
    idx += size_edges_flat

    # 残りはメタ情報（graph_diameter, N, A, E）
    # これらは既に先頭で取得済み

    # ---------------------------------------------------------
    # 4. edges 変換
    # ---------------------------------------------------------
    if E > 0 and len(edges_flat) >= E * 3:
        edges_arr = edges_flat.reshape(E, 3)
        edges = []
        for a, b, w in edges_arr:
            a_int = int(a)
            b_int = int(b)
            # 🔧 範囲チェック
            if 0 <= a_int < N and 0 <= b_int < N:
                edges.append((a_int, b_int, float(w)))
            else:
                print(f"⚠️ Invalid edge: ({a_int}, {b_int}) - N={N}")
    else:
        edges = []
        print(f"⚠️ No valid edges: E={E}, edges_flat length={len(edges_flat)}")

    # ---------------------------------------------------------
    # 5. ノード座標 shape = (B, N, 2)
    # ---------------------------------------------------------
    nodes = nodes_flat.reshape(B, N, 2)

    # ---------------------------------------------------------
    # 6. グラフ構築（辞書形式）
    # ---------------------------------------------------------
    graph = {i: [] for i in range(N)}
    for a, b, w in edges:
        graph[a].append((b, w))
        graph[b].append((a, w))

    # ---------------------------------------------------------
    # 7. Dijkstra
    # ---------------------------------------------------------
    def dijkstra(start, goal_node):
        if start == goal_node:
            return 0.0
        if start < 0 or start >= N or goal_node < 0 or goal_node >= N:
            return graph_diameter
        
        dist = {v: np.inf for v in range(N)}
        dist[start] = 0.0
        pq = [(0.0, start)]
        
        while pq:
            d, v = heapq.heappop(pq)
            if v == goal_node:
                return d
            if d > dist[v]:
                continue
            for nv, w in graph[v]:
                nd = d + w
                if nd < dist[nv]:
                    dist[nv] = nd
                    heapq.heappush(pq, (nd, nv))
        
        return graph_diameter if np.isinf(dist[goal_node]) else dist[goal_node]

    # ---------------------------------------------------------
    # 8. 部分距離推定
    # ---------------------------------------------------------
    def estimate_partial_distance(pos_vec, goal_node):
        nz = np.where(pos_vec > 1e-8)[0]

        if len(nz) == 1:
            return dijkstra(int(nz[0]), goal_node)

        if len(nz) == 2:
            i, j = int(nz[0]), int(nz[1])
            wi, wj = pos_vec[i], pos_vec[j]
            s = wi + wj + eps
            alpha = wj / s
            Di = dijkstra(i, goal_node)
            Dj = dijkstra(j, goal_node)
            if np.isinf(Di) or np.isinf(Dj):
                return min(Di, Dj) if not (np.isinf(Di) and np.isinf(Dj)) else graph_diameter
            return (1 - alpha) * Di + alpha * Dj

        if len(nz) == 0:
            return graph_diameter
        
        pivot = int(np.argmax(pos_vec))
        return dijkstra(pivot, goal_node)

    # ---------------------------------------------------------
    # 9. 座標系での位置
    # ---------------------------------------------------------
    agent_curr_pos = np.sum(own_curr[:, :, None] * nodes, axis=1)

    # 他エージェントの現在位置を抽出（前+現在から現在のみ）
    others_curr_pos_list = []
    for k in range(A - 1):
        start_idx = k * N * 2 + N  # 現在位置の開始インデックス
        end_idx = start_idx + N
        other_curr = others_curr_raw[:, start_idx:end_idx]
        others_curr_pos_list.append(
            np.sum(other_curr[:, :, None] * nodes, axis=1)
        )
    
    if len(others_curr_pos_list) > 0:
        others_curr_pos = np.stack(others_curr_pos_list, axis=1)
    else:
        others_curr_pos = np.zeros((B, 1, 2))

    # ---------------------------------------------------------
    # 10. グラフ距離計算
    # ---------------------------------------------------------
    dist_goal = np.zeros((B, 1))
    dist_goal_prev = np.zeros((B, 1))

    for b in range(B):
        goal_node = int(np.argmax(goal[b]))
        dist_goal[b, 0] = estimate_partial_distance(own_curr[b], goal_node)
        dist_goal_prev[b, 0] = estimate_partial_distance(own_prev[b], goal_node)

    dist_goal = np.where(np.isinf(dist_goal), graph_diameter, dist_goal)
    dist_goal_prev = np.where(np.isinf(dist_goal_prev), graph_diameter, dist_goal_prev)

    prog_goal = dist_goal_prev - dist_goal
    at_goal = (dist_goal < eps).astype(float)
    dist_goal_norm = dist_goal / (graph_diameter + eps)

    # ---------------------------------------------------------
    # 11. 分離距離
    # ---------------------------------------------------------
    if A > 1:
        sep = np.linalg.norm(agent_curr_pos[:, None, :] - others_curr_pos, axis=2)
        min_sep = np.min(sep, axis=1, keepdims=True)
        avg_sep = np.mean(sep, axis=1, keepdims=True)
    else:
        min_sep = np.full((B, 1), graph_diameter)
        avg_sep = np.full((B, 1), graph_diameter)

    min_sep_norm = min_sep / (graph_diameter + eps)
    avg_sep_norm = avg_sep / (graph_diameter + eps)

    safety_margin = np.clip(min_sep / (collision_distance + eps), 0, 100)
    collision_risk = (min_sep < collision_distance * 2).astype(float)

    wait_norm = wait_count.astype(float)
    in_collision = collision_info[:, 0:1]
    others_in_collision = collision_info[:, 1:2]

    # NaN対策
    prog_goal = np.nan_to_num(prog_goal, nan=0.0)
    dist_goal_norm = np.nan_to_num(dist_goal_norm, nan=1.0)
    min_sep_norm = np.nan_to_num(min_sep_norm, nan=0.0)
    avg_sep_norm = np.nan_to_num(avg_sep_norm, nan=0.0)
    safety_margin = np.nan_to_num(safety_margin, nan=0.0)

    # ---------------------------------------------------------
    # 12. 戻り値
    # ---------------------------------------------------------

    return [
        prog_goal,           # 1
        in_collision,        # 2
        others_in_collision, # 3
        wait_norm,           # 4
        dist_goal_norm,      # 5
        min_sep_norm,        # 6
        avg_sep_norm,        # 7
        safety_margin,       # 8
        collision_risk,      # 9
        at_goal              # 10
    ]



# import numpy as np

# FACTOR_NUMBER = 10

# def evaluation_func(observation, eps=1e-6):
#     """
#     汎用 DRP 報酬関数（可変ノード数N, 可変エージェント数A, 可変エッジ数E）

#     観測末尾:
#         [..., edges_flat(3E), graph_diameter, N, A, E]
#     """
#     import heapq

#     B = observation.shape[0]

#     # ---------------------------------------------------------
#     # 1. 末尾から graph_diameter, N, A, E を取得（全バッチ共通）
#     # ---------------------------------------------------------
#     graph_diameter = float(observation[0, -4])
#     N = int(observation[0, -3])
#     A = int(observation[0, -2])
#     E = int(observation[0, -1])

#     # ---------------------------------------------------------
#     # 2. 可変長スライス用のサイズ計算
#     # ---------------------------------------------------------
#     # size_own_prev    = N
#     # size_own_curr    = N
#     # size_goal        = N
#     # size_others_prev = (A - 1) * N
#     size_others_curr = (A - 1) * N
#     # size_collision_distance = 1
#     # size_collision_info     = 2
#     # size_wait_count         = 1
#     size_nodes_flat         = 2 * N
#     size_edges_flat         = 3 * E         # ★可変

#     # ---------------------------------------------------------
#     # 3. 観測スライス
#     # ---------------------------------------------------------
#     idx = 0

#     own_prev = observation[:, idx : idx + N]; idx += N
#     own_curr = observation[:, idx : idx + N]; idx += N
#     goal     = observation[:, idx : idx + N]; idx += N

#     # others_prev = observation[:, idx : idx + size_others_prev]; idx += size_others_prev
#     others_curr = observation[:, idx : idx + size_others_curr]; idx += size_others_curr

#     collision_distance = observation[:, idx : idx + 1]; idx += 1
#     collision_info     = observation[:, idx : idx + 2]; idx += 2
#     wait_count         = observation[:, idx : idx + 1]; idx += 1

#     nodes_flat = observation[:, idx : idx + size_nodes_flat]; idx += size_nodes_flat

#     edges_flat = observation[0, idx : idx + size_edges_flat]  # flat, バッチ共通
#     idx += size_edges_flat

#     # ---------------------------------------------------------
#     # 4. edges 変換（リスト形式）
#     # ---------------------------------------------------------
#     edges_arr = edges_flat.reshape(E, 3)
#     edges = [(int(a), int(b), float(w)) for a, b, w in edges_arr]

#     # ---------------------------------------------------------
#     # 5. ノード座標 shape = (B, N, 2)
#     # ---------------------------------------------------------
#     nodes = nodes_flat.reshape(B, N, 2)

#     # ---------------------------------------------------------
#     # 6. グラフ構築（辞書形式）
#     # ---------------------------------------------------------
#     graph = {i: [] for i in range(N)}
#     for a, b, w in edges:
#         graph[a].append((b, w))
#         graph[b].append((a, w))

#     # ---------------------------------------------------------
#     # 7. Dijkstra
#     # ---------------------------------------------------------
#     def dijkstra(start, goal_node):
#         if start == goal_node:
#             return 0.0
#         dist = {v: np.inf for v in range(N)}
#         dist[start] = 0.0
#         pq = [(0.0, start)]
#         while pq:
#             d, v = heapq.heappop(pq)
#             if v == goal_node:
#                 return d
#             if d > dist[v]:
#                 continue
#             for nv, w in graph[v]:
#                 nd = d + w
#                 if nd < dist[nv]:
#                     dist[nv] = nd
#                     heapq.heappush(pq, (nd, nv))
#         return dist[goal_node]

#     # ---------------------------------------------------------
#     # 8. エッジ上補間（α補間：i→j）
#     # ---------------------------------------------------------
#     def estimate_partial_distance(pos_vec, goal_node):
#         nz = np.where(pos_vec > 1e-8)[0]

#         # ノード上
#         if len(nz) == 1:
#             return dijkstra(int(nz[0]), goal_node)

#         # エッジ上
#         if len(nz) == 2:
#             i, j = int(nz[0]), int(nz[1])
#             wi, wj = pos_vec[i], pos_vec[j]

#             s = wi + wj + eps
#             alpha = wj / s  # j側の割合

#             Di = dijkstra(i, goal_node)
#             Dj = dijkstra(j, goal_node)

#             return (1 - alpha) * Di + alpha * Dj

#         # （通常起こらない）fallback
#         pivot = int(np.argmax(pos_vec))
#         return dijkstra(pivot, goal_node)

#     # ---------------------------------------------------------
#     # 9. 座標系での位置（エージェント・他エージェント）
#     # ---------------------------------------------------------
#     # agent_prev_pos = np.sum(own_prev[:, :, None] * nodes, axis=1)
#     agent_curr_pos = np.sum(own_curr[:, :, None] * nodes, axis=1)
#     # goal_pos       = np.sum(goal[:,     :, None] * nodes, axis=1)

#     # 他エージェントは A-1 個
#     others_curr_pos_list = []
#     for k in range(A - 1):
#         others_curr_pos_list.append(
#             np.sum(others_curr[:, k*N:(k+1)*N, None] * nodes, axis=1)
#         )
#     others_curr_pos = np.stack(others_curr_pos_list, axis=1)  # (B, A-1, 2)

#     # ---------------------------------------------------------
#     # 10. グラフ距離（ダイクストラ）＋ 正規化
#     # ---------------------------------------------------------
#     dist_goal      = np.zeros((B, 1))
#     dist_goal_prev = np.zeros((B, 1))

#     for b in range(B):
#         goal_node = int(np.argmax(goal[b]))
#         dist_goal[b, 0]      = estimate_partial_distance(own_curr[b], goal_node)
#         dist_goal_prev[b, 0] = estimate_partial_distance(own_prev[b], goal_node)

#     prog_goal = dist_goal_prev - dist_goal
#     at_goal = (dist_goal < eps).astype(float)

#     dist_goal_norm = dist_goal / (graph_diameter + eps)

#     # ---------------------------------------------------------
#     # 11. 分離距離（ユークリッド）＋ 正規化
#     # ---------------------------------------------------------
#     sep = np.linalg.norm(agent_curr_pos[:, None, :] - others_curr_pos, axis=2)

#     min_sep = np.min(sep, axis=1, keepdims=True)
#     avg_sep = np.mean(sep, axis=1, keepdims=True)

#     min_sep_norm = min_sep / (graph_diameter + eps)
#     avg_sep_norm = avg_sep / (graph_diameter + eps)

#     safety_margin = min_sep / (collision_distance + eps)
#     collision_risk = (min_sep < collision_distance * 2).astype(float)

#     wait_norm = wait_count.astype(float)
#     in_collision = collision_info[:, 0:1]
#     others_in_collision = collision_info[:, 1:2]

#     print(f"prog_goal: {prog_goal.flatten()}")
#     print(f"in_collision: {in_collision.flatten()}")
#     print(f"others_in_collision: {others_in_collision.flatten()}")
#     print(f"wait_norm: {wait_norm.flatten()}")
#     print(f"dist_goal_norm: {dist_goal_norm.flatten()}")
#     print(f"min_sep_norm: {min_sep_norm.flatten()}")
#     print(f"avg_sep_norm: {avg_sep_norm.flatten()}")
#     print(f"safety_margin: {safety_margin.flatten()}")
#     print(f"collision_risk: {collision_risk.flatten()}")
#     print(f"at_goal: {at_goal.flatten()}")

#     # ---------------------------------------------------------
#     # 12. 戻り値（従来と同じ順番で10要素）
#     # ---------------------------------------------------------
#     return [
#         prog_goal,           # 1
#         in_collision,        # 2
#         others_in_collision, # 3
#         wait_norm,           # 4
#         dist_goal_norm,      # 5
#         min_sep_norm,        # 6
#         avg_sep_norm,        # 7
#         safety_margin,       # 8
#         collision_risk,      # 9
#         at_goal              # 10
#     ]
