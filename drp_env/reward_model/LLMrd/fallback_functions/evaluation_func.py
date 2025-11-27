import numpy as np

FACTOR_NUMBER = 10

def evaluation_func(observation, eps=1e-6):
    """
    汎用 DRP 報酬関数（可変ノード数N, 可変エージェント数A, 可変エッジ数E）

    観測末尾:
        [..., edges_flat(3E), graph_diameter, N, A, E]
    """
    import heapq

    B = observation.shape[0]

    # ---------------------------------------------------------
    # 1. 末尾から graph_diameter, N, A, E を取得（全バッチ共通）
    # ---------------------------------------------------------
    graph_diameter = float(observation[0, -4])
    N = int(observation[0, -3])
    A = int(observation[0, -2])
    E = int(observation[0, -1])

    # ---------------------------------------------------------
    # 2. 可変長スライス用のサイズ計算
    # ---------------------------------------------------------
    # size_own_prev    = N
    # size_own_curr    = N
    # size_goal        = N
    size_others_prev = (A - 1) * N
    size_others_curr = (A - 1) * N
    # size_collision_distance = 1
    # size_collision_info     = 2
    # size_wait_count         = 1
    size_nodes_flat         = 2 * N
    size_edges_flat         = 3 * E         # ★可変

    # ---------------------------------------------------------
    # 3. 観測スライス
    # ---------------------------------------------------------
    idx = 0

    own_prev = observation[:, idx : idx + N]; idx += N
    own_curr = observation[:, idx : idx + N]; idx += N
    goal     = observation[:, idx : idx + N]; idx += N

    # others_prev = observation[:, idx : idx + size_others_prev]; idx += size_others_prev
    others_curr = observation[:, idx : idx + size_others_curr]; idx += size_others_curr

    collision_distance = observation[:, idx : idx + 1]; idx += 1
    collision_info     = observation[:, idx : idx + 2]; idx += 2
    wait_count         = observation[:, idx : idx + 1]; idx += 1

    nodes_flat = observation[:, idx : idx + size_nodes_flat]; idx += size_nodes_flat

    edges_flat = observation[0, idx : idx + size_edges_flat]  # flat, バッチ共通
    idx += size_edges_flat

    # ---------------------------------------------------------
    # 4. edges 変換（リスト形式）
    # ---------------------------------------------------------
    edges_arr = edges_flat.reshape(E, 3)
    edges = [(int(a), int(b), float(w)) for a, b, w in edges_arr]

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
        return dist[goal_node]

    # ---------------------------------------------------------
    # 8. エッジ上補間（α補間：i→j）
    # ---------------------------------------------------------
    def estimate_partial_distance(pos_vec, goal_node):
        nz = np.where(pos_vec > 1e-8)[0]

        # ノード上
        if len(nz) == 1:
            return dijkstra(int(nz[0]), goal_node)

        # エッジ上
        if len(nz) == 2:
            i, j = int(nz[0]), int(nz[1])
            wi, wj = pos_vec[i], pos_vec[j]

            s = wi + wj + eps
            alpha = wj / s  # j側の割合

            Di = dijkstra(i, goal_node)
            Dj = dijkstra(j, goal_node)

            return (1 - alpha) * Di + alpha * Dj

        # （通常起こらない）fallback
        pivot = int(np.argmax(pos_vec))
        return dijkstra(pivot, goal_node)

    # ---------------------------------------------------------
    # 9. 座標系での位置（エージェント・他エージェント）
    # ---------------------------------------------------------
    # agent_prev_pos = np.sum(own_prev[:, :, None] * nodes, axis=1)
    agent_curr_pos = np.sum(own_curr[:, :, None] * nodes, axis=1)
    # goal_pos       = np.sum(goal[:,     :, None] * nodes, axis=1)

    # 他エージェントは A-1 個
    others_curr_pos_list = []
    for k in range(A - 1):
        others_curr_pos_list.append(
            np.sum(others_curr[:, k*N:(k+1)*N, None] * nodes, axis=1)
        )
    others_curr_pos = np.stack(others_curr_pos_list, axis=1)  # (B, A-1, 2)

    # ---------------------------------------------------------
    # 10. グラフ距離（ダイクストラ）＋ 正規化
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
    # 11. 分離距離（ユークリッド）＋ 正規化
    # ---------------------------------------------------------
    sep = np.linalg.norm(agent_curr_pos[:, None, :] - others_curr_pos, axis=2)

    min_sep = np.min(sep, axis=1, keepdims=True)
    avg_sep = np.mean(sep, axis=1, keepdims=True)

    min_sep_norm = min_sep / (graph_diameter + eps)
    avg_sep_norm = avg_sep / (graph_diameter + eps)

    safety_margin = min_sep / (collision_distance + eps)
    collision_risk = (min_sep < collision_distance * 2).astype(float)

    wait_norm = wait_count.astype(float)
    in_collision = collision_info[:, 0:1]
    others_in_collision = collision_info[:, 1:2]

    # ---------------------------------------------------------
    # 12. 戻り値（従来と同じ順番で10要素）
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
