import numpy as np

def evaluation_func(observation, eps=1e-6):

    B = observation.shape[0]

    # ---- 1. 各部分を切り出し ----
    own_prev = observation[:, 0:9]           # 自分の前ステップ位置
    own_curr = observation[:, 9:18]          # 自分の現在位置
    goal = observation[:, 18:27]             # ゴール位置
    other_agents_prev = observation[:, 27:45]  # 他2エージェントの前ステップ位置 (18d)
    other_agents_curr = observation[:, 45:63]  # 他2エージェントの現在位置 (18d)
    collision_distance = observation[:, 63:64]  # 衝突距離
    collision_info = observation[:, 64:66]     # [自分衝突, 他者衝突]
    wait_count = observation[:, 66:67]        # 連続待機カウント
    nodes_flat = observation[:, 67:85]        # 9ノード×2座標

    # ---- 2. ノード座標を (B, 9, 2) に変形 ----
    nodes = nodes_flat.reshape(B, 9, 2)

    # ---- 3. 自エージェント位置計算（ワンホット×ノード座標の積和） ----
    agent_prev_pos = np.sum(own_prev[:, :, None] * nodes, axis=1)  # (B,2)
    agent_curr_pos = np.sum(own_curr[:, :, None] * nodes, axis=1)  # (B,2)
    goal_pos = np.sum(goal[:, :, None] * nodes, axis=1)            # (B,2)

    # ---- 4. 他エージェント位置計算（2エージェント×前後） ----
    # 2エージェント×9次元 → 2エージェント×ノード座標 (B,2,2)
    other_prev_pos = np.stack([
        np.sum(other_agents_prev[:, 0:9, None] * nodes, axis=1),
        np.sum(other_agents_prev[:, 9:18, None] * nodes, axis=1)
    ], axis=1)  # (B,2,2)

    other_curr_pos = np.stack([
        np.sum(other_agents_curr[:, 0:9, None] * nodes, axis=1),
        np.sum(other_agents_curr[:, 9:18, None] * nodes, axis=1)
    ], axis=1)  # (B,2,2)

    # ---- 5. ゴール距離と進捗 ----
    dist_goal = np.linalg.norm(agent_curr_pos - goal_pos, axis=1, keepdims=True)
    dist_goal_prev = np.linalg.norm(agent_prev_pos - goal_pos, axis=1, keepdims=True)
    prog_goal = dist_goal_prev - dist_goal
    at_goal = (dist_goal < eps).astype(float)

    # ---- 6. 他エージェントとの分離距離 ----
    sep = np.linalg.norm(agent_curr_pos[:, None, :] - other_curr_pos, axis=2)  # (B,2)
    min_sep = np.min(sep, axis=1, keepdims=True)
    avg_sep = np.mean(sep, axis=1, keepdims=True)
    sep_var = np.var(sep, axis=1, keepdims=True)

    # ---- 7. 安全余裕・衝突指標 ----
    safety_margin = min_sep / (collision_distance + eps)
    collision_count = (sep < collision_distance).sum(axis=1, keepdims=True).astype(float)
    collision_risk = (min_sep < collision_distance).astype(float)

    # ---- 8. 待機カウント ----
    wait_norm = wait_count.astype(float)

    # ---- 9. 提供衝突情報 ----
    in_collision = collision_info[:, 0:1]
    others_in_collision = collision_info[:, 1:2]

    # ---- 10. 結果をリストで返す ----
    return [
        prog_goal,
        in_collision,
        others_in_collision,
        wait_norm,
        dist_goal,
        min_sep,
        avg_sep,
        sep_var,
        safety_margin,
        collision_count,
        collision_risk,
        at_goal
    ]

FACTOR_NUMBER = 12  # 評価指標の数