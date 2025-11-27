import numpy as np

def evaluation_func(observation, eps=1e-6):
    B = observation.shape[0]

    own_prev = observation[:, 0:20]           # 自分の前ステップ位置 (20d)
    own_curr = observation[:, 20:40]          # 自分の現在位置 (20d)
    goal = observation[:, 40:60]              # ゴール位置 (20d)
    other_agents_prev = observation[:, 60:100]  # 他2エージェントの前ステップ位置 (2×20d)
    other_agents_curr = observation[:, 100:140]  # 他2エージェントの現在位置 (2×20d)
    collision_distance = observation[:, 140:141]  # 衝突距離
    collision_info = observation[:, 141:143]     # [自分衝突, 他者衝突]
    wait_count = observation[:, 143:144]        # 連続待機カウント
    nodes_flat = observation[:, 144:184]       # 20ノード×2座標 = 40d

    nodes = nodes_flat.reshape(B, 20, 2)

    agent_prev_pos = np.sum(own_prev[:, :, None] * nodes, axis=1)  # (B,2)
    agent_curr_pos = np.sum(own_curr[:, :, None] * nodes, axis=1)  # (B,2)
    goal_pos = np.sum(goal[:, :, None] * nodes, axis=1)            # (B,2)

    other_prev_pos = np.stack([
        np.sum(other_agents_prev[:, 0:20, None] * nodes, axis=1),
        np.sum(other_agents_prev[:, 20:40, None] * nodes, axis=1)
    ], axis=1)  # (B,2,2)

    other_curr_pos = np.stack([
        np.sum(other_agents_curr[:, 0:20, None] * nodes, axis=1),
        np.sum(other_agents_curr[:, 20:40, None] * nodes, axis=1)
    ], axis=1)  # (B,2,2)

    dist_goal = np.linalg.norm(agent_curr_pos - goal_pos, axis=1, keepdims=True)
    dist_goal_prev = np.linalg.norm(agent_prev_pos - goal_pos, axis=1, keepdims=True)
    prog_goal = dist_goal_prev - dist_goal
    at_goal = (dist_goal < eps).astype(float)

    sep = np.linalg.norm(agent_curr_pos[:, None, :] - other_curr_pos, axis=2)  # (B,2)
    min_sep = np.min(sep, axis=1, keepdims=True)
    avg_sep = np.mean(sep, axis=1, keepdims=True)
    sep_var = np.var(sep, axis=1, keepdims=True)

    safety_margin = min_sep / (collision_distance + eps)
    collision_count = (sep < collision_distance).sum(axis=1, keepdims=True).astype(float)
    collision_risk = (min_sep < collision_distance).astype(float)

    wait_norm = wait_count.astype(float)

    in_collision = collision_info[:, 0:1]
    others_in_collision = collision_info[:, 1:2]

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