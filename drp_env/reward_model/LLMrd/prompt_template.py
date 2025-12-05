import json
import numpy as np
import sys
import os
import csv

factor_role_instrutor = f"ROLE INSTRUCTION: You are good at understanding tasks and writing python codes.\
You should fully understand the provided task and describe the exact observation form in the current map. \
Then, based on your understanding, analyze potential positive and negative behaviours or statuses that can be reflected in the observation.\
Finally, write an evaluation function that returns factors evaluating the current status from different aspects. \
Note:1. Do not use information you are not given! \
2. Focus on the most relevant evaluation factors and use information in observation as little as possible. \
3. The code should be as generic, complete and not contain omissions! \
4. Avoid dividing by zero!\
5. The input variable is in the form of (batch_size, dim), please return a list of several evaluation factor arrays, each in the form of (batch_size, 1). \
Please think step by step and adhere to the following JSON format (just replace the () with your answer):"+\
"{\
Understand: (give your thought about the exact observation form in current map),\
Analyze: (think step by step and analyze potential positive and negative behaviors or statuses that can be reflected in observations), \
Functions: (a python function with the form of 'def evaluation_func(observation): ... return [a list of evaluation factor arrays]')\
}"

class Base_prompt(object):
    def __init__(self, map_name, factor_decomp=False) -> None:
        self.map_name = map_name
        self.factor_decomp = factor_decomp
        self.task_description = ''
        self.state_form = ''
        self.role_instruction = ''

    def get_message(self):
        message=[]
        message.append({'role':'user','content':self.task_description+self.state_form+self.role_instruction})
        return message

    def factor_check(self, out_content):
        error_idx, error_content = -1, ''
        pass_check = True
        factor_num = 0
        for i in range(len(out_content)):
            try:
                func = json.loads(out_content[i])['Functions']
                namespace = {}
                exec(func,namespace)
                active_evaluation_func = namespace['evaluation_func']
                evaluation_factors = active_evaluation_func(np.stack(self.obs, axis=0))
                factor_num = len(evaluation_factors)
                for factor in evaluation_factors:
                    if len(factor.shape) != 2 or factor.shape[0] != self.n_agents or factor.shape[1] != 1:
                        pass_check=False
                        error_idx = i
                        error_content = f'There is an error in your previous answer. Error: The shape of the output factors should be (batch_size, 1).'
            except Exception as e:
                pass_check=False
                error_idx = i
                error_content = f'There is an error in your previous answer. Error:{e.args}' 
        return pass_check, error_idx, error_content, factor_num

class DRP_prompt(Base_prompt):
    def __init__(self, map_name, factor_decomp=False, n_agents=None, collision_dis=None, speed=None) -> None:
        super().__init__(map_name, factor_decomp)
        
        # マップ情報の設定
        self.map_info = self._get_map_info(map_name)
        self.n_nodes = self.map_info["total_nodes"]

        # CSVファイルからの追加情報読み込み
        self.csv_node_info = self._load_csv_node_info(map_name)
        self.csv_edge_info = self._load_csv_edge_info(map_name)

        self.n_edges = len(self.get_csv_edges())
        
        # エージェント数の設定（パラメータで指定可能）
        if n_agents is not None:
            self.n_agents = n_agents
        else:
            # マップに応じたデフォルト値
            self.n_agents = self._get_default_agent_count(map_name)
        
        # 衝突判定距離の設定
        self.collision_dis = collision_dis if collision_dis is not None else 0.1
        
        # 移動速度の設定
        self.speed = speed if speed is not None else 5.0
        
        # 観測次元の計算: agent_location + goal_location + other_agents_relative + collision_distance + collsion_info+ node_info
        # エッジ情報を除外
        # 💥 修正: drp_env.pyの結合順序と完全に一致させる
        self.obs_dim = (self.n_nodes * 2 + 						    # agent_location
                        self.n_nodes + 							    # goal_location
                        (self.n_agents - 1) * self.n_nodes * 2 + 	# other_agents_relative
                        1 + 									    # collision_distance
                        2 + 									    # collision_info
                        1 + 								        # wait_count
                        self.n_nodes * 2 +                          # node_info
                        self.n_edges * 3 +                          # edge_info                                              
                        1 +                                         # graph_diameter
                        3                                           # meta_info (node_num, agent_num, edge_num)
                        )						    
        
        # ダミー観測の作成（factor_checkのため）
        self.obs = []
        for i in range(self.n_agents):
            # 🔧 修正: エージェントの位置を[前の状態, 現在の状態]形式に変更
            agent_pos_prev = np.zeros(self.n_nodes)
            agent_pos_prev[i % self.n_nodes] = 1.0  # 前の状態: i番目のノードにいたと仮定
            
            agent_pos_current = np.zeros(self.n_nodes)
            agent_pos_current[(i + 1) % self.n_nodes] = 1.0  # 現在の状態: 次のノードに移動したと仮定
            
            # [前の状態, 現在の状態]を結合 (80次元)
            agent_pos_combined = np.concatenate([agent_pos_prev, agent_pos_current])
            
            # ゴール位置（変化しない）
            goal_pos = np.zeros(self.n_nodes)
            goal_pos[(i + 3) % self.n_nodes] = 1.0  # 適当なゴール
            
            # 🔧 修正: 他のエージェントの位置も[前の状態, 現在の状態]形式に変更
            other_agents = []
            for j in range(self.n_agents):
                if j != i:
                    # 他のエージェントの前の状態
                    other_prev = np.zeros(self.n_nodes)
                    other_prev[j % self.n_nodes] = 1.0
                    
                    # 他のエージェントの現在の状態
                    other_current = np.zeros(self.n_nodes)
                    other_current[(j + 1) % self.n_nodes] = 1.0
                    
                    # [前の状態, 現在の状態]を結合
                    other_combined = np.concatenate([other_prev, other_current])
                    other_agents.append(other_combined)
            
            other_agents = np.concatenate(other_agents) if other_agents else np.array([])
            
            # 衝突距離（スカラー値）
            collision_distance = np.array([self.collision_dis])

            # 💥 追加: ダミーの衝突情報を作成
            # 例: agent 0 は衝突に関与、agent 1 は他者の衝突を観測
            self_involved = 1 if i == 0 else 0
            others_exist = 1 if i == 1 else 0
            collision_info = np.array([self_involved, others_exist], dtype=float)

            wait_count = np.array([float(i % 3)])  # 待機カウント（ダミー値）
            
            # ノード座標情報を平坦化 (x, y) × n_nodes
            node_coords = self.get_csv_node_coordinates()
            node_coordinates = []
            for node_id, x, y in sorted(node_coords, key=lambda item: item[0]):
                node_coordinates.extend([float(x), float(y)])
            node_coordinates = np.array(node_coordinates)

            # 💥 修正: 全ての情報をdrp_env.pyと同じ順序で結合
            obs_i = np.concatenate([
                agent_pos_combined,      # shape: (n_nodes * 2,)
                goal_pos,                # shape: (n_nodes,)
                other_agents,            # shape: ((n_agents-1) * n_nodes * 2,)
                collision_distance,      # shape: (1,)
                collision_info,          # shape: (2,)
                wait_count,              # shape: (1,)  
                node_coordinates,        # shape: (n_nodes * 2,)
            ])
            
            self.obs.append(obs_i)

        # 観測次元の検証（デバッグ用、必要に応じて非表示にできる）
        expected_dim = self.obs_dim
        actual_dim = len(self.obs[0]) if self.obs else 0
        
        if expected_dim != actual_dim:
            print(f"⚠️ 観測次元不一致: 期待値={expected_dim}, 実際={actual_dim}")
            print(f"   - n_nodes: {self.n_nodes}")
            print(f"   - n_agents: {self.n_agents}")
            print(f"   - node_info size: {self.n_nodes * 3}")

        # タスクの説明
        self.task_description = f"TASK: We are solving the Drone Routing Problem (DRP). In this task, {self.n_agents} agents are each assigned an individual goal location. \
Each agent should navigate toward its own goal while avoiding collisions with other agents. \
The objective is to plan each agent's path so that all agents successfully reach their respective goals and collisions are avoided at all times.\n\
Movement Model(speed): At every time step, an agent may move up to {self.speed} units of Euclidean distance. \
If the Euclidean distance from the agent's current position to its goal node is less than {self.speed}, the agent moves directly to the goal node and stops there.\n\
Graph connectivity(Movement constraints):\
- Agents can only move along edges.\
- To move, an agent:\
    1. Selects one adjacent node via a connecting edge.\
    2. Travels along that edge up to {self.speed} units.\
    3. If the edge length is ≤ {self.speed}, traverses the full edge.\
    4. If the edge length > {self.speed}, moves partway (no shortcuts).\
Collision Definition: A collision is defined as two agents coming within {self.collision_dis} units of each other.\n\
Distance Calculation: The Euclidean distance between two points (x1, y1) and (x2,y2) is calculated as: \
distance = sqrt((x2 - x1)^2 + (y2 - y1)^2)\n\
Where:x1,y1 are the coordinates of the first point (either the agent's current position or the goal position), \
x2,y2 are the coordinates of the second point (either the other agent's position or the goal position).\n\
Objective: Plan each agent's path so that all agents successfully reach their respective goals and collisions are avoided at all times. \
Please suggest appropriate next actions for each agent based on this task.\n\n"

        self.state_form = f"OBSERVATION FORM: At each time step, each agent receives an observation represented as a single array. \
This observation consists of the following concatenated components (total dimension: {self.obs_dim}): \
- The agent's own location: a {self.n_nodes * 2}-dimensional vector, split into two {self.n_nodes}-dimensional parts: \
Previous state (first {self.n_nodes} dimensions): the agent's location before taking the current action. \
Current state (last {self.n_nodes} dimensions): the agent's location after completing the action. \
Each part is normally a one-hot vector, but possibly containing continuous values if the agent is currently on an edge between two nodes. In such cases, the vector represents the proportional position along that edge. \
- The agent's goal location: a {self.n_nodes}-dimensional one-hot vector (unchanged between time steps). \
- The absolute positions of the other {self.n_agents - 1} agents, each represented as a {self.n_nodes * 2}-dimensional vector in the same [previous_state, current_state] format. \
- The collision distance {self.collision_dis}: a scalar value representing the minimum required separation between agents to avoid collisions. This can be used as a threshold to determine whether the agents are at risk of colliding. \
- The collision information: a 2-demensional vector where the first elsement indicates whether the agent is involved in a collision (1 if yes, 0 if no), and the second element indicates whether there are other agents within collision distance (1 if yes, 0 if no). \
- The wait count: a scalar value indicating how many time steps the agent has been waiting at its current node. This value resets to 0 when the agent moves. Importantly, this counter does not increment if the agent is waiting at its assigned goal node. \
- Node coordinates: A flattened array of {self.n_nodes * 2} values representing node coordinates. Every 2 consecutive values represent one node as (x, y). \
For example, with 3 nodes: [x0, y0, x1, y1, x2, y2] where node 0 is at (x0, y0), node 1 is at (x1, y1), and node 2 is at (x2, y2). \
The node coordinates are ordered by node ID (0, 1, 2, ..., {self.n_nodes-1}). \ \
To access coordinates for node i: x_coordinate = observation[base_index + i*2], y_coordinate = observation[base_index + i*2 + 1], where base_index = {self.n_nodes * 2 + self.n_nodes + (self.n_agents - 1) * self.n_nodes *  2 + 1}. \n\
A one-hot vector is a binary vector where only the element corresponding to the current node index is 1, and all other elements are 0. \
For example, if there are 10 nodes and the agent is at node 3, the vector will be: [0, 0, 0, 1, 0, ...]. \
However, if the agent is between node 3 and node 7, and 30% of the way from node 3 to node 7, the vector might look like: [0, 0, 0, 0.7, 0, 0, 0, 0.3, 0, ...]. \n\
MAP INFORMATION: Please refer to the following map data. You should not assume any information beyond what is given in this structure.\n{self._format_detailed_map_info()}\n"

        self.role_instruction = factor_role_instrutor

    def _get_default_agent_count(self, map_name):
        """マップに応じたデフォルトエージェント数を取得"""
        default_counts = {
            'map_3x3': 3,
            'map_5x5': 5,
            'map_8x5': 5,
            'aoba01': 4
        }
        
        for key, count in default_counts.items():
            if key in map_name:
                return count
        
        # 未知のマップの場合、ノード数に基づいて決定
        return min(5, max(2, self.n_nodes // 8))

    def _get_map_info(self, map_name):
        """マップ名からマップ情報を取得"""
        if 'map_3x3' in map_name:
            return {
                "topology": "3x3_grid",
                "total_nodes": 9,
                "nodes": {
                    0: {"pos": (8.94, 5.52), "neighbors": [1, 3]},
                    1: {"pos": (25.12, 2.35), "neighbors": [0, 2, 4]},
                    2: {"pos": (48.72, 8.74), "neighbors": [1, 5]},
                    3: {"pos": (7.17, 22.54), "neighbors": [0, 4, 6]},
                    4: {"pos": (24.75, 20.62), "neighbors": [1, 3, 5, 7]},
                    5: {"pos": (44.14, 22.09), "neighbors": [2, 4, 8]},
                    6: {"pos": (0.24, 45.30), "neighbors": [3, 7]},
                    7: {"pos": (29.60, 41.13), "neighbors": [4, 6, 8]},
                    8: {"pos": (48.90, 46.12), "neighbors": [5, 7]}
                },
                "edges": [
                    (0,1), (0,3), (1,2), (1,4), (2,5), (3,4), (3,6), (4,5),
                    (4,7), (5,8), (6,7), (7,8)
                ]
            }
        elif 'map_5x4' in map_name:
            return {
                "topology": "5x4_grid",
                "total_nodes": 20,
                "nodes": {
                    0: {"pos": (6.49, 8.21), "neighbors": [1, 5]},
                    1: {"pos": (29.71, 5.59), "neighbors": [0, 2, 6]},
                    2: {"pos": (43.04, 7.42), "neighbors": [1, 3, 7]},
                    3: {"pos": (62.44, 2.57), "neighbors": [2, 4, 8]},
                    4: {"pos": (81.93, 4.08), "neighbors": [3, 9]},
                    5: {"pos": (1.50, 22.00), "neighbors": [0, 6, 10]},
                    6: {"pos": (21.05, 21.17), "neighbors": [1, 5, 7, 11]},
                    7: {"pos": (44.05, 25.84), "neighbors": [2, 6, 8, 12]},
                    8: {"pos": (67.30, 28.27), "neighbors": [3, 7, 9, 13]},
                    9: {"pos": (81.26, 20.42), "neighbors": [4, 8, 14]},
                    10: {"pos": (0.69, 46.80), "neighbors": [5, 11, 15]},
                    11: {"pos": (29.05, 40.06), "neighbors": [6, 10, 12, 16]},
                    12: {"pos": (48.26, 47.20), "neighbors": [7, 11, 13, 17]},
                    13: {"pos": (69.57, 47.68), "neighbors": [8, 12, 14, 18]},
                    14: {"pos": (89.73, 48.80), "neighbors": [9, 13, 19]},
                    15: {"pos": (4.76, 65.91), "neighbors": [10, 16]},
                    16: {"pos": (28.93, 69.91), "neighbors": [11, 15, 17]},
                    17: {"pos": (42.95, 67.29), "neighbors": [12, 16, 18]},
                    18: {"pos": (62.85, 64.52), "neighbors": [13, 17, 19]},
                    19: {"pos": (87.80, 68.21), "neighbors": [14, 18]}
                },
                "edges": [
                    (0, 1), (0, 5), (1, 2), (1, 6), (2, 3), (2, 7), (3, 4), (3, 8),
                    (4, 9), (5, 6), (5, 10), (6, 7), (6, 11), (7, 8), (7, 12), (8, 9),
                    (8, 13), (9, 14), (10, 11), (10, 15), (11, 12), (11, 16), (12, 13),
                    (12, 17), (13, 14), (13, 18), (14, 19), (15, 16), (16, 17), (17, 18),
                    (18, 19)
                ]
            }                
        elif 'map_8x5' in map_name:
            return {
                "topology": "8x5_grid",
                "total_nodes": 40,
                "nodes": {
                    0: {"pos": (4.07, 5.27), "neighbors": [1, 8]},
                    1: {"pos": (28.27, 0.11), "neighbors": [0, 2, 9]},
                    2: {"pos": (44.32, 4.64), "neighbors": [1, 3, 10]},
                    3: {"pos": (61.37, 5.26), "neighbors": [2, 4, 11]},
                    4: {"pos": (82.81, 9.73), "neighbors": [3, 5, 12]},
                    5: {"pos": (102.41, 9.04), "neighbors": [4, 6, 13]},
                    6: {"pos": (129.12, 8.81), "neighbors": [5, 7, 14]},
                    7: {"pos": (143.94, 5.12), "neighbors": [6, 15]},
                    8: {"pos": (9.86, 20.91), "neighbors": [0, 9, 16]},
                    9: {"pos": (28.03, 26.45), "neighbors": [1, 8, 10, 17]},
                    10: {"pos": (40.88, 26.18), "neighbors": [2, 9, 11, 18]},
                    11: {"pos": (69.12, 23.20), "neighbors": [3, 10, 12, 19]},
                    12: {"pos": (88.66, 23.05), "neighbors": [4, 11, 13, 20]},
                    13: {"pos": (108.31, 20.09), "neighbors": [5, 12, 14, 21]},
                    14: {"pos": (124.01, 28.24), "neighbors": [6, 13, 15, 22]},
                    15: {"pos": (141.18, 20.68), "neighbors": [7, 14, 23]},
                    16: {"pos": (0.01, 46.76), "neighbors": [8, 17, 24]},
                    17: {"pos": (28.44, 48.24), "neighbors": [9, 16, 18, 25]},
                    18: {"pos": (46.87, 41.75), "neighbors": [10, 17, 19, 26]},
                    19: {"pos": (65.07, 48.73), "neighbors": [11, 18, 20, 27]},
                    20: {"pos": (82.15, 47.62), "neighbors": [12, 19, 21, 28]},
                    21: {"pos": (107.31, 48.91), "neighbors": [13, 20, 22, 29]},
                    22: {"pos": (124.67, 40.49), "neighbors": [14, 21, 23, 30]},
                    23: {"pos": (146.14, 48.17), "neighbors": [15, 22, 31]},
                    24: {"pos": (9.89, 60.28), "neighbors": [16, 25, 32]},
                    25: {"pos": (24.99, 66.67), "neighbors": [17, 24, 26, 33]},
                    26: {"pos": (43.66, 65.61), "neighbors": [18, 25, 27, 34]},
                    27: {"pos": (68.31, 62.13), "neighbors": [19, 26, 28, 35]},
                    28: {"pos": (86.35, 61.71), "neighbors": [20, 27, 29, 36]},
                    29: {"pos": (102.64, 63.98), "neighbors": [21, 28, 30, 37]},
                    30: {"pos": (123.61, 62.95), "neighbors": [22, 29, 31, 38]},
                    31: {"pos": (141.72, 63.18), "neighbors": [23, 30, 39]},
                    32: {"pos": (7.00, 82.20), "neighbors": [24, 33]},
                    33: {"pos": (22.01, 82.82), "neighbors": [25, 32, 34]},
                    34: {"pos": (47.54, 84.78), "neighbors": [26, 33, 35]},
                    35: {"pos": (63.59, 80.92), "neighbors": [27, 34, 36]},
                    36: {"pos": (82.12, 88.68), "neighbors": [28, 35, 37]},
                    37: {"pos": (106.04, 80.13), "neighbors": [29, 36, 38]},
                    38: {"pos": (129.06, 89.70), "neighbors": [30, 37, 39]},
                    39: {"pos": (143.12, 84.81), "neighbors": [31, 38]}
                },
                "edges": [
                    (0,1), (0,8), (1,2), (1,9), (2,3), (2,10), (3,4), (3,11),
                    (4,5), (4,12), (5,6), (5,13), (6,7), (6,14), (7,15), (8,9),
                    (8,16), (9,10), (9,17), (10,11), (10,18), (11,12), (11,19),
                    (12,13), (12,20), (13,14), (13,21), (14,15), (14,22), (15,23),
                    (16,17), (16,24), (17,18), (17,25), (18,19), (18,26), (19,20),
                    (19,27), (20,21), (20,28), (21,22), (21,29), (22,23), (22,30),
                    (23,31), (24,25), (24,32), (25,26), (25,33), (26,27), (26,34),
                    (27,28), (27,35), (28,29), (28,36), (29,30), (29,37), (30,31),
                    (30,38), (31,39), (32,33), (33,34), (34,35), (35,36), (36,37),
                    (37,38), (38,39)
                ]
            }
        elif 'aoba01' in map_name:
            return {
                "topology": "complex_network",
                "total_nodes": 18,
                "nodes": {
                    0: {"pos": (800.38, 945.74), "neighbors": [1, 9]},
                    1: {"pos": (862.71, 935.52), "neighbors": [0, 2, 10]},
                    2: {"pos": (922.10, 928.50), "neighbors": [1, 3, 12]},
                    3: {"pos": (962.78, 923.87), "neighbors": [2, 4, 13]},
                    4: {"pos": (1001.31, 920.72), "neighbors": [3, 5]},
                    5: {"pos": (1003.78, 887.41), "neighbors": [4, 6]},
                    6: {"pos": (1021.21, 870.80), "neighbors": [5, 7]},
                    7: {"pos": (1072.62, 848.82), "neighbors": [6, 8]},
                    8: {"pos": (1060.85, 811.55), "neighbors": [7, 13, 15, 17]},
                    9: {"pos": (842.07, 778.24), "neighbors": [0, 10]},
                    10: {"pos": (898.13, 792.96), "neighbors": [1, 9, 11]},
                    11: {"pos": (960.05, 807.38), "neighbors": [10, 12, 14]},
                    12: {"pos": (952.12, 838.59), "neighbors": [2, 11]},
                    13: {"pos": (982.15, 847.61), "neighbors": [3, 8]},
                    14: {"pos": (1013.58, 798.26), "neighbors": [11, 15]},
                    15: {"pos": (1054.73, 787.94), "neighbors": [8, 14, 16]},
                    16: {"pos": (1135.16, 800.41), "neighbors": [15, 17]},
                    17: {"pos": (1132.87, 829.27), "neighbors": [8, 16]}
                },
                "edges": [
                    (0, 1), (0, 9), (1, 2), (1, 10), (2, 3), (2, 12), (3, 4), (3, 13),
                    (4, 5), (5, 6), (6, 7), (7, 8), (8, 13), (8, 15), (8, 17), (9, 10),
                    (10, 11), (11, 12), (11, 14), (14, 15), (15, 16), (16, 17)
                ]
            }
        else:
            # デフォルト（3x3グリッド）
            return self._get_map_info('map_3x3')
    
    def _load_csv_node_info(self, map_name):
        """CSVファイルからノード情報を読み込む（相対パス）"""
        csv_path_candidates = [
            f"../../drp_env/map/{map_name}/node.csv",
            f"../../../drp_env/map/{map_name}/node.csv",
            f"drp_env/map/{map_name}/node.csv",
            f"../../../../drp_env/map/{map_name}/node.csv",
        ]
        
        for csv_path in csv_path_candidates:
            if os.path.exists(csv_path):
                try:
                    return self._parse_node_csv(csv_path)
                except Exception as e:
                    continue
        
        return None

    def _load_csv_edge_info(self, map_name):
        """CSVファイルからエッジ情報を読み込む（相対パス）"""
        csv_path_candidates = [
            f"../../drp_env/map/{map_name}/edge.csv",
            f"../../../drp_env/map/{map_name}/edge.csv",
            f"drp_env/map/{map_name}/edge.csv",
            f"../../../../drp_env/map/{map_name}/edge.csv",
        ]
        
        for csv_path in csv_path_candidates:
            if os.path.exists(csv_path):
                try:
                    return self._parse_edge_csv(csv_path)
                except Exception as e:
                    continue
        
        return None

    def _parse_node_csv(self, csv_path):
        """node.csvファイルをパースしてノード座標情報を抽出"""
        node_coordinates = []
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                try:
                    node_id = int(row['ID(ignored)'].strip())
                    x = float(row['x'].strip())
                    y = float(row['y'].strip())
                    node_coordinates.append((node_id, x, y))
                except Exception as e:
                    continue
        
        return {
            "node_coordinates": node_coordinates,
            "total_csv_nodes": len(node_coordinates),
            "csv_path": csv_path
        }

    def _parse_edge_csv(self, csv_path):
        """edge.csvファイルをパースしてエッジ情報を抽出"""
        edges = []
        edge_set = set()
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                try:
                    from_node = int(row['from'].strip())
                    to_node = int(row['to'].strip())
                    
                    # エッジを正規化（小さい方を先に）- 双方向として扱う
                    edge = tuple(sorted([from_node, to_node]))
                    
                    if edge not in edge_set:
                        edge_set.add(edge)
                        edges.append(edge)
                        
                except Exception as e:
                    continue
        
        return {
            "edges": edges,
            "total_csv_edges": len(edges),
            "csv_path": csv_path
        }

    def get_csv_node_coordinates(self):
        """CSVから読み込んだ(ID, x, y)タプルのリストを取得"""
        if self.csv_node_info and 'node_coordinates' in self.csv_node_info:
            return self.csv_node_info['node_coordinates']
        else:
            # CSVがない場合は、ハードコードされた座標からタプル生成
            coordinates = []
            for node_id, node_data in self.map_info['nodes'].items():
                x, y = node_data['pos']
                coordinates.append((node_id, x, y))
            return sorted(coordinates, key=lambda item: item[0])

    def get_csv_edges(self):
        """CSVから読み込んだエッジリストを取得"""
        if self.csv_edge_info and 'edges' in self.csv_edge_info:
            return self.csv_edge_info['edges']
        else:
            # CSVがない場合は、ハードコードされたエッジを返す
            return self.map_info.get('edges', [])

    def _format_detailed_map_info(self):
        """詳細なマップ情報を文字列としてフォーマット（プロンプト用）"""
        info = self.map_info
        
        formatted_info = f"Map topology: {info['topology']}\n"
        formatted_info += f"Total nodes: {info['total_nodes']}\n"

        # CSVから読み込んだ情報がある場合は併記
        if self.csv_node_info:
            formatted_info += "\n=== ACTUAL DATA FROM CSV FILES ===\n"
            formatted_info += "Node coordinates from CSV (for reference):\n"
            csv_coordinates = self.get_csv_node_coordinates()
            for coord in csv_coordinates:
                node_id, x, y = coord
                formatted_info += f"  Node {node_id}: ({x:.2f}, {y:.2f})\n"  # ID: (x, y) 形式
            formatted_info += "\n"
            formatted_info += "IMPORTANT: These coordinates are embedded in the observation array as flattened (x, y) pairs!\n"
            formatted_info += "Access them using the index formula provided in the observation form.\n\n"

        formatted_info += "=== GRAPH STRUCTURE ===\n"
        formatted_info += "Node connectivity information:\n"
        
        # ノード情報の詳細
        for node_id, node_data in info['nodes'].items():
            pos = node_data['pos']
            neighbors = node_data['neighbors']
            formatted_info += f"  Node {node_id}: position {pos}, neighbors {neighbors}\n"
        
        # エッジ情報
        edges = self.get_csv_edges() if self.csv_edge_info else info.get('edges', [])
        formatted_info += f"\nEdges: {edges}\n"
        
        # 追加の説明
        if info['topology'] == 'complex_network':
            formatted_info += "\nThis is a complex network where nodes have irregular connectivity patterns. "
            formatted_info += "Node positions are given in coordinate system and connectivity varies significantly between nodes.\n"
        elif '8x5_grid' in info['topology']:
            formatted_info += "\nThis is an 8x5 grid-based network with 40 nodes arranged in 8 columns and 5 rows. "
            formatted_info += "The grid has irregular spacing between nodes and varying connectivity patterns.\n"
        elif 'grid' in info['topology']:
            formatted_info += "\nThis is a regular grid network where nodes are arranged in a grid pattern. "
            formatted_info += "Most nodes have 4 neighbors (up, down, left, right) except for boundary nodes.\n"
        
        return formatted_info
    
def get_prompt(env_name, map_name, factor_decomp=False, **kwargs):
    """プロンプト取得関数（パラメータ対応）"""
    return DRP_prompt(
        map_name, 
        factor_decomp=factor_decomp,
        n_agents=kwargs.get('n_agents'),
        collision_dis=kwargs.get('collision_dis'),
        speed=kwargs.get('speed')
    )