import gym  # gymライブラリのインポート
import numpy as np  # numpyライブラリのインポート
import sys  # sysモジュールのインポート
import copy  # copyモジュールのインポート
import os  # osモジュールのインポート

# 状態表現とマップ生成のインポート
from drp_env.state_repre import REGISTRY  # 状態表現管理用のレジストリをインポート
from drp_env.EE_map import MapMake  # マップ生成クラスのインポート

sys.path.append(os.path.join(os.path.dirname(__file__), ''))  # カレントディレクトリをパスに追加

class DrpEnv(gym.Env):  # DrpEnvクラスの定義（gym.Envを継承）
	# 環境の初期化
	def __init__(self,
			agent_num,  # エージェント数
			speed,  # エージェントの速度
			start_ori_array,  # スタートノード配列
			goal_array,  # ゴールノード配列
			visu_delay,  # 可視化ディレイ
			state_repre_flag,  # 状態表現フラグ
			time_limit,  # タイムリミット
			collision,  # 衝突判定方式
			map_name="map_3x3",  # マップ名
			reward_list={
				"goal": 100.0,      # ゴール報酬（変更なし）
				"collision": -20.0,  # 衝突報酬（より重いペナルティ）
				"wait": -5.0,       # 待機報酬（緩めのペナルティ）
				"move": -1.0,       # 移動報酬（変更なし）
			},  # 報酬設定
		  ):
		# エージェント数や各種パラメータの設定
		self.agent_num = agent_num  # エージェント数
		self.n_agents = agent_num # epymarl用
		self.state_repre_flag = state_repre_flag  # 状態表現フラグ
		self.map_name = map_name  # マップ名
		self.speed = speed  # 速度
		self.visu_delay = visu_delay  # 可視化ディレイ
		self.start_ori_array = start_ori_array  # スタートノード配列
		self.goal_array = goal_array  # ゴールノード配列

		 # 報酬の設定
		self.r_goal = reward_list["goal"]  # ゴール報酬
		self.r_coll = reward_list["collision"]  # 衝突報酬
		self.r_wait = reward_list["wait"]  # 待機報酬
		self.r_move = reward_list["move"]  # 移動報酬

		 # 衝突判定の仕組み
		self.collision = collision  # 衝突判定方式

		self.time_limit = time_limit  # タイムリミット

		self.colli_distan_value = 0.1  # 衝突判定距離
		self.r_flag = 0  # 報酬フラグ
		self.flag_indicate = 0  # フラグ指標
		self.episode_account = 0  # エピソードカウント

		self.distance_from_start = np.zeros(self.agent_num)  # スタートからの距離

		# EE環境の生成と変数の受け渡し
		self.ee_env = MapMake(self.agent_num, self.start_ori_array, self.goal_array, self.map_name)  # マップ生成
		self.pos = self.ee_env.pos  # ノード座標
		self.start_ori_array = self.ee_env.start_ori_array  # スタートノード配列
		self.goal_array = self.ee_env.goal_array  # ゴールノード配列
		self.G = self.ee_env.G  # グラフ
		self.edge_labels = self.ee_env.edge_labels # 未使用

		self.current_goal  = [ None for i in range(self.agent_num)]  # 現在のゴール

		# 観測管理クラスの生成
		self.obs_manager = REGISTRY[self.state_repre_flag](self)  # 状態表現管理

		# gym形式のMDP要素の作成
		self.n_nodes = len(self.G.nodes)  # ノード数
		self.n_actions = self.n_nodes  # 行動数
		self.action_space = gym.spaces.Tuple(tuple([gym.spaces.Discrete(self.n_nodes)] * self.agent_num))  # 行動空間
		
		obs_box = self.obs_manager.get_obs_box()  # 観測空間ボックス
		self.observation_space = gym.spaces.Tuple(tuple([obs_box] * self.agent_num))  # 観測空間
		

	# 現在の観測を返す
	def get_obs(self):
		return self.obs  # 観測を返す

	# 状態取得（未使用）
	def get_state(self): # unused
		return self.s  # 状態を返す（未使用）

	# 指定エージェントの利用可能な行動を取得
	def _get_avail_agent_actions(self, agent_id, n_actions):
		avail_actions = self.ee_env.get_avail_action_fun(self.obs[agent_id], self.current_start[agent_id], self.current_goal[agent_id], self.goal_array[agent_id])  # 利用可能な行動
		avail_actions_one_hot = np.zeros(n_actions)  # one-hotベクトル
		avail_actions_one_hot[avail_actions] = 1  # 利用可能な行動を1に
		return avail_actions_one_hot, avail_actions  # one-hotとリストを返す
	
	def get_avail_agent_actions(self, agent_id, n_actions):
		return self._get_avail_agent_actions(agent_id, n_actions)  # 内部関数を呼ぶ

	# 環境のリセット
	def reset(self):
		# ゴールとスタートが未設定の場合は毎エピソードランダム生成
		self.start_ori_array = copy.deepcopy(self.ee_env.input_start_ori_array)  # スタートノード配列
		self.goal_array = copy.deepcopy(self.ee_env.input_goal_array)  # ゴールノード配列
		print("self.start_ori_array", self.start_ori_array)  # デバッグ出力
		if self.start_ori_array == []:
			self.ee_env.random_start()  # ランダムスタート
			self.start_ori_array = self.ee_env.start_ori_array  # スタートノード配列
		if self.goal_array == []:
			self.ee_env.random_goal()  # ランダムゴール
			self.goal_array = self.ee_env.goal_array  # ゴールノード配列
		print("self.start_ori_array after", self.start_ori_array)  # デバッグ出力

		# 観測の初期化
		self.obs = tuple(np.array([self.pos[self.start_ori_array[i]][0], self.pos[self.start_ori_array[i]][1], self.start_ori_array[i], self.goal_array[i]]) for i in range(self.agent_num))  # 観測
		self.obs_current_chache = copy.deepcopy(self.obs)# 報酬計算用
		
		# one-hot観測の初期化
		self.obs_onehot = np.zeros((self.agent_num, self.n_nodes*2))  # one-hot観測
		for i in range(self.agent_num):
			self.obs_onehot[i][int(self.start_ori_array[i])] = 1 # 現在位置
			self.obs_onehot[i][int(self.goal_array[i])+self.n_nodes] = 1 # ゴール

		self.current_start = self.start_ori_array # [0,1] 現在のスタート
		self.current_goal  = [None for _ in range(self.agent_num)]  # 現在のゴール
		self.terminated    = [False for _ in range(self.agent_num)]  # 終了フラグ

		self.distance_from_start = np.zeros(self.agent_num) # info スタートからの距離
		self.wait_count = np.zeros(self.agent_num) # info 待機回数

		self.reach_account = 0  # ゴール到達数
		self.step_account = 0  # ステップ数
		self.episode_account += 1  # エピソード数
		print('Environment reset obs: \n', self.obs)  # デバッグ出力

		obs = self.obs_manager.calc_obs()  # 状態表現計算

		return obs  # 観測を返す
		

	# 環境の1ステップ進行
	def step(self, joint_action):
		# joint_actionに基づき環境を遷移
		self.step_account += 1  # ステップ数加算
		self.obs_current_chache = copy.deepcopy(self.obs)  # 前回観測の保存

		self.obs_prepare = []  # 次状態観測準備
		self.obs_onehot_prepare = copy.deepcopy(self.obs_onehot)  # one-hot観測準備
		self.current_start_prepare = copy.deepcopy(self.current_start)  # スタートノード準備
		self.current_goal_prepare = copy.deepcopy(self.current_goal)  # ゴールノード準備
		# 各エージェントの行動可否判定と観測準備
		for i in range(self.agent_num):
			action_i = joint_action[i]  # 各エージェントの行動
			# 利用不可の場合は前の観測を維持
			if action_i not in self._get_avail_agent_actions(i, self.n_actions)[1]:
				self.obs_prepare.append(self.obs_current_chache[i])  # 前回観測を維持
				self.wait_count[i] += 1  # 待機回数加算

			# 現在ノードで停止の場合
			elif self.pos[int(action_i)][0]==self.obs[i][0] and self.pos[int(action_i)][1]==self.obs[i][1]:
				self.obs_prepare.append(self.obs_current_chache[i])  # 前回観測を維持
				self.wait_count[i] += 1  # 待機回数加算
			# 利用可能な場合は観測を更新
			else:
				self.current_goal_prepare[i] = joint_action[i] # 行き先ノードを更新
				obs_i = self.obs[i]  # 現在観測
		
				# 現在の距離計算
				current_goal = list(self.pos[int(action_i)])  # 行き先座標
				current_x1,current_y1 = obs_i[0], obs_i[1]  # 現在座標
				x = current_goal[0] - current_x1  # x方向差分
				y = current_goal[1] - current_y1  # y方向差分
				dist_to_cgoal = np.sqrt(np.square(x) + np.square(y))# 行き先までの距離

				if dist_to_cgoal>self.speed:# エッジ上を移動
					current_x1 = round(current_x1+(self.speed*x/dist_to_cgoal), 2)  # 新しいx座標
					current_y1 = round(current_y1+(self.speed*y/dist_to_cgoal), 2)  # 新しいy座標
					obs_i_ = [round(current_x1,2), round(current_y1,2), obs_i[2], obs_i[3]]  # 新しい観測
					
					# one-hot状態の更新
					x = list(self.pos[self.current_start[i]])[0] - current_x1  # 出発点x差分
					y = list(self.pos[self.current_start[i]])[1] - current_y1  # 出発点y差分
					dist_to_cstart = np.sqrt(np.square(x) + np.square(y))# 出発点までの距離
					dist_to_cstart_rate = round(dist_to_cstart/(dist_to_cstart+dist_to_cgoal), 2)  # 割合
					
					self.obs_onehot_prepare[i] = np.zeros((1, len(list(self.G.nodes()))*2))  # one-hot初期化
					self.obs_onehot_prepare[i][int(action_i)] = dist_to_cstart_rate  # 行き先ノード
					self.obs_onehot_prepare[i][int(self.current_start[i])] = 1-dist_to_cstart_rate  # 出発ノード
					self.obs_onehot_prepare[i][int(self.goal_array[i])+len(list(self.G.nodes()))] = 1 # ゴール
					self.distance_from_start[i] += self.speed  # 移動距離加算
				# ノードに到着
				else:
					obs_i_ = [round(self.pos[int(action_i)][0],2), round(self.pos[int(action_i)][1],2), obs_i[2], obs_i[3]]  # 新しい観測
					
					# one-hot状態の更新
					self.obs_onehot_prepare[i] = np.zeros((1, len(list(self.G.nodes()))*2))  # one-hot初期化
					self.obs_onehot_prepare[i][int(action_i)] = 1  # 行き先ノード
					self.obs_onehot_prepare[i][int(self.goal_array[i])+len(list(self.G.nodes()))] = 1 # ゴール
					
					# ノード到着時のみcurrent_startを更新
					self.current_start_prepare[i] = int(action_i) # 出発ノードを更新
					self.current_goal_prepare[i] = None # 行き先ノードをリセット

					self.distance_from_start[i] += dist_to_cgoal  # 移動距離加算

				self.obs_prepare.append(obs_i_)  # 新しい観測を追加
		
		# 衝突判定
		collision_flag = self.ee_env.collision_detect(self.obs_prepare)  # 衝突判定
		info = {
			"goal": False,  # ゴール到達フラグ
			"collision": False,  # 衝突フラグ
			"timeup": False, # epymarl用
			"distance_from_start": None,  # スタートからの距離
			"step": self.step_account,  # ステップ数
			"wait": list(self.wait_count),  # 待機回数
		}
		# 衝突発生時
		if collision_flag==1:#collision
			collision_reward = self.r_coll*self.speed  # 衝突報酬
			if self.collision == "bounceback":
				self.terminated = [False for _ in range(self.agent_num)]  # 終了しない
			else: # デフォルトはterminated
				self.terminated = [True for _ in range(self.agent_num)]  # 全員終了
			info["collision"] = True  # 衝突フラグ
			obs = self.obs_manager.calc_obs()  # 状態表現計算
			ri_array = [collision_reward for _ in range(self.agent_num)]  # 報酬配列
			
		# 衝突なし
		else: #non collision
			self.obs = tuple([np.array(i) for i in self.obs_prepare])  # 観測更新
			self.obs_onehot = copy.deepcopy(self.obs_onehot_prepare)  # one-hot観測更新
			self.current_start = copy.deepcopy(self.current_start_prepare)   # スタートノード更新
			self.current_goal = copy.deepcopy(self.current_goal_prepare)  # ゴールノード更新

			team_reward = 0  # チーム報酬
			ri_array = []  # 報酬配列
			for i in range(self.agent_num):
				ri = self.reward(i)  # 報酬計算
				team_reward += ri  # チーム報酬加算
				ri_array.append(ri)  # 報酬配列に追加
			
			if self.terminated == [True for _ in range(self.agent_num)]: # 全員ゴール到達
				print("!!!all reach goal!!!")  # デバッグ出力
				self.reach_account = 0  # ゴール到達数リセット
				info["goal"] = True  # ゴールフラグ
			
			else:
				pass  # 何もしない

			obs = self.obs_manager.calc_obs()  # 状態表現計算

		# 時間切れ判定
		if self.step_account >= self.time_limit:
			print("!!!time up!!!")  # デバッグ出力
			info["timeup"]= True  # タイムアップフラグ
			self.terminated = [True for _ in range(self.agent_num)]  # 全員終了

		info["distance_from_start"] = list(self.distance_from_start)  # スタートからの距離

		return obs, ri_array, self.terminated, info  # 観測・報酬・終了・情報を返す

	# 報酬計算
	def reward(self, i):
		pre_pos_agenti = [self.obs_current_chache[i][0], self.obs_current_chache[i][1]]
		pos_agenti = [self.obs[i][0], self.obs[i][1]]
		goal_pos = self.pos[self.goal_array[i]]
		
		# ゴールまでの距離の変化を計算
		prev_dist_to_goal = np.sqrt(np.sum(np.square(np.array(pre_pos_agenti) - np.array(goal_pos))))
		curr_dist_to_goal = np.sqrt(np.sum(np.square(np.array(pos_agenti) - np.array(goal_pos))))
		dist_reward = (prev_dist_to_goal - curr_dist_to_goal) * self.speed  # 距離の減少に対する報酬
		
		# 基本報酬の計算（既存のロジック）
		if str(pos_agenti) == str(goal_pos):
			if pre_pos_agenti != pos_agenti:
				r_i = self.r_goal
				self.reach_account += 1
				self.terminated[i] = True
			else:
				r_i = 0
		else:
			if pre_pos_agenti == pos_agenti:
				r_i = self.r_wait * self.speed
			else:
				r_i = self.r_move * self.speed

		return r_i + dist_reward  # 基本報酬と距離報酬の合計

	# 環境の描画
	def render(self, mode='human'):
		self.ee_env.plot_map_dynamic(
			self.visu_delay,self.obs_current_chache,  # 可視化ディレイ・前回観測
			self.obs,self.goal_array,  # 現在観測・ゴール配列
			self.agent_num,  # エージェント数
			self.current_goal,  # 現在のゴール
			self.reach_account,  # ゴール到達数
			self.step_account,  # ステップ数
			self.episode_account  # エピソード数
		) # a must be a angle !!!list!!!

	# 環境のクローズ処理
	def close(self):
		print('Environment CLOSE')  # 終了メッセージ
		return None  # 何も返さない

	# 各エージェントの位置情報を取得
	def get_pos_list(self):
		pos_list = []  # 位置リスト
		all_onehot_obs = np.array(self.obs_onehot)  # one-hot観測
		onehot_obs = all_onehot_obs[:, :self.n_nodes]  # ノード部分のみ

		# 全エージェントの状態と位置を取得
		for i, obs_i in enumerate(onehot_obs):
			edge_or_node = tuple([i for i, o in enumerate(obs_i) if o!=0])  # ノードorエッジ判定
			if len(edge_or_node)==1:
				node = edge_or_node[0]  # ノード
				pos = {"type": "n", "pos": node}  # ノード情報
				obs_i = np.array(obs_i)*self.agent_num  # スケーリング
			else:
				edge = edge_or_node  # エッジ
				pos = {"type": "e", "pos": edge, "current_goal": self.current_goal[i], "current_start": self.current_start[i], "obs": obs_i}  # エッジ情報
			pos_list.append(pos)  # リストに追加

		return pos_list  # 位置リストを返す