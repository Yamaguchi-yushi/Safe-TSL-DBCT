import gym
import numpy as np
import sys
import copy
import os
import time
import torch
from torch import nn

from drp_env.state_repre import REGISTRY
from drp_env.EE_map import MapMake
from epymarl.src.utils.util import make_train_step
from epymarl.src.utils.replay_memory import ReplayMemory_episode
from reward_model.mard.mard import STAS

class DrpEnv(gym.Env):
	def __init__(self,
			agent_num,
			speed,
			start_ori_array,
			goal_array,
			visu_delay,
			state_repre_flag,
			time_limit,
			collision,
			map_name="map_3x3",
			reward_list={"goal": 100, "collision": -10, "wait": -10, "move": -1},
			use_lare_reward = False,			# LaReを学習に使うか
			use_lare_training = False,			# Falseの場合はLARE報酬でNN学習、Q値は従来報酬で学習、Trueの場合はLARE報酬でNN学習、Q値もLARE報酬で学習
			use_pretrained_model = False,		# 事前学習モデルを使うか
			pretrained_model_name = "_QMIX_LARE_map_aoba00_4agents_2.0M_checkpoint.pth",	# 事前学習モデルのパス
			use_separete_memory = False,			# 分離メモリを使うか
			show_debug_logs = False,			# デバッグログをコンソールに表示するか（Trueで表示、Falseで非表示）
			use_finetuning = False,			# 事前学習モデルを追加学習するか
			finetuning_model_name = "Safe_QMIX_LARE_map_8x5_2agents_1.1M_final.pth",		# 追加学習に使う事前学習モデルのパス
		  ):
		
		self.agent_num = agent_num
		self.n_agents = agent_num # for epymarl
		self.state_repre_flag = state_repre_flag
		self.map_name = map_name
		self.speed = speed
		self.visu_delay = visu_delay
		self.start_ori_array = start_ori_array
		self.goal_array = goal_array
		self.use_lare_reward = use_lare_reward
		self.use_lare_training = use_lare_training
		self.use_pretrained_model = use_pretrained_model
		self.pretrained_model_name = pretrained_model_name
		self.use_separete_memory = use_separete_memory

		self.show_debug_logs = show_debug_logs

		self.use_finetuning = use_finetuning
		self.finetuning_model_name = finetuning_model_name
		
		# reward
		self.r_goal = reward_list["goal"]
		self.r_coll = reward_list["collision"]
		self.r_wait = reward_list["wait"]
		self.r_move = reward_list["move"]

		# collision mechanism
		self.collision = collision

		self.time_limit = time_limit
		self.colli_distan_value = 5.0
		self.r_flag = 0
		self.flag_indicate = 0
		self.episode_account = 0
		self.total_step_account = 0

		self.distance_from_start = np.zeros(self.agent_num)

		self.agent_arrival_steps = np.full(self.agent_num, -1, dtype=int)
		self.episode_cost = 0

		self.reward_norm = False

		# create ee_env and pass self.variable
		self.ee_env = MapMake(self.agent_num, self.start_ori_array, self.goal_array, self.map_name)
		self.pos = self.ee_env.pos
		self.start_ori_array = self.ee_env.start_ori_array
		self.goal_array = self.ee_env.goal_array
		self.G = self.ee_env.G
		self.edge_labels = self.ee_env.edge_labels # unused

		self.current_goal  = [ None for i in range(self.agent_num)]

		self.obs_manager = REGISTRY[self.state_repre_flag](self)

		# create gym-like mdp elements
		self.n_nodes = len(self.G.nodes)
		self.n_actions = self.n_nodes
		self.action_space = gym.spaces.Tuple(tuple([gym.spaces.Discrete(self.n_nodes)] * self.agent_num))
		
		obs_box = self.obs_manager.get_obs_box()
		self.observation_space = gym.spaces.Tuple(tuple([obs_box] * self.agent_num))
		

		if self.use_lare_reward:
			print("✅ Using LARE reward decomposition system.")
			if self.use_lare_training:
				print("🧠 LARE rewards will be used for training.")
			else:
				print("📊 LARE rewards will be calculated but NOT used for training.")
				print("🎓 LARE neural network will still be trained and saved.")
			self.initialize_lare_system()
	
	def initialize_lare_system(self):
		"""Initialize the LARE system for reward decomposition."""
		try:
			# Import the LARE reward decomposition system
			from reward_model.LLMrd.factor_reward_decompose import FactorRewardDecomposer
			import_path = os.path.join(os.path.dirname(__file__), 'reward_model', 'factor_reward_decomposer.py')
			print(f"🔧 Importing FactorRewardDecomposer from {import_path}")

			self.edge_info_cache = self._precompute_edge_info()

			self.graph_diameter = self._compute_graph_diameter()

			class LaReArgs:
				def __init__(self, env_instance):
					self.n_agents = env_instance.agent_num
					self.env_name = "drp"
					self.map_name = env_instance.map_name
					self.seed = 42
					n_edges = len(env_instance.G.edges())
					# calculate the dimension of the observation space
					self.obs_dim = (env_instance.n_nodes * 2 + # agent's own location(One-hot)
									env_instance.n_nodes + # agent's goal location(One-hot)
									(env_instance.agent_num - 1) * env_instance.n_nodes * 2 + # other agents' locations(One-hot)
									1 + # collision distance
									2 + # collision information [self_involved, others_exist]
									1 + # wait count
									env_instance.n_nodes * 2 + # node corrdinates [x, y]
									n_edges * 3 + # edge information [from, to, distance]
									1 + # graph diameter
									1 + # node_num
									1 + # agent_num
									1   # edge_num
									)  
					
					# LARE specific parameters
					self.only_s = False 				# whether to use only state representation
					self.llm_response_dir = "responses" # directory to save LLM responses
					self.llm_n = 1 						# number of reward functions to generate
					self.factor_reward_model_layers = 3 # number of layers in the factor reward model
					self.port = 8080 					# port for the LLM server
					self.obs_agent_id = False 			# whether to include agent ID in the observation
					self.use_next_state = False			# whether to use next state in the observation
					self.expected_factors = None 		# expected number of factors in the reward decomposition

					#DRP specific parameters
					self.pos = env_instance.pos							# dictionary of node positions
					self.G = env_instance.G 							# graph structure
					self.n_nodes = env_instance.n_nodes 				# number of nodes in the graph
					self.goal_array = env_instance.goal_array 			# goal array for each agent
					self.start_ori_array = env_instance.start_ori_array # start orientation array for each agent
					self.collision_dis = env_instance.colli_distan_value # collision distance threshold
					self.speed = env_instance.speed 					# speed of the agents

			args = LaReArgs(self)
			print(f"🔧 Creating FactorRewardDecomposer with args...")
			print(f"   - n_agents: {args.n_agents}")
			print(f"   - obs_dim: {args.obs_dim}")
			print(f"   - env_name: {args.env_name}")
			print(f"   - import_path: {import_path}")
			
			self.lare_decompose = FactorRewardDecomposer(args)

			if self.use_separete_memory:
				self.buffer_size = 512
				self.goal_memory = ReplayMemory_episode(self.buffer_size, self.time_limit, self.reward_norm)
				self.collision_memory = ReplayMemory_episode(self.buffer_size, self.time_limit, self.reward_norm)
				self.timeup_memory = ReplayMemory_episode(self.buffer_size, self.time_limit, self.reward_norm)
				print("f✅ Using separate replay memories for each reward factor.")
			else:
				self.buffer_size = 1024
				self.memory_e = ReplayMemory_episode(self.buffer_size, self.time_limit, self.reward_norm)
				print("✅ Using single replay memory for all reward factors.")

			self.current_state = None 			# current state for the LARE system
			self.reward_model_update_freq = 256 # frequency of reward model updates
			self.evaluation_episodes = 16 		# number of episodes to evaluate the reward model
			self.current_evaluation_count = 0 	# current evaluation counter
			self.evaluation_base_episode = None # base episode number for evaluation
			self.is_evaluation_period = False   # evaluation period flag
			self.batch_size = 2048 				# batch size for the first reward model updates
			self.rewardbatch_size = 256 		# batch size for the reward model updates
			self.reward_model_starts = 256 		# minimum samples required to start training the reward model
			self.reward_model = self.lare_decompose # initialize the reward model
			loss_fn = nn.MSELoss(reduction='mean') # loss function for the reward model
			lr_reward = 5e-4 # learning rate for the reward model
			opt = torch.optim.Adam(params=self.reward_model.parameters(), lr=lr_reward, weight_decay=1e-5)
			self.device = 'cpu'
			self.train_step = make_train_step(self.reward_model,
												loss_fn,
												opt,
												self.n_agents,
												self.device,
												env = self,
												reg=False,		# 正規化するかどうか
												alpha=0.0)		# 正規化の強さ

			self.max_train_steps = self.load_max_steps_from_config()
			self.training_completed = False
			self.training_start_time = time.time()
			self.checkpoint_saved = False  # チェックポイント保存済みフラグ
			self.total_update_count = 0

			self.episode_data = {
				"x_e" : [],
				"action_e": [],
				"mask_e" : [],
				"reward_e" : [],
			}

			actual_factors = getattr(self.lare_decompose, 'factor_num', None)
			if actual_factors is not None:
				print(f"✅ LARE system initialized successfully with {actual_factors} factors!")
				print(f"🔧 LARE system parameters: obs_dim={args.obs_dim}, n_agents={args.n_agents}")

				if self.use_pretrained_model:
					print(f"🔍  [PRETRAINED] Attempting to load pretrained reward model...")
					load_success = self.load_pretrained_lare_model()

					if load_success:
						print(f"✅  [PRETRAINED] Successfully loaded pretrained reward model.")
						print(f"ℹ️  [PRETRAINED] Model will be used for inference only (no training)")
					else:
						print(f"❌  [PRETRAINED] Failed to load pretrained reward model.")
						self.use_pretrained_model = False
				elif self.use_finetuning:
					print(f"🔍  [FINETUNE] Attempting to load finetuning model from: {self.finetuning_model_name}")
					load_success = self.load_finetuning_lare_model()

					if load_success:
						print(f"✅  [FINETUNE] Successfully loaded finetuning model.")
						print(f"ℹ️  [FINETUNE] Model will be further trained.")
					else:
						print(f"❌  [FINETUNE] Failed to load finetuning model.")
						self.use_finetuning = False					
				else:
					print(f"ℹ️  [PRETRAINED] Starting fresh training (use_pretrained_model=False)")
			else:
				print("❌ LARE system initialization failed: 'factor_num' attribute not found.")
				print(f"🔧 LARE system parameters: obs_dim={args.obs_dim}, n_agents={args.n_agents}")
		
		except ImportError as e:
			print(f"❌ ImportError: {e}")
			print(f"🔍 sys.path contains: {[p for p in sys.path if 'reward_model' in p or 'LLMrd' in p]}")
			print("📋 Falling back to traditional reward system.")
			self.use_lare_reward = False
			self.lare_decompose = None
		
		except Exception as e:
			print(f"❌ Error initializing LARE system: {e}")
			import traceback
			print(f"🔍 Full traceback: {traceback.format_exc()}")
			print("📋 Falling back to traditional reward system.")
			self.use_lare_reward = False
			self.lare_decompose = None
	
	def load_pretrained_lare_model(self):
		"""
		Attempts to load a pretrained LARE reward model from a specified path.
		
		Returns:
			bool: True if the model was loaded successfully, False otherwise.
		"""
		try:
			base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
			save_dir = os.path.join(base_dir, "epymarl", "src", "saved_models")

			if self.pretrained_model_name is None:
				print(f"❌  [PRETRAINED] pretrained_model_name is not specified.")
				sys.exit(1)

			model_name = self.pretrained_model_name
			if not model_name.endswith(".pth"):
				model_name += ".pth"
			model_path = os.path.join(save_dir, model_name)
			self.pretrained_model_name = model_path

			# ファイルの存在確認
			if not os.path.exists(model_path):
				print(f"❌  [PRETRAINED] Model file not found at {model_path}")
				sys.exit(1)
			
			# モデルのロード
			print(f"🔍  [PRETRAINED] Loading model from {model_path}...")
			checkpoint = torch.load(model_path, map_location=self.device)

			# モデルの重みをロード
			self.lare_decompose.load_state_dict(checkpoint['model_state_dict'])
			self.lare_decompose.eval()  # Set model to evaluation mode

			print(f"✅  [PRETRAINED] Model loaded successfully")
			file_size_kb = os.path.getsize(model_path) / 1024
			print(f"ℹ️  [PRETRAINED] Model file size: {file_size_kb:.2f} KB")

			return True
		
		except KeyError as e:
			print(f"❌  [PRETRAINED] Error loading pretrained model: {e}")
			print(f"🔍  [PRETRAINED] This may happen if the model archtecture has changed.")
			print(f"💡 [PRETRAINED] Try training from scratch or use a compatible model.")
			sys.exit(1)

		except Exception as e:
			print(f"❌  [PRETRAINED] Error loading pretrained model: {e}")
			import traceback
			print(f"🔍  [PRETRAINED] Full traceback: {traceback.format_exc()}")
			sys.exit(1)

	def load_finetuning_lare_model(self):
		"""
		ファインチューニング用にモデルをロードする
		事前学習モデルの重みをロードし、学習を継続できる状態にする

		Returns:
			bool: True if the model was loaded successfully, False otherwise.
		"""
		try:
			base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
			save_dir = os.path.join(base_dir, "epymarl", "src", "saved_models")

			if self.finetuning_model_name is None:
				print(f"❌  [FINETUNE] finetuning_model_name is not specified.")
				sys.exit(1)


			model_name = self.finetuning_model_name

			if not model_name.endswith(".pth"):
				model_name += ".pth"

			model_path = os.path.join(save_dir, model_name)
			self.finetuning_model_path = model_path

			if not os.path.exists(model_path):
				print(f"❌  [FINETUNE] Model file not found at {model_path}")
				sys.exit(1)
			
			print(f"🔍  [FINETUNE] Loading model from {model_path}...")
			checkpoint = torch.load(model_path, map_location=self.device)

			self.lare_decompose.load_state_dict(checkpoint['model_state_dict'])
			self.lare_decompose.train()  # Set model to train mode for finetuning

			if 'experiment_info' in checkpoint:
				exp_info = checkpoint['experiment_info']
				print(f"ℹ️  [FINETUNE] Experiment info:")
				print(f"   - Algorithm: {exp_info.get('algorithm_name', 'unknown')}")
				print(f"   - Map: {exp_info.get('map_name', 'unknown')}")
				print(f"   - Agents: {exp_info.get('n_agents', 'unknown')}")

			if 'episode_data' in checkpoint:
				print(f"   - Original episodes: {checkpoint['episode_data']}")
			if 'total_training_steps' in checkpoint:
				print(f"   - Original steps: {checkpoint['total_training_steps']}")

			file_size_kb = os.path.getsize(model_path) / 1024
			print(f"✅  [FINETUNE] Model file size: {file_size_kb:.2f} KB")
			print(f"    [FINETUNE] Model set to trainable state for finetuning.")

			return True
		
		except KeyError as e:
			print(f"❌  [FINETUNE] Error loading finetuning model: {e}")
			print(f"🔍  [FINETUNE] This may happen if the model archtecture has changed.")
			print(f"💡 [FINETUNE] Try training from scratch or use a compatible model.")
			sys.exit(1)
		
		except Exception as e:
			print(f"❌  [FINETUNE] Error loading finetuning model: {e}")
			import traceback
			print(f"🔍  [FINETUNE] Full traceback: {traceback.format_exc()}")
			sys.exit(1)

	def load_max_steps_from_config(self):
		"""
		gymma.yamlファイルからt_max値を読み込む
		
		Returns:
			int: 最大ステップ数（デフォルト: 2050000）
		"""
		try:
			import yaml
			
			# gymma.yamlファイルのパスを構築
			config_path = os.path.join(os.path.dirname(__file__), '..', 'epymarl', 'src', 'config', 'envs', 'gymma.yaml')
			
			# ファイルが存在しない場合のフォールバック
			if not os.path.exists(config_path):
				print(f"⚠️ [CONFIG] gymma.yaml not found at {config_path}")
				print(f"🔧 [CONFIG] Using default max_training_steps: 2050000")
				return 2050000
			
			# YAMLファイルを読み込み
			with open(config_path, 'r') as file:
				config = yaml.safe_load(file)
			
			# t_max値を取得
			t_max = config.get('t_max', 2050000)
			
			print(f"✅ [CONFIG] Loaded t_max from gymma.yaml: {t_max}")
			return int(t_max)
			
		except ImportError:
			print(f"⚠️ [CONFIG] PyYAML not available. Using default max_training_steps: 2050000")
			return 2050000
		except Exception as e:
			print(f"❌ [CONFIG] Error loading gymma.yaml: {e}")
			print(f"🔧 [CONFIG] Using default max_training_steps: 2050000")
			return 2050000

	def get_node_coordinates_flat_array(self):
		"""		
		Returns a flat array of node coordinates in the format [x0, y0, x1, y1, x2, y2, ...].
    
   	 	Returns:
        	np.array: [x0, y0, x1, y1, x2, y2, ...]
    	"""
		coordinates = []
		sorted_nodes = sorted(self.pos.keys())
    
		for node_id in sorted_nodes:
			x, y = self.pos[node_id]
			coordinates.extend([float(x), float(y)])
    
		return np.array(coordinates)

	def _precompute_edge_info(self):
		"""
		初期化時にエッジ情報を前計算してキャッシュする
		Returns:
			np.array: [node1_0, node2_0, distance_0, node1_1, node2_1, distance_1, ...]
			shape=(n_edges * 3,)
		"""
		edge_info = []

		sorted_edges = sorted(self.G.edges())

		for edge in sorted_edges:
			node1, node2 = edge
			
			pos1 = self.pos[node1]
			pos2 = self.pos[node2]
			edge_distance = np.linalg.norm(np.array(pos1) - np.array(pos2))

			edge_info.extend([int(node1), int(node2), float(edge_distance)])
		
		return np.array(edge_info)
	
	def _compute_graph_diameter(self):
		"""
		グラフの直径（最長最短経路長）を計算する
		Returns:
			float: グラフの直径
		"""
		
		import heapq

		n_nodes = len(self.G.nodes())

		def dijkstra(graph, start, goal):
			"""
			ダイクストラ法による
			"""
			if start == goal:
				return 0.0
			
			dist = {node: float('inf') for node in graph.nodes()}
			dist[start] = 0.0
			pq = [(0.0, start)]
			visited = set()

			while pq:
				d, node = heapq.heappop(pq)

				if node in visited:
					continue
				visited.add(node)

				if node == goal:
					return d
				
				if d > dist[node]:
					continue

				for neighbor in graph.neighbors(node):
					if neighbor in visited:
						continue

					edge_data = graph.get_edge_data(node, neighbor)
					if edge_data and 'weight' in edge_data:
						weight = edge_data['weight']
					else:
						pos1 = self.pos[node]
						pos2 = self.pos[neighbor]
						weight = np.linalg.norm(np.array(pos1) - np.array(pos2))

					nd = d + weight
					if nd < dist[neighbor]:
						dist[neighbor] = nd
						heapq.heappush(pq, (nd, neighbor))
			return float('inf')
		
		print("🔧 Computing graph diameter...")
		max_distance = 0.0
		valid_distances = []

		for i in range(n_nodes):
			for j in range(i + 1, n_nodes):
				dist = dijkstra(self.G, i, j)
				if dist < float('inf'):
					valid_distances.append(dist)
					if dist > max_distance:
						max_distance = dist

		if len(valid_distances) > 0:
			diameter = max(max_distance, 1.0)
			print(f"✅ Graph diameter computed: {diameter:.2f}")
			print(f"  - Max distance: {max_distance:.2f}")
			print(f"ℹ️ Number of valid node pairs: {len(valid_distances)}")

			self.cached_graph_diameter = diameter
			return diameter
		else:
			print("⚠️ No valid paths found between any node pairs. Setting diameter to 1.0")
			return 100.0

	
	def get_all_edge_info_flat(self):
		"""
		キャッシュされた全エッジ情報をフラットな配列として返す
		"""
		if not hasattr(self, 'edge_info_cache'):
			print("⚠️ Edge info cache not found, Returning empty array")
			return np.array([])
		
		return self.edge_info_cache
	
	def _get_lare_compatible_obs(self, agent_id):
		"""
		Returns an observation compatible with the LARE system for the specified agent.
		
		Args:
			agent_id (int): The ID of the agent for which to get the observation.
		
		Returns:
			np.array: An observation array compatible with the LARE system.
		"""
		# 1. Agent's own position (One-hot)
		agent_location = self._extract_agent_location_from_onehot(agent_id)
		
		# 2. Agent's goal position (One-hot)
		agent_goal = self._extract_agent_goal_from_onehot(agent_id)
		
		# 3. Absolute positions of other agents (One-hot)
		other_agents_locations = self._get_other_agents_absolute_from_onehot(agent_id)
		
		# 4. Collision distance
		collision_distance = np.array([self.colli_distan_value])

		# 5. Collision information
		if hasattr(self, 'current_colliding_pairs') and self.current_colliding_pairs is not None:
			# Convert the colliding pairs to agent-specific information
			collision_info = self._convert_colliding_pairs_to_agent_info(self.current_colliding_pairs, agent_id)
			collision_info_flat = collision_info.flatten().astype(float)
		else:
			# フォールバック: 衝突情報がない場合はゼロで初期化
			collision_info_flat = np.zeros(2, dtype=float)
		
		# 6. Wait count
		wait_count = np.array([self.wait_count[agent_id]])

		# 7. Node coordinates (flat array)
		node_coordinates = self.get_node_coordinates_flat_array()

		# 8. Edge information (flat array)
		# edge_info = self.get_all_edge_info_flat()		edge_info = n
		edge_info = self.edge_info_cache

		# 9. Graph diameter (single value)
		graph_diameter = np.array([self.graph_diameter])

		# 10. node_num, agent_num, edge_num
		node_num = self.n_nodes
		agent_num = self.agent_num
		edge_num = len(self.G.edges())

		# 各コンポーネントを1次元配列に変換
		agent_location_flat = agent_location.flatten()
		agent_goal_flat = agent_goal.flatten()
		other_agents_flat = other_agents_locations.flatten()
		collision_distance_flat = collision_distance.flatten()
		node_coordinates_flat = node_coordinates.flatten()
		edge_info_flat = edge_info.flatten()

		meta_info = np.array([node_num, agent_num, edge_num], dtype=int)
		
		# Combine all components into a single observation array
		obs = np.concatenate([
			agent_location_flat,
			agent_goal_flat,
			other_agents_flat,
			collision_distance_flat,
			collision_info_flat,
			wait_count,
			node_coordinates_flat,
			edge_info_flat,
			graph_diameter,
			meta_info
		])
		
		return obs
	
	def _extract_agent_location_from_onehot(self, agent_id):
		"""
		エージェントの位置を抽出し、[前の状態, 現在の状態]として結合したnode_num*2次元のフラットなベクトルを返す
		
		Args:
			agent_id (int): エージェントID
		
		Returns:
			np.array: [前の状態(node_num次元), 現在の状態(node_num次元)]の結合ベクトル (node_num*2次元)
		"""
		try:
			# 🔍 現在の状態を取得 (node_num次元) - obs_onehotから直接取得
			current_location = np.zeros(self.n_nodes)
			
			if hasattr(self, 'obs_onehot') and self.obs_onehot is not None:
				# 🔧 obs_onehotの形状を正規化（2D → 1D）
				if len(self.obs_onehot[agent_id].shape) == 2:
					# (1, node_num*2) → (node_num*2,) に変換
					agent_onehot_flat = self.obs_onehot[agent_id].flatten()
				else:
					# (node_num*2,) → そのまま
					agent_onehot_flat = self.obs_onehot[agent_id]
				
				# 位置情報を抽出（最初のn_nodes要素）
				if len(agent_onehot_flat) >= self.n_nodes:
					current_location = agent_onehot_flat[:self.n_nodes].copy()
			
			# 🔍 前の状態を位置キャッシュから取得 (node_num次元)
			previous_location = np.zeros(self.n_nodes)
			
			if hasattr(self, 'obs_onehot_position_cache') and self.obs_onehot_position_cache is not None:
				# 位置キャッシュから直接取得
				previous_location = self.obs_onehot_position_cache[agent_id].copy()
			else:
				# フォールバック: 現在の状態をコピー
				previous_location = current_location.copy()
			
			# 🔗 [前の状態, 現在の状態] を結合してフラットなnode_num*2次元ベクトルを作成
			combined_location = np.concatenate([previous_location, current_location])
			
			return combined_location
			
		except Exception as e:
			print(f"❌ [DEBUG] Error extracting combined location for agent {agent_id}: {e}")
			
			# 緊急フォールバック
			fallback_location = np.zeros(self.n_nodes)
			if hasattr(self, 'current_start') and self.current_start[agent_id] is not None:
				current_node = self.current_start[agent_id]
				if 0 <= current_node < self.n_nodes:
					fallback_location[current_node] = 1.0
			
			combined_fallback = np.concatenate([fallback_location, fallback_location])
			return combined_fallback
	
	def _extract_agent_goal_from_onehot(self, agent_id):
		"""
		Extracts the agent's goal position from the one-hot encoded observation.
		
		Args:
			agent_id (int): The ID of the agent.
		
		Returns:
			np.array: One-hot encoded goal position of the agent.
		"""
		try:
			agent_goal = np.zeros(self.n_nodes)
			
			if hasattr(self, 'obs_onehot') and self.obs_onehot is not None:
				# obs_onehotが(agent_num, node_num*2)の形状の場合
				# 最初のnode_num要素がエージェント位置、次のnode_num要素がゴール位置
				if len(self.obs_onehot[agent_id].shape) == 1 and len(self.obs_onehot[agent_id]) >= self.n_nodes*2:
					goal_part = self.obs_onehot[agent_id][self.n_nodes:self.n_nodes*2]
					agent_goal = goal_part.copy()
				else:
					# 2D配列の場合
					if len(self.obs_onehot[agent_id].shape) == 2:
						goal_part = self.obs_onehot[agent_id][0][self.n_nodes:self.n_nodes*2]
						agent_goal = goal_part.copy()
			else:
				# フォールバック: obsからゴール位置を取得
				if hasattr(self, 'obs') and self.obs is not None and len(self.obs) > agent_id:
					goal_node = int(self.obs[agent_id][3])  # goal position
					if 0 <= goal_node < self.n_nodes:
						agent_goal[goal_node] = 1.0
				elif hasattr(self, 'goal_array') and self.goal_array[agent_id] is not None:
					goal_node = self.goal_array[agent_id]
					if 0 <= goal_node < self.n_nodes:
						agent_goal[goal_node] = 1.0
			
			return agent_goal
			
		except Exception as e:
			print(f"❌ [DEBUG] Error extracting agent goal for agent {agent_id}: {e}")
			import traceback
			print(f"🔍 [DEBUG] Full traceback: {traceback.format_exc()}")
			
			# 緊急フォールバック
			agent_goal = np.zeros(self.n_nodes)
			if hasattr(self, 'goal_array') and self.goal_array[agent_id] is not None:
				goal_node = self.goal_array[agent_id]
				if 0 <= goal_node < self.n_nodes:
					agent_goal[goal_node] = 1.0
			
			return agent_goal
		
	def _get_other_agents_absolute_from_onehot(self, agent_id):
		"""
		Extracts the absolute positions of other agents from the one-hot encoded observation.
		
		Args:
			agent_id (int): The ID of the agent.
		
		Returns:
			np.array: One-hot encoded absolute positions of other agents.
		"""
		other_agents_locations = []

		for other_id in range(self.agent_num):
			if other_id != agent_id:
				other_location = self._extract_agent_location_from_onehot(other_id)
				other_agents_locations.append(other_location)
		
		return np.array(other_agents_locations)
	
	def _call_lare_reward_system(self, agent_id, next_obs=None, use_cache=False):
		"""
		Calls the LARE reward decomposition system to get the reward for the given observation.
		Also handles goal checking when LARE is active.
		
		Args:
			obs (np.array): The observation for which to calculate the reward.
			agent_id (int): The ID of the agent.
			next_obs (np.array, optional): The next observation.
		
		Returns:
			float: The calculated reward.
		"""
		if not self.use_lare_reward or not hasattr(self, 'lare_decompose') or self.lare_decompose is None:
			return None
		
		# LARE利用時のゴール到達判定
		if not self.terminated[agent_id]:
			pre_pos_agenti = [self.obs_current_chache[agent_id][0], self.obs_current_chache[agent_id][1]]
			pos_agenti = [self.obs[agent_id][0], self.obs[agent_id][1]]
			goal_pos = self.pos[self.goal_array[agent_id]]

			if str(pos_agenti) == str(goal_pos) and pre_pos_agenti != pos_agenti:
				self.reach_account += 1
				self.terminated[agent_id] = True
				self._record_agent_arrival(agent_id)
				print(f"  🎯 Agent {agent_id}: GOAL REACHED! (LARE mode, Total: {self.reach_account})")

		try:
			if use_cache and hasattr(self, 'current_state') and self.current_state is not None:
				current_obs = self.current_state.get(agent_id)
				if current_obs is None:
					current_obs = self._get_lare_compatible_obs(agent_id)
			else:
				# 現在の観測を取得
				current_obs = self._get_lare_compatible_obs(agent_id)
			
			# 次の観測が提供されていない場合は現在の観測を使用
			if next_obs is None:
				next_obs = current_obs
			
			# LaReシステムに観測を渡して報酬を取得
			lare_reward = self.lare_decompose.get_reward(
				obs=current_obs,
				next_obs=next_obs,
				debug_agent_id=agent_id,  # デバッグ用エージェントID
				debug_step=self.step_account  # デバッグ用ステップ数
			)

			return float(lare_reward)  # 単一の報酬を返す
			
		except Exception as e:
			if self.step_account <= 5:
				print(f"❌ LaRe reward system error for agent {agent_id}: {e}")
				import traceback
				print(f"🔍 Full traceback: {traceback.format_exc()}")
			return None

	def _convert_colliding_pairs_to_agent_info(self, colliding_pairs, agent_id):
		"""
		衝突ペアのリストから、特定エージェント用の衝突情報に変換します。
		
		Args:
			colliding_pairs (list): 衝突したエージェントのペアのリスト。例: [[0, 2], [3, 4]]
			agent_id (int): 対象エージェントのID
		
		Returns:
			np.array: [self_involved, others_exist] (2次元、int型)
					  - self_involved: 自身が衝突に関与しているか (1 or 0)
					  - others_exist: 自身が関与しない他の衝突が存在するか (1 or 0)
		"""
		self_involved = 0
		others_exist = 0

		if not colliding_pairs:
			return np.array([0, 0], dtype=int)

		for pair in colliding_pairs:
			if agent_id in pair:
				self_involved = 1
			else:
				# このペアに自分は含まれていない -> 他者間の衝突が存在する
				others_exist = 1
		
		return np.array([self_involved, others_exist], dtype=int)
	
	def perform_episode_update(self):
		"""メモリから256エピソードをサンプリングして1回の更新を実行"""
		if not self.use_lare_reward or not hasattr(self, 'lare_decompose') or self.lare_decompose is None:
			print("❌ LARE reward model is not available. Skipping episode update.")
			return

		try:
			if self.use_separete_memory:
				print(f" [SAMPLING] Using separate memory")

				goal_size = len(self.goal_memory)
				collision_size = len(self.collision_memory)
				timeup_size = len(self.timeup_memory)
				total_size = goal_size + collision_size + timeup_size

				if total_size < self.reward_model_starts:
					print(f"  ❌ [SAMPLING] Separate memory insufficient!")
					print(f"    - Available: {total_size} episodes (Goal: {goal_size}, Collision: {collision_size}, Timeup: {timeup_size})")
					print(f"    - Needed: {self.reward_model_starts}")
					print(f"  ⏭️ [SKIP UPDATE] Skipping this update - need {self.reward_model_starts - total_size} more episodes")
					return
				
				print(f"  ✅ [SAMPLING] Separate memory sufficient")
				print(f"    - Available: {total_size} episodes (Goal: {goal_size}, Collision: {collision_size}, Timeup: {timeup_size})")
				print(f"	- Sampling: {self.rewardbatch_size} episodes")

				if hasattr(self, 'current_termination_reason'):
					focus_category = self.current_termination_reason
					print(f"  🎯 [FOCUS] 60% from {focus_category}, 20% from others")
				else:
					category_size = {
						'goal': goal_size,
						'collision': collision_size,
						'timeup': timeup_size
					}
					focus_category = max(category_size, key=category_size.get)
					print(f"  ⚠️ [FALLBACK] No current termination reason. Focusing on largest category: {focus_category}")
				
				sample_weightsq = {
					'collision': 0.2,
					'timeup': 0.2,
					'goal': 0.2
				}
				sample_weightsq[focus_category] = 0.6

				n_collision = min(int(self.rewardbatch_size * sample_weightsq['collision']), collision_size)
				n_timeup = min(int(self.rewardbatch_size * sample_weightsq['timeup']), timeup_size)
				n_goal = min(int(self.rewardbatch_size * sample_weightsq['goal']), goal_size)

				total_sampled = n_collision + n_timeup + n_goal

				if total_sampled < self.rewardbatch_size:
					shortage = self.rewardbatch_size - total_sampled
					other_categories = [cat for cat in ['collision', 'goal', 'timeup'] if cat != focus_category]
					
					# 各カテゴリーの残り容量を計算
					remaining_capacity = {
						cat: {'collision': collision_size, 'goal': goal_size, 'timeup': timeup_size}[cat] - 
							 {'collision': n_collision, 'goal': n_goal, 'timeup': n_timeup}[cat]
						for cat in other_categories
					}
					
					total_remaining = sum(remaining_capacity.values())
					
					# 🔧 CASE A: 他カテゴリーの残りで足りる場合
					if total_remaining >= shortage:
						shortage_per_category = shortage // len(other_categories)
						extra = shortage % len(other_categories)
						
						filled = 0  # 実際に補充できた数
						for i, cat in enumerate(other_categories):
							target = shortage_per_category + (1 if i < extra else 0)
							additional = min(target, remaining_capacity[cat])
							
							if cat == 'collision': n_collision += additional
							elif cat == 'goal': n_goal += additional
							else: n_timeup += additional
							
							filled += additional
						
						# 🔧 まだ足りない場合はfocus_categoryから補充
						if filled < shortage:
							still_needed = shortage - filled
							focus_remaining = {'collision': collision_size, 'goal': goal_size, 'timeup': timeup_size}[focus_category] - \
											 {'collision': n_collision, 'goal': n_goal, 'timeup': n_timeup}[focus_category]
							final_fill = min(still_needed, focus_remaining)
							
							if final_fill > 0:
								if focus_category == 'collision': n_collision += final_fill
								elif focus_category == 'goal': n_goal += final_fill
								else: n_timeup += final_fill
					
					# 🔧 CASE B: 他カテゴリーの残りで足りない場合
					else:
						# 他カテゴリーから全部取る
						for cat in other_categories:
							additional = remaining_capacity[cat]
							if cat == 'collision': n_collision += additional
							elif cat == 'goal': n_goal += additional
							else: n_timeup += additional
						
						# 残りをfocus_categoryから補充
						still_needed = shortage - total_remaining
						focus_remaining = {'collision': collision_size, 'goal': goal_size, 'timeup': timeup_size}[focus_category] - \
										 {'collision': n_collision, 'goal': n_goal, 'timeup': n_timeup}[focus_category]
						final_fill = min(still_needed, focus_remaining)
						
						if final_fill > 0:
							if focus_category == 'collision': n_collision += final_fill
							elif focus_category == 'goal': n_goal += final_fill
							else: n_timeup += final_fill
				
				print(f"  📊 [SAMPLE] Collision={n_collision}, Goal={n_goal}, Timeup={n_timeup} (Total={n_collision+n_goal+n_timeup})")

				states_list = []
				actions_list = []
				return_list = []
				rewards_list = []
				length_list = []

				if n_collision > 0:
					s_states, s_actions, s_return, s_reward, s_length = \
						self.collision_memory.sample_trajectory(n_trajectories=n_collision)
					states_list.append(s_states)
					actions_list.append(s_actions)
					return_list.append(s_return)
					rewards_list.append(s_reward)
					length_list.append(s_length)
				if n_timeup > 0:
					s_states, s_actions, s_return, s_reward, s_length = \
						self.timeup_memory.sample_trajectory(n_trajectories=n_timeup)
					states_list.append(s_states)
					actions_list.append(s_actions)
					return_list.append(s_return)
					rewards_list.append(s_reward)
					length_list.append(s_length)
				if n_goal > 0:
					s_states, s_actions, s_return, s_reward, s_length = \
						self.goal_memory.sample_trajectory(n_trajectories=n_goal)
					states_list.append(s_states)
					actions_list.append(s_actions)
					return_list.append(s_return)
					rewards_list.append(s_reward)
					length_list.append(s_length)
				# リストを結合
				states = np.concatenate(states_list, axis=0)
				actions = np.concatenate(actions_list, axis=0)
				episode_return = np.concatenate(return_list, axis=0)
				episode_reward = np.concatenate(rewards_list, axis=0)
				episode_length = np.concatenate(length_list, axis=0)

			else:
			# メモリサイズをチェック
				memory_size = len(self.memory_e)
			
				if memory_size < self.reward_model_starts:
					print(f"  ❌ [SAMPLING] Unified memory insufficient!")
					print(f"    - Available: {memory_size} episodes")
					print(f"    - Needed: {self.reward_model_starts}")
					print(f"  ⏭️ [SKIP UPDATE] Skipping this update - need {self.reward_model_starts - memory_size} more episodes")
					return
							
				# メモリから256エピソードをサンプリング
				states, actions, episode_return, episode_reward, episode_length = \
					self.memory_e.sample_trajectory(n_trajectories=self.rewardbatch_size)
			
			# データをテンソルに変換
			if isinstance(states, torch.Tensor):
				states = states.clone().detach().float().to(self.device)
			else:
				states = torch.tensor(states, dtype=torch.float32).to(self.device)

			if isinstance(actions, torch.Tensor):
				actions = actions.clone().detach().float().to(self.device)
			else:
				actions = torch.tensor(actions, dtype=torch.float32).to(self.device)

			if isinstance(episode_return, torch.Tensor):
				episode_return = episode_return.clone().detach().float().to(self.device)
			else:
				episode_return = torch.tensor(episode_return, dtype=torch.float32).to(self.device)

			if isinstance(episode_length, torch.Tensor):
				episode_length = episode_length.clone().detach().float().to(self.device)
			else:
				episode_length = torch.tensor(episode_length, dtype=torch.float32).to(self.device)
						
			# 更新実行
			loss = self.train_step(states, actions, episode_return, episode_length)
			
		except Exception as e:
			print(f"❌ Error in episode update: {e}")
			import traceback
			print(f"🔍 Full traceback: {traceback.format_exc()}")

	def save_final_checkpoint(self):
		"""
		学習終了時に最終チェックポイントを保存
		
		Returns:
			str or None: 保存されたファイルパス
		"""
		try:
			# 保存ディレクトリの作成
			base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
			save_dir = os.path.join(base_dir, "epymarl", "src", "saved_models")
			os.makedirs(save_dir, exist_ok=True)
			
			algorithm_name = self._get_algorithm_name()  # configから取得
			map_name = getattr(self, 'map_name', 'unknown_map')  # マップ名
			agent_count = getattr(self, 'agent_num', 'unknown_agents')  # エージェント数
			step_in_millions = self.total_step_account / 1_000_000
			steps_str = f"{step_in_millions:.1f}M"

			is_safe = self.__class__.__name__ == "SafeEnv"
			safe_prefix = "Safe" if is_safe else ""
			source_base_name = None
			if self.use_finetuning:
				source_base_name = self._get_source_model_base_name()  # ファインチューニング元のベース名を取得
				filename = f"FT_{safe_prefix}_{source_base_name}_{map_name}_{agent_count}agents_{steps_str}_final.pth"
			# 最終モデルのファイル名（実験設定情報付き）
			else:
				filename = f"{safe_prefix}_{algorithm_name}_LARE_{map_name}_{agent_count}agents_{steps_str}_final.pth"
			
			save_path = os.path.join(save_dir, filename)
			
			# 学習時間の計算
			training_duration = time.time() - self.training_start_time
			
			# 最終チェックポイントデータ（必要最小限 + 実験設定情報）
			save_data = {
				'model_state_dict': self.lare_decompose.state_dict(),
				'episode_count': self.episode_account,
				'total_training_steps': self.total_step_account,
				'max_training_steps': self.max_train_steps,
				'training_duration_hours': training_duration / 3600,
				'training_completed': True,
				'save_timestamp': time.time(),
				# 🔧 NEW: 実験設定情報を追加
				'experiment_info': {
					'algorithm': algorithm_name,
					'map_name': map_name,
					'agent_count': agent_count,
					'time_limit': getattr(self, 'time_limit', None),
					'collision_type': getattr(self, 'collision', None),
					'reward_system': 'LARE' if self.use_lare_reward else 'Traditional',
					'use_lare_training': self.use_lare_training,
					'source_model': source_base_name,
				}
			}
			
			# 保存実行
			torch.save(save_data, save_path)
			
			# ファイルサイズの確認
			file_size_kb = os.path.getsize(save_path) / 1024
			
			print(f"✅ [FINAL SAVE] Final model saved successfully!")
			print(f"  - File: {filename}")
			print(f"  - Algorithm: {algorithm_name}")
			if self.use_finetuning:
				print(f"  - Fine-tuned from: {source_base_name}")
			print(f"  - Map: {map_name}")
			print(f"  - Agents: {agent_count}")
			print(f"  - Size: {file_size_kb:.1f} KB")
			print(f"  - Episodes: {self.episode_account}")
			print(f"  - Total steps: {self.total_step_account}/{self.max_train_steps}")
			print(f"  - Training time: {training_duration/3600:.1f} hours")
			
			return save_path
			
		except Exception as e:
			print(f"❌ Error saving final checkpoint: {e}")
			return None
		
	def save_checkpoint(self):
		"""
		定期的にチェックポイントを保存
		
		Returns:
			str or None: 保存されたファイルパス
		"""
		if not self.use_lare_reward or not hasattr(self, 'lare_decompose') or self.lare_decompose is None:
			print("❌ LARE reward model is not available. Skipping checkpoint save.")
			return None
		try:
			base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
			save_dir = os.path.join(base_dir, "epymarl", "src", "saved_models")
			os.makedirs(save_dir, exist_ok=True)

			algorithm_name = self._get_algorithm_name()  # configから取得
			map_name = getattr(self, 'map_name', 'unknown_map')  # マップ
			steps_in_millions = self.total_step_account / 1_000_000
			steps_str = f"{steps_in_millions:.1f}M"

			is_safe = self.__class__.__name__ == "SafeEnv"
			safe_prefix = "Safe" if is_safe else ""
			source_base_name = None
			if self.use_finetuning:
				source_base_name = self._get_source_model_base_name()  # ファインチューニング元のベース名を取得
				file_name = f"FT_{safe_prefix}_{source_base_name}_{map_name}_{self.agent_num}agents_{steps_str}_checkpoint.pth"
			else:
				file_name = f"{safe_prefix}_{algorithm_name}_LARE_{map_name}_{self.agent_num}agents_{steps_str}_checkpoint.pth"
			
			save_path = os.path.join(save_dir, file_name)

			training_duration = time.time() - self.training_start_time
			save_data = {
				'model_state_dict': self.lare_decompose.state_dict(),
				'episode_count': self.episode_account,
				'total_training_steps': self.total_step_account,
				'max_training_steps': self.max_train_steps,
				'training_duration_hours': training_duration / 3600,
				'training_completed': self.training_completed,
				'save_timestamp': time.time(),
				'experiment_info': {
					'algorithm': algorithm_name,
					'map_name': map_name,
					'agent_count': self.agent_num,
					'time_limit': getattr(self, 'time_limit', None),
					'collision_type': getattr(self, 'collision', None),
					'reward_system': 'LARE' if self.use_lare_reward else 'Traditional',
					'use_lare_training': self.use_lare_training,
					'source_model': source_base_name,
				}

			}
			torch.save(save_data, save_path)

			file_size_kb = os.path.getsize(save_path) / 1024
			is_first_save = not hasattr(self, 'last_checkpoint_path') or not self.checkpoint_saved
			action = "saved" if is_first_save else "overwritten"

			print(f"✅ [CHECKPOINT] Checkpoint {action} successfully!")
			print(f"  - File: {file_name}")
			print(f"  - Size: {file_size_kb:.1f} KB")
			print(f"  - Episodes: {self.episode_account}")
			print(f"  - Training time: {training_duration/3600:.1f} hours")

			self.checkpoint_saved = True

			return save_path
		
		except Exception as e:
			print(f"❌ Error saving checkpoint: {e}")
			return None
		
	def _get_source_model_base_name(self):
		"""
		ファインチューニング元のモデル名からベース名を抽出
		
		Returns:
			str: ベース名
		"""
		if not self.use_finetuning or not hasattr(self, 'finetuning_model_path') or self.finetuning_model_path is None:
			print("❌ Fine-tuning model path not set.")
			return "unknown_source_model"
		
		try:
			file_name = os.path.basename(self.finetuning_model_path)
			base_name = file_name.replace('.pth', '')

			base_name = base_name.replace('_final', '').replace('_checkpoint', '')

			while base_name.startswith('FT_'):
				base_name = base_name[3:]

			return base_name
		
		except Exception as e:
			print(f"❌ Error extracting source model base name: {e}")
			return "unknown_source_model"
		
	def _get_algorithm_name(self):
		"""
		実行時の設定からアルゴリズム名を取得
		
		Returns:
			str: アルゴリズム名（例: "QMIX", "IQL", "VDN"等）
		"""
		try:
			# sys.argvから--configパラメータを探す
			import sys
			args = sys.argv
			
			for i, arg in enumerate(args):
				if (arg == '--config' and i + 1 < len(args)):
					config_value = args[i + 1]
					# config値をアルゴリズム名として使用（大文字に変換）
					return config_value.upper()
				elif arg.startswith('--config='):
					config_value = arg.split('=')[1]
					# config値をアルゴリズム名として使用（大文字に変換）
					return config_value.upper()
			
			# configが見つからない場合のデフォルト
			print(f"⚠️ [ALGORITHM] Could not detect algorithm from command line args")
			return "UNKNOWN"
			
		except Exception as e:
			print(f"❌ [ALGORITHM] Error detecting algorithm name: {e}")
			return "UNKNOWN"

	def get_obs(self):
		return self.obs

	def get_state(self): # unused
		return self.s

	def _get_avail_agent_actions(self, agent_id, n_actions):
		avail_actions = self.ee_env.get_avail_action_fun(self.obs[agent_id], self.current_start[agent_id], self.current_goal[agent_id], self.goal_array[agent_id])
		avail_actions_one_hot = np.zeros(n_actions)
		avail_actions_one_hot[avail_actions] = 1
		return avail_actions_one_hot, avail_actions
	
	def get_avail_agent_actions(self, agent_id, n_actions):
		return self._get_avail_agent_actions(agent_id, n_actions)

	def reset(self):
		if (self.episode_account > 0 and
	  		self.use_lare_reward and
			not self.use_pretrained_model):

			# ステップ数の更新
			if hasattr(self, 'total_step_account') and hasattr(self, 'step_account'):
				self.total_step_account += self.step_account

			# 学習終了判定
			if (hasattr(self, 'max_train_steps') and
				self.total_step_account >= self.max_train_steps and
				not getattr(self, 'training_completed', False)):

				print(f"🏁 [TRAINING COMPLETE] Reached max training steps: {self.total_step_account}/{self.max_train_steps}")
				print(f"  - Total episodes completed: {self.episode_account}")

				self.training_completed = True

				final_save_path = self.save_final_checkpoint()
				if final_save_path:
					print(f"💾 [FINAL CHECKPOINT] Final model saved to: {final_save_path}")
					self.final_model_saved = True			# 重複保存の防止
				else:
					print(f"❌ [FINAL CHECKPOINT FAIL] Failed to save final model"
		   )
			
			if len(self.episode_data['x_e']) > 0:
				# エピソードデータを統一メモリに追加
				if not self.use_separete_memory:
					self.memory_e.push(
						self.episode_data['x_e'],
						self.episode_data['action_e'],
						self.episode_data['mask_e'],
						self.episode_data['reward_e']
					)
				else:
					if hasattr(self, 'current_termination_reason'):
						termination_reason = self.current_termination_reason
					else:
						print("⚠️ [TERMINATION] Termination reason not specified, defaulting to 'unknown'")
						termination_reason = "unknown"
					
					if termination_reason == "collision":
						self.collision_memory.push(
							self.episode_data['x_e'],
							self.episode_data['action_e'],
							self.episode_data['mask_e'],
							self.episode_data['reward_e']
						)
						print(f"💾 [MEMORY] Episode {self.episode_account} saved to COLLISION memory (size: {len(self.collision_memory)})")
					elif termination_reason == "goal":
						self.goal_memory.push(
							self.episode_data['x_e'],
							self.episode_data['action_e'],
							self.episode_data['mask_e'],
							self.episode_data['reward_e']
						)
						print(f"💾 [MEMORY] Episode {self.episode_account} saved to GOAL memory (size: {len(self.goal_memory)})")
					elif termination_reason == "timeup":
						self.timeup_memory.push(
							self.episode_data['x_e'],
							self.episode_data['action_e'],
							self.episode_data['mask_e'],
							self.episode_data['reward_e']
						)
						print(f"💾 [MEMORY] Episode {self.episode_account} saved to TIMEUP memory (size: {len(self.timeup_memory)})")
					else:
						self.collision_memory.push(
							self.episode_data['x_e'],
							self.episode_data['action_e'],
							self.episode_data['mask_e'],
							self.episode_data['reward_e']
						)
						print(f"💾 [MEMORY] Episode {self.episode_account} saved to OTHER memory (size: {len(self.collision_memory)})")
					
					print(f"📊 [MEMORY SUMMARY] Goal: {len(self.goal_memory)}, Collision: {len(self.collision_memory)}, Timeup: {len(self.timeup_memory)}")
					if hasattr(self, 'current_termination_reason'):
						delattr(self, 'current_termination_reason')
			else:
				print(f"⚠️ [MEMORY] No data to save for episode {self.episode_account}")
			
			self.episode_data = {
				"x_e" : [],
				"action_e": [],
				"mask_e" : [],
				"reward_e" : [],
			}

			# 基準エピソードの判定
			if (hasattr(self, 'reward_model_update_freq') and
				self.episode_account % self.reward_model_update_freq == 0):

				if self.use_separete_memory:
					total_memory_size = (len(self.goal_memory) +
										 len(self.collision_memory) +
										 len(self.timeup_memory))
				else:
					total_memory_size = len(self.memory_e)

				if total_memory_size >= self.reward_model_starts:
					# print(f"🔍 [EVALUATION CHECK] Episode {self.episode_account}: Sufficient data (have {total_memory_size}, need {self.reward_model_starts})")
					self.is_evaluation_period = True
					self.current_evaluation_count = 0
				else:
					print(f"⚠️ [EVALUATION SKIP] Episode {self.episode_account}: No enough data for evaluation (have {total_memory_size}, need {self.reward_model_starts})")

			# 更新期間中の処理
			if getattr(self, 'is_evaluation_period', False):
				#print(f"🔄 [EVALUATION UPDATE] Episode {self.episode_account}: Update {self.current_evaluation_count + 1}/{self.evaluation_episodes}")

				self.perform_episode_update()

				self.current_evaluation_count += 1

				# 更新期間終了判定
				if self.current_evaluation_count >= self.evaluation_episodes:
					#print(f"✅ [EVALUATION COMPLETE] Episode {self.episode_account}: Completed {self.evaluation_episodes} updates")
					self.total_update_count += 1

					if self.total_update_count % 70 == 0:
						if self.use_lare_reward and hasattr(self, 'lare_decompose') and self.lare_decompose is not None:
							checkpoint_save_path = self.save_checkpoint()
							if checkpoint_save_path:
								print(f"🎉 [MILESTONE SAVE] Episode {self.episode_account}")
								print(f"💾 [CHECKPOINT] Model saved to: {checkpoint_save_path}")
							else:
								print(f"❌ [CHECKPOINT FAIL] Episode {self.episode_account}: Failed to save checkpoint")

					self.is_evaluation_period = False
					self.current_evaluation_count = 0
			else:
				if self.episode_account > 0:
					#print(f"📝 [NOMAL EPISODE] Episode {self.episode_account}: Data collected, no update")
					pass

		# if goal and start are not assigned, randomly generate every episode    
		self.start_ori_array = copy.deepcopy(self.ee_env.input_start_ori_array)
		self.goal_array = copy.deepcopy(self.ee_env.input_goal_array)
		self._log(f"self.start_ori_array {self.start_ori_array}")
		if self.start_ori_array == []:
			self.ee_env.random_start()
			self.start_ori_array = self.ee_env.start_ori_array
		if self.goal_array == []:
			self.ee_env.random_goal()
			self.goal_array = self.ee_env.goal_array
		self._log(f"self.start_ori_array after {self.start_ori_array}")

		#initialize obs
		self.obs = tuple(np.array([self.pos[self.start_ori_array[i]][0], self.pos[self.start_ori_array[i]][1], self.start_ori_array[i], self.goal_array[i]]) for i in range(self.agent_num))
		self.obs_current_chache = copy.deepcopy(self.obs)# used for calculating reward

		# 🔧 obs_onehotを1D配列として初期化
		self.obs_onehot = np.zeros((self.agent_num, self.n_nodes*2))
		for i in range(self.agent_num):
			self.obs_onehot[i][int(self.start_ori_array[i])] = 1 #current position
			self.obs_onehot[i][int(self.goal_array[i])+self.n_nodes] = 1 #current goal

		# 🔧 位置情報のみをキャッシュ
		self.obs_onehot_position_cache = np.zeros((self.agent_num, self.n_nodes))
		for i in range(self.agent_num):
			self.obs_onehot_position_cache[i][int(self.start_ori_array[i])] = 1

		self.current_start = self.start_ori_array # [0,1]
		self.current_goal  = [None for _ in range(self.agent_num)]
		self.terminated    = [False for _ in range(self.agent_num)]

		self.distance_from_start = np.zeros(self.agent_num) # info
		self.wait_count = np.zeros(self.agent_num) # info

		self.agent_arrival_steps = np.full(self.agent_num, -1, dtype=int) # info
		self.episode_cost = 0 # info

		self.reach_account = 0
		self.step_account = 0
		self.episode_account += 1

		obs = self.obs_manager.calc_obs()

		return obs
		
	def step(self, joint_action):
		#transite env based on joint_action
		self.step_account += 1
		self.obs_current_chache = copy.deepcopy(self.obs)
		
		# 🔧 位置情報のみをキャッシュ（行動実行前の位置）
		self.obs_onehot_position_cache = np.zeros((self.agent_num, self.n_nodes))
		for i in range(self.agent_num):
			# 🔧 obs_onehotの形状を正規化してキャッシュ
			if len(self.obs_onehot[i].shape) == 2:
				agent_onehot_flat = self.obs_onehot[i].flatten()
				self.obs_onehot_position_cache[i] = agent_onehot_flat[:self.n_nodes].copy()
			else:
				self.obs_onehot_position_cache[i] = self.obs_onehot[i][:self.n_nodes].copy()
		
		self.obs_prepare = []
		self.obs_onehot_prepare = copy.deepcopy(self.obs_onehot)
		self.current_start_prepare = copy.deepcopy(self.current_start)
		self.current_goal_prepare = copy.deepcopy(self.current_goal)
		# 1) first judge action_i whether available, to output !!!obs_prepare & obs_onehot_prepare!!!
		for i in range(self.agent_num):
			action_i = joint_action[i]  
			# 1) first judge action_i whether available, to output obs_prepare: 
			# if unavailable ⇢ obs_prepare.append( self.obs_old[i])
			#print("Avaible actions",self.get_avail_agent_actions(i, self.n_actions)[1])
			if action_i not in self._get_avail_agent_actions(i, self.n_actions)[1]:
				#print("This is not Avaible",i,action_i,self.get_avail_agent_actions(i, self.n_actions)[1])
				self.obs_prepare.append(self.obs_current_chache[i])
				self.wait_count[i] += 1

			# if action_i is current start node -> stop
			elif self.pos[int(action_i)][0]==self.obs[i][0] and self.pos[int(action_i)][1]==self.obs[i][1]:
				self.obs_prepare.append(self.obs_current_chache[i])
				
				# 💥 修正: ゴールに到達している場合は待機カウントを増やさない
				current_pos = [self.obs[i][0], self.obs[i][1]]
				goal_pos = list(self.pos[self.goal_array[i]])
				
				if current_pos != goal_pos:
					self.wait_count[i] += 1
			# if available ⇢ obs_prepare update by obs_i_
			else:
				# 💥 修正: エージェントが移動したので待機カウントをリセット
				self.wait_count[i] = 0

				#self.joint_action_old[i] = joint_action[i]
				self.current_goal_prepare[i] = joint_action[i] #update 行き先ノード when avable action is taken
				obs_i = self.obs[i]
		
				#calculate current distance
				current_goal = list(self.pos[int(action_i)])
				current_x1,current_y1 = obs_i[0], obs_i[1]
				x = current_goal[0] - current_x1
				y = current_goal[1] - current_y1
				dist_to_cgoal = np.sqrt(np.square(x) + np.square(y))# the distance to current goal

				if dist_to_cgoal>self.speed:# move on edge
					current_x1 = round(current_x1+(self.speed*x/dist_to_cgoal), 2)
					current_y1 = round(current_y1+(self.speed*y/dist_to_cgoal), 2)
					obs_i_ = [round(current_x1,2), round(current_y1,2), obs_i[2], obs_i[3]]
					
					# for one-hot state
					x = list(self.pos[self.current_start[i]])[0] - current_x1
					y = list(self.pos[self.current_start[i]])[1] - current_y1
					dist_to_cstart = np.sqrt(np.square(x) + np.square(y))# the distance to current goal
					dist_to_cstart_rate = round(dist_to_cstart/(dist_to_cstart+dist_to_cgoal), 2)
					
					#print("self.obs_onehot_prepare before",self.obs_onehot_prepare )
					self.obs_onehot_prepare[i] = np.zeros((1, len(list(self.G.nodes()))*2))
					self.obs_onehot_prepare[i][int(action_i)] = dist_to_cstart_rate
					self.obs_onehot_prepare[i][int(self.current_start[i])] = 1-dist_to_cstart_rate
					self.obs_onehot_prepare[i][int(self.goal_array[i])+len(list(self.G.nodes()))] = 1 #current goal
					#print("self.obs_onehot_prepare after",self.obs_onehot_prepare )
					self.distance_from_start[i] += self.speed
				# arrive at node
				else:
					obs_i_ = [round(self.pos[int(action_i)][0],2), round(self.pos[int(action_i)][1],2), obs_i[2], obs_i[3]]
					
					# for one-hot state
					self.obs_onehot_prepare[i] = np.zeros((1, len(list(self.G.nodes()))*2))
					self.obs_onehot_prepare[i][int(action_i)] = 1
					self.obs_onehot_prepare[i][int(self.goal_array[i])+len(list(self.G.nodes()))] = 1 #current goal
					
					# update current_start only when arrive at node
					self.current_start_prepare[i] = int(action_i) #update 出発ノード when　行き先ノード　has been arrived
					self.current_goal_prepare[i] = None #update 行き先ノード when it has been arrived

					self.distance_from_start[i] += dist_to_cgoal

				self.obs_prepare.append(obs_i_)
		
		# 2) !!!obs_prepare & obs_onehot_prepare!!! を持って、
		# second judge whether to !!! obs & obs_onehot !!! according to collision happen
		collision_flag = self.ee_env.collision_detect(self.obs_prepare)
		# 💥 修正: 衝突ペアのリストを取得
		colliding_pairs = self.ee_env.get_collision_agents(self.obs_prepare)
		self.current_colliding_pairs = colliding_pairs

		# 💥 修正: 衝突に関与した全エージェントのIDリストを作成（従来の後方互換性のため）
		collision_list = np.zeros(self.agent_num, dtype=int)
		for pair in colliding_pairs:
			for agent in pair:
				collision_list[agent] = 1

		self.current_collision_list = collision_list

		info = {
			"goal": False,
			"collision": False,
			"timeup": False, # for epymarl
			"distance_from_start": None,
			"step": self.step_account,
			"wait": list(self.wait_count),
			"cost": 0,
			"goal_cost": None,
		}
		# happen
		if collision_flag==1:#collision
			#collision_reward=-1
			collision_reward = self.r_coll*self.speed
			if self.collision == "bounceback":
				self.terminated = [False for _ in range(self.agent_num)]
			else: # default -> self.collision == "terminated"
				self.terminated = [True for _ in range(self.agent_num)]

				self.episode_cost = self.agent_num * self.time_limit
				info["cost"] = self.episode_cost
			info["collision"] = True
			obs = self.obs_manager.calc_obs()

			# NN学習用の報酬のメモリ
			memory_rewards = []

			# 各エージェントのメモリ用報酬を個別に計算
			for i in range(self.agent_num):
				is_collision_agent = (collision_list[i] == 1)

				if is_collision_agent:
					memory_reward = collision_reward
				else:
					# 衝突していないエージェント
					# obsが更新されていないので一時的に更新
					self.obs = tuple([np.array(j) for j in self.obs_prepare])
					memory_reward = self.reward(i)
					self.obs = self.obs_current_chache # 元に戻す

					#ゴール到達済みか
					pre_pos_agenti = [self.obs_current_chache[i][0],self.obs_current_chache[i][1]]
					pos_agenti = [self.obs_prepare[i][0],self.obs_prepare[i][1]]
					goal_pos = self.pos[self.goal_array[i]]
				
				memory_rewards.append(memory_reward)
			
			# LARE報酬システムが有効な場合は衝突報酬を無視
			if self.use_lare_reward and hasattr(self, 'lare_decompose') and self.lare_decompose is not None:

				obs_backup = self.obs
				onehot_backup = copy.deepcopy(self.obs_onehot)

				self.obs = tuple([np.array(j) for j in self.obs_prepare])
				self.obs_onehot = copy.deepcopy(self.obs_onehot_prepare)

				self.current_state = {
					agent_id: self._get_lare_compatible_obs(agent_id) for agent_id in range(self.agent_num)
				}

				self.obs = obs_backup
				self.obs_onehot = onehot_backup

				# LARE報酬システムを使用している場合は各エージェントのLARE報酬を計算
				ri_array = []
				
				# 修正箇所: _call_lare_reward_system の呼び出し
				for i in range(self.agent_num):
					lare_reward = self._call_lare_reward_system(i, next_obs=None, use_cache=True)
					pos_str = self._get_position_transition_str(i)

					if self.use_lare_training and lare_reward is not None:
						ri = lare_reward
						self._log(f"   {i}: LARE={ri:.6f} ({memory_rewards[i]}) {pos_str}" )
					else:
						# LARE報酬システムが失敗した場合のみ衝突報酬を使用
						ri = collision_reward
						if lare_reward is not None:
							self._log(f"   {i}: TRAD={ri}, LARE({lare_reward:.6f}) {pos_str}")
						else:
							print(f" ⚠️  {i}: TRAD={ri}, LARE failed {pos_str}")
        
					ri_array.append(ri)
			else:
				# 従来の報酬システムを使用している場合は衝突報酬を適用
				ri_array = [collision_reward for _ in range(self.agent_num)]
				self._log(f"🔴 [COLLISION] Step {self.step_account}: All agents receive collision reward: {collision_reward}")
			
		# not happen
		else: #non collision
			self.obs = tuple([np.array(i) for i in self.obs_prepare])
			self.obs_onehot = copy.deepcopy(self.obs_onehot_prepare)
			self.current_start = copy.deepcopy(self.current_start_prepare)   
			self.current_goal = copy.deepcopy(self.current_goal_prepare)

			team_reward = 0
			ri_array = []
			memory_rewards = []

			if self.use_lare_reward:
				self.current_state = {
					agent_id: self._get_lare_compatible_obs(agent_id) for agent_id in range(self.agent_num)
				}
			
			
			self._log(f" Step {self.step_account}")
			
			for i in range(self.agent_num):
				traditional_reward = self.reward(i)
				memory_rewards.append(traditional_reward)
				pos_str = self._get_position_transition_str(i)

				# LARE報酬システムを使用するかどうかで分岐
				if self.use_lare_reward and hasattr(self, 'lare_decompose') and self.lare_decompose is not None:
					# LARE報酬システムを使用
					lare_reward = self._call_lare_reward_system(i, next_obs=None, use_cache=True)
					if self.use_lare_training and lare_reward is not None:
						ri = lare_reward
						self._log(f"   {i}: LARE={ri:.6f}({traditional_reward}) {pos_str}" )
					else:
						# LARE報酬システムが失敗した場合は従来の報酬を使用
						ri = traditional_reward
						if lare_reward is not None:
							self._log(f"   {i}: TRAD={ri}, LARE({lare_reward:.6f}) {pos_str}")
						else:
							print(f" ⚠️  {i}: TRAD={ri}, LARE failed {pos_str}")
				else:
					# 従来の報酬システムを使用
					ri = traditional_reward
					
					# 従来の報酬の詳細を表示
					pre_pos_agenti = [self.obs_current_chache[i][0],self.obs_current_chache[i][1]]
					pos_agenti = [self.obs[i][0],self.obs[i][1]]
					goal_pos = self.pos[self.goal_array[i]]
					pos_str = self._get_position_transition_str(i)
					
					if str(pos_agenti)==str(goal_pos): # at goal
						if pre_pos_agenti!=pos_agenti : #first time to reach goal 
							self._log(f"   {i}:reward = {ri} {pos_str}")
						else: # stop at goal
							self._log(f"   {i}:reward = {ri} {pos_str}")
					else: #at a general node 
						if pre_pos_agenti==pos_agenti: # stop at a general node 
							self._log(f"   {i}:reward = {ri} {pos_str}")
						else: # just move 
							self._log(f"   {i}:reward = {ri} {pos_str}")
				
				team_reward += ri
				ri_array.append(ri)
			
			if self.terminated == [True for _ in range(self.agent_num)]: # all reach goal
				self._log("!!!all reach goal!!!")
				self.reach_account = 0
				# info
				info["goal"] = True

				self.episode_cost = self._calculate_episode_cost(info)
				info["cost"] = self.episode_cost
				info["goal_cost"] = self.episode_cost
				self._log(f"Episode cost: {self.episode_cost}")
			
			else:
				pass
			

			obs = self.obs_manager.calc_obs()

		# Check whether time is over
		if self.step_account >= self.time_limit:
			self._log(f"!!!TIME UP!!! (Step {self.step_account}/{self.time_limit})")
			info["timeup"]= True
			self.terminated = [True for _ in range(self.agent_num)]

			self.episode_cost = self.agent_num * self.time_limit
			info["cost"] = self.episode_cost
		
		if self.use_separete_memory and all(self.terminated):
			self.current_termination_reason = self._get_termination_reason(info)

		masks = np.array([
				[
					[1 if t < self.time_limit and not done else 0 for done in self.terminated]
					for t in range(self.time_limit)
				]
			])
		
		# 🔧 SIMPLIFIED: エピソード継続中は常にデータを収集
		if (self.use_lare_reward and 
            not self.use_pretrained_model and
            hasattr(self, 'current_state')):
            
			self.episode_data["x_e"].append(np.array(list(self.current_state.values())))
			self.episode_data["action_e"].append(np.array(joint_action).reshape(1, -1))
			self.episode_data["mask_e"].append(masks)
			self.episode_data["reward_e"].append(np.array(memory_rewards))

		info["distance_from_start"] = list(self.distance_from_start)

		return obs, ri_array, self.terminated, info

	def _get_position_transition_str(self, agent_id):
		"""
		エージェントの位置遷移を文字列で返す

		Args:
			agent_id (int): エージェントのID
		Returns:
			str: 位置遷移文字列
		"""
		try:
			prev_nodes = None
			prev_on_node = False
			if hasattr(self, 'obs_onehot_position_cache') and self.obs_onehot_position_cache is not None:
				prev_part = self.obs_onehot_position_cache[agent_id]
				prev_indices = np.where(prev_part != 0)[0]
				if len(prev_indices) == 1:
					prev_nodes = (prev_indices[0],)
					prev_on_node = True
				elif len(prev_indices) >= 2:
					prev_nodes = tuple(prev_indices[:2])
					prev_on_node = False
				
			curr_nodes = None
			curr_on_node = False
			if hasattr(self, 'obs_onehot') and self.obs_onehot is not None:
				if len(self.obs_onehot[agent_id].shape) == 2:
					agent_onehot = self.obs_onehot[agent_id].flatten()
				else:
					agent_onehot = self.obs_onehot[agent_id]

				curr_part = agent_onehot[:self.n_nodes]
				curr_indices = np.where(curr_part != 0)[0]
				if len(curr_indices) == 1:
					curr_nodes = (curr_indices[0],)
					curr_on_node = True
				elif len(curr_indices) >= 2:
					curr_nodes = tuple(curr_indices[:2])
					curr_on_node = False

			goal_node = self.goal_array[agent_id] if hasattr(self, 'goal_array') else None

			if prev_nodes is None or curr_nodes is None:
				return f"?({goal_node})"
			
			def get_edge_direction(edge_nodes, reference_nodes):
				"""
				エッジ上の２ノードから移動方向を特定する
				
				Args:
					edge_nodes: エッジ上の2ノード (タプル)
					reference_nodes: 参照用のノード (タプル)
				Returns:
					(start, end) のタプル
				"""
				if len(edge_nodes) != 2:
					return edge_nodes
				
				node_a, node_b = edge_nodes[0], edge_nodes[1]
				
				# reference_nodesをセットに変換（タプルの場合）
				if isinstance(reference_nodes, tuple):
					ref_set = set(reference_nodes)
				else:
					ref_set = {reference_nodes}

				if node_a in ref_set and node_b not in ref_set:
					return (node_a, node_b)
				elif node_b in ref_set and node_a not in ref_set:
					return (node_b, node_a)

				return tuple(sorted(edge_nodes))
			
			def format_node(node, goal):
				"""単一ノードをフォーマット"""
				if node == goal:
					return f"<{node}>"
				else:
					return f"[{node}]"
			
			# ケース1: 両方ノード上
			if prev_on_node and curr_on_node:
				prev_node = prev_nodes[0]
				curr_node = curr_nodes[0]
				prev_str = format_node(prev_node, goal_node)
				curr_str = format_node(curr_node, goal_node)

				if prev_node == curr_node:
					return f"{curr_str}({goal_node})"
				else:
					return f"{prev_str}→{curr_str}({goal_node})"
			
			# ケース2: ノード→エッジ（移動開始）
			elif prev_on_node and not curr_on_node:
				prev_node = prev_nodes[0]
				directed = get_edge_direction(curr_nodes, prev_nodes)
				dest_node = directed[1]
				prev_str = format_node(prev_node, goal_node)
				return f"{prev_str}→{dest_node}({goal_node})"
			
			# ケース3: エッジ→ノード（到着）
			elif not prev_on_node and curr_on_node:
				curr_node = curr_nodes[0]
				node_a, node_b = prev_nodes[0], prev_nodes[1]
				if node_a == curr_node:
					start_node = node_b
				elif node_b == curr_node:
					start_node = node_a
				else:
					start_node = min(node_a, node_b)  # デフォルト
				curr_str = format_node(curr_node, goal_node)
				return f"{start_node}→{curr_str}({goal_node})"
			
			# ケース4: エッジ→エッジ（移動中）
			else:
				directed = get_edge_direction(curr_nodes, prev_nodes)
				return f"{directed[0]}→{directed[1]}({goal_node})"
			
		except Exception as e:
			return f"Error({e})"

	def reward(self, i):
		pre_pos_agenti = [self.obs_current_chache[i][0],self.obs_current_chache[i][1]]
		pos_agenti = [self.obs[i][0],self.obs[i][1]]

		if str(pos_agenti)==str(self.pos[self.goal_array[i]]): # at goal
			if pre_pos_agenti!=pos_agenti : #first time to reach goal 
				r_i = self.r_goal
				self.reach_account += 1
				self.terminated[i] = True
				self._record_agent_arrival(i)
			else: # stop at goal
				r_i = 0   
				# self.distance_from_start[i] -= self.speed
		
		else: #at a general node 
			if pre_pos_agenti==pos_agenti: # stop at a general node 
				r_i = self.r_wait*self.speed
			else: # just move 
				r_i = self.r_move*self.speed
			
		return r_i


	def render(self, mode='human'):
		self.ee_env.plot_map_dynamic(
			self.visu_delay,self.obs_current_chache,
			self.obs,self.goal_array,
			self.agent_num,
			self.current_goal,
			self.reach_account,
			self.step_account,
			self.episode_account
		) # a must be a angle !!!list!!!

	def close(self):
		print('Environment CLOSE')
		return None

	def _log(self, message, force_print=False):
		"""
		メッセージをログファイルとコンソールに出力するユーティリティ関数

		Args:
			message (str): ログメッセージ
			force_print (bool): コンソールにも出力するかどうか
		"""
		if self.show_debug_logs or force_print:
			print(message)	

	def _calculate_episode_cost(self, info):
		"""エピソードのコストを計算して返す"""
		
		if info.get("collision", False) or info.get("timeup", False):
			cost = self.agent_num * self.time_limit
			print(f"✅ [Episode{self.episode_account}] Total cost due to {'collision' if info.get('collision', False) else 'timeup'}: {cost}")
			return int(cost)
		
		if info.get("goal", False):
			cost = 0
			for i in range(self.agent_num):
				if self.agent_arrival_steps[i] > 0:
					cost += self.agent_arrival_steps[i]
				else:
					print(f"⚠️ [COST CALCULATION] Agent {i} has invalid arrival step: {self.agent_arrival_steps[i]}")
					cost += self.time_limit
			return int(cost)
		
		print("⚠️ [COST CALCULATION] Episode did not end with goal, collision, or timeup. Assigning maximum cost.")
		return self.agent_num * self.time_limit
	
	def get_episode_cost(self):
		return self.episode_cost
	
	def _record_agent_arrival(self, agent_id):
		"""エージェントの到着ステップを記録する"""
		if self.agent_arrival_steps[agent_id] < 0:
			self.agent_arrival_steps[agent_id] = self.step_account
			self._log(f"✅ [ARRIVAL] Agent {agent_id} arrived at step {self.step_account}")

	def _get_termination_reason(self, info):
		"""エピソード終了の理由を判定して返す"""
		if info.get("collision", False):
			return "collision"
		elif info.get("goal", False):
			return "goal"
		elif info.get("timeup", False):
			return "timeup"
		else:
			return "unknown"
    
	def get_pos_list(self):
		pos_list = []
		all_onehot_obs = np.array(self.obs_onehot)
		onehot_obs = all_onehot_obs[:, :self.n_nodes]

		# get all agent state and position
		for i, obs_i in enumerate(onehot_obs):
			edge_or_node = tuple([i for i, o in enumerate(obs_i) if o!=0])
			if len(edge_or_node)==1:
				node = edge_or_node[0]
				pos = {"type": "n", "pos": node}
				obs_i = np.array(obs_i)*self.agent_num
			else:
				edge = edge_or_node
				pos = {"type": "e", "pos": edge, "current_goal": self.current_goal[i], "current_start": self.current_start[i], "obs": obs_i}
			pos_list.append(pos)

		return pos_list