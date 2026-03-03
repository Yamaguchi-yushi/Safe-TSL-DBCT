"""
LareCentralTrainer
==================
ParallelRunner と組み合わせて使う LaRe モデルの中央管理クラス。

問題:
  ParallelRunner は複数のワーカープロセスを立ち上げ、それぞれが
  独立した DrpEnv (= 独立した LaRe モデル) を持つ。
  ワーカーが独立して学習すると経験の 1/N しか使えず、学習効率が低下する。

解決策:
  - ワーカーは LaRe モデルを推論専用として使う (lare_training_disabled=True)
  - 各ワーカーが集めたエピソードデータをメインプロセスに送る
  - メインプロセスの LareCentralTrainer が集約データで一括学習する
  - 学習後にメインプロセスの weights をワーカーに配布する
"""

import os
import time
import copy

import numpy as np
import torch
from torch import nn

from epymarl.src.utils.util import make_train_step
from epymarl.src.utils.replay_memory import ReplayMemory_episode


class LareCentralTrainer:
	"""メインプロセスで LaRe モデルを一元管理するクラス。"""

	def __init__(self, model_info: dict):
		"""
		Args:
			model_info: DrpEnv.get_lare_model_info() が返す辞書。
		"""
		self.use_lare_reward = model_info.get('use_lare_reward', False)
		if not self.use_lare_reward:
			return

		# ---- LaRe モデル ----
		self.lare_decompose = model_info['lare_decompose']
		self.n_agents       = model_info['n_agents']
		self.device         = 'cuda' if torch.cuda.is_available() else 'cpu'
		self.lare_decompose.to(self.device)

		# ---- 観測キャッシュ (静的部分の再構成に使う) ----
		self.static_obs_cache = model_info.get('static_obs_cache', None)

		# ---- リプレイメモリ ----
		self.use_separete_memory = model_info.get('use_separete_memory', False)
		buffer_size  = model_info.get('buffer_size', 1024)
		time_limit   = model_info.get('time_limit', 300)
		reward_norm  = model_info.get('reward_norm', False)

		if self.use_separete_memory:
			self.goal_memory      = ReplayMemory_episode(buffer_size, time_limit, reward_norm)
			self.collision_memory = ReplayMemory_episode(buffer_size, time_limit, reward_norm)
			self.timeup_memory    = ReplayMemory_episode(buffer_size, time_limit, reward_norm)
		else:
			# ワーカー分まとめて受け取るので容量を大きめに取る
			self.memory_e = ReplayMemory_episode(buffer_size * 2, time_limit, reward_norm)

		# ---- 学習ステップ関数 ----
		self.reward_model = self.lare_decompose
		loss_fn = nn.MSELoss(reduction='mean')
		lr      = 5e-4
		opt     = torch.optim.Adam(
			params=self.reward_model.parameters(),
			lr=lr, weight_decay=1e-5,
		)
		self.train_step = make_train_step(
			self.reward_model, loss_fn, opt,
			self.n_agents, self.device,
			env=None, reg=False, alpha=0.0,
		)

		# ---- 学習スケジュール ----
		self.reward_model_update_freq = model_info.get('reward_model_update_freq', 128)
		self.evaluation_episodes      = model_info.get('evaluation_episodes', 50)
		self.rewardbatch_size         = model_info.get('rewardbatch_size', 256)
		self.reward_model_starts      = model_info.get('reward_model_starts', 256)
		self.max_train_steps          = model_info.get('max_train_steps', None)

		# ---- 状態変数 ----
		self.episode_account          = 0
		self.total_step_account       = 0
		self.is_evaluation_period     = False
		self.current_evaluation_count = 0
		self.total_update_count       = 0
		self.training_completed       = False
		self.training_start_time      = time.time()
		self.checkpoint_saved         = False
		self.last_checkpoint_path     = None

		# ---- ファイル名用メタ情報 ----
		self.map_name         = model_info.get('map_name', 'unknown_map')
		self.use_finetuning   = model_info.get('use_finetuning', False)
		self.source_base_name = model_info.get('source_base_name', None)
		self.is_safe          = model_info.get('is_safe', False)

		print(f"✅ [LARE CENTRAL] Trainer initialized on device: {self.device}")
		print(f"   - map: {self.map_name}, n_agents: {self.n_agents}")
		print(f"   - memory: {'separate' if self.use_separete_memory else 'unified'}")

	# ------------------------------------------------------------------
	# データ追加
	# ------------------------------------------------------------------

	def add_episode_data(self, episode_data: dict):
		"""ワーカーから受け取ったエピソードデータを中央メモリに追加する。"""
		if not self.use_lare_reward:
			return

		x_e               = episode_data.get('x_e', [])
		action_e          = episode_data.get('action_e', [])
		reward_e          = episode_data.get('reward_e', [])
		termination_reason = episode_data.get('termination_reason', 'unknown')
		step_count        = episode_data.get('step_count', len(x_e))

		if len(x_e) == 0:
			return

		self.episode_account    += 1
		self.total_step_account += step_count

		if self.use_separete_memory:
			if termination_reason == 'goal':
				self.goal_memory.push(x_e, action_e, reward_e)
			elif termination_reason == 'collision':
				self.collision_memory.push(x_e, action_e, reward_e)
			else:
				self.timeup_memory.push(x_e, action_e, reward_e)
		else:
			self.memory_e.push(x_e, action_e, reward_e)

	# ------------------------------------------------------------------
	# 学習
	# ------------------------------------------------------------------

	def _total_memory_size(self) -> int:
		if self.use_separete_memory:
			return (len(self.goal_memory) +
					len(self.collision_memory) +
					len(self.timeup_memory))
		return len(self.memory_e)

	def maybe_update(self):
		"""条件を満たす場合に LaRe モデルを更新する。"""
		if not self.use_lare_reward or self.episode_account == 0:
			return

		# 更新期間の開始判定
		if (self.episode_account % self.reward_model_update_freq == 0 and
				self._total_memory_size() >= self.reward_model_starts):
			self.is_evaluation_period     = True
			self.current_evaluation_count = 0
			print(f"🔄 [LARE CENTRAL] Update period started "
				  f"(episode={self.episode_account}, memory={self._total_memory_size()})")

		if self.is_evaluation_period:
			self._perform_update()
			self.current_evaluation_count += 1
			if self.current_evaluation_count >= self.evaluation_episodes:
				print(f"✅ [LARE CENTRAL] Update period done "
					  f"({self.total_update_count + 1} total rounds)")
				self.is_evaluation_period = False
				self.total_update_count  += 1

	def _perform_update(self):
		"""実際のモデル更新を 1 ステップ実行する。"""
		try:
			if self.use_separete_memory:
				goal_size = len(self.goal_memory)
				col_size  = len(self.collision_memory)
				time_size = len(self.timeup_memory)
				total     = goal_size + col_size + time_size
				if total < self.reward_model_starts:
					return

				n = self.rewardbatch_size // 3
				n_goal = min(n, goal_size)
				n_col  = min(n, col_size)
				n_time = min(self.rewardbatch_size - n_goal - n_col, time_size)

				parts = []
				for mem, n_sample in [(self.goal_memory, n_goal),
									  (self.collision_memory, n_col),
									  (self.timeup_memory, n_time)]:
					if n_sample > 0:
						parts.append(mem.sample_trajectory(n_sample))

				if not parts:
					return
				states         = np.concatenate([p[0] for p in parts], axis=0)
				actions        = np.concatenate([p[1] for p in parts], axis=0)
				episode_return = np.concatenate([p[2] for p in parts], axis=0)
				episode_length = np.concatenate([p[3] for p in parts], axis=0)
			else:
				if len(self.memory_e) < self.reward_model_starts:
					return
				states, actions, episode_return, episode_length = \
					self.memory_e.sample_trajectory(self.rewardbatch_size)

			# 静的 obs を結合してフル obs を再構成
			if self.static_obs_cache is not None:
				static_broadcast = np.broadcast_to(
					self.static_obs_cache,
					states.shape[:-1] + self.static_obs_cache.shape,
				)
				states = np.concatenate([states, static_broadcast], axis=-1)

			# Tensor 変換
			to_tensor = lambda x: torch.tensor(x, dtype=torch.float32).to(self.device)
			states         = to_tensor(states)
			actions        = to_tensor(actions)
			episode_return = to_tensor(episode_return)
			episode_length = to_tensor(episode_length)

			self.train_step(states, actions, episode_return, episode_length)

		except Exception as e:
			import traceback
			print(f"❌ [LARE CENTRAL] Update error: {e}")
			print(traceback.format_exc())

	# ------------------------------------------------------------------
	# weights の受け渡し
	# ------------------------------------------------------------------

	def get_state_dict(self) -> dict:
		"""CPU に移した state_dict を返す (Pipe で送信可能)。"""
		return {k: v.cpu() for k, v in self.lare_decompose.state_dict().items()}

	# ------------------------------------------------------------------
	# チェックポイント保存
	# ------------------------------------------------------------------

	def check_and_save(self, t_env: int, algorithm_name: str):
		"""総ステップ数を確認し、必要に応じてモデルを保存する。"""
		if not self.use_lare_reward or self.training_completed:
			return

		if (self.max_train_steps is not None and
				t_env >= self.max_train_steps):
			self.training_completed = True
			self._save_final(algorithm_name)

		elif (self.total_update_count > 0 and
			  not self.checkpoint_saved):
			self._save_checkpoint(algorithm_name)

	def _save_dir(self) -> str:
		base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
		d = os.path.join(base, "epymarl", "src", "saved_models")
		os.makedirs(d, exist_ok=True)
		return d

	def _make_filename(self, algorithm_name: str, steps_str: str, suffix: str) -> str:
		safe_prefix = "SAFE_" if self.is_safe else ""
		if self.use_finetuning and self.source_base_name:
			return (f"{safe_prefix}FT_{self.map_name}_{self.n_agents}agents_"
					f"{steps_str}_{self.source_base_name}_{suffix}.pth")
		return (f"{safe_prefix}{algorithm_name}_LARE_{self.map_name}_"
				f"{self.n_agents}agents_{steps_str}_{suffix}.pth")

	def _save_data(self, algorithm_name: str, suffix: str) -> str:
		steps_str = f"{self.total_step_account / 1_000_000:.1f}M"
		filename  = self._make_filename(algorithm_name, steps_str, suffix)
		path      = os.path.join(self._save_dir(), filename)
		torch.save({
			'model_state_dict': self.lare_decompose.state_dict(),
			'total_steps':      self.total_step_account,
			'episodes':         self.episode_account,
			'save_timestamp':   time.time(),
		}, path)
		return path

	def _save_final(self, algorithm_name: str):
		path = self._save_data(algorithm_name, "final")
		print(f"💾 [LARE CENTRAL] Final model saved: {os.path.basename(path)}")

	def _save_checkpoint(self, algorithm_name: str):
		path = self._save_data(algorithm_name, "checkpoint")
		self.checkpoint_saved    = True
		self.last_checkpoint_path = path
		print(f"💾 [LARE CENTRAL] Checkpoint saved: {os.path.basename(path)}")
