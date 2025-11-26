import requests
import os
import json
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from prompt_template import get_prompt
import numpy as np
import argparse
import torch as th
from factor_chat_with_gpt import call_drp_gpt
import torch
import random
from factor_reward_model import Factor_Reward_Model
from torch import nn
import copy

class FactorRewardDecomposer(nn.Module):  # 報酬分解器クラスを定義
    def __init__(self, args):  # 初期化メソッド
        super(FactorRewardDecomposer, self).__init__()  # 親クラスの初期化を呼び出し
        self.args = args  # 引数をインスタンス変数に保存
        self.n_agents = args.n_agents  # エージェント数を設定
        self.device = 'cpu'  # デバイスを設定
        
        # DRP用にディレクトリパスを調整
        if hasattr(args, 'llm_response_dir'):  # llm_response_dir属性が存在する場合
            response_dir = args.llm_response_dir  # 属性値を使用
        else:
            response_dir = 'responses'  # デフォルト値を設定
        
        if hasattr(args, 'map_name'):  # map_name属性が存在する場合
            self.map_name = args.map_name  # 属性値を使用（DRP用）
        else:
            self.map_name = 'drp_default'  # デフォルト値を設定
        
        # 保存ディレクトリを設定
        self.rd_save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), response_dir, self.map_name, str(args.seed))
        self.id = 0  # IDを初期化
        
        # DRP用にenv_nameとscenarioを調整
        env_name = getattr(args, 'env_name', 'drp')  # env_name属性を取得（デフォルトは'drp'）
        self.prompt = get_prompt(env_name,
                                 self.map_name,
                                 factor_decomp=True,
                                 n_agents = self.n_agents,
                                 collision_dis = getattr(self.args, "collision_dis", None),
                                 speed = getattr(self.args, "speed", None)
                                )  # プロンプトを生成
        
        if not args.only_s:  # only_sがFalseの場合
            self.load_rd_functions()  # RD関数をロード
            self.get_factor_num()  # 因子数を取得
        
        agent_id_num = 0 if not getattr(args, 'obs_agent_id', False) else self.n_agents  # エージェントID数を設定
        
        if args.only_s:  # only_sがTrueの場合
            self.reward_model = Factor_Reward_Model(args.obs_dim, n_layers=getattr(args, 'factor_reward_model_layers', 3), device=self.device)  # 報酬モデルを初期化
        else:
            self.reward_model = Factor_Reward_Model(self.factor_num, n_layers=getattr(args, 'factor_reward_model_layers', 3), device=self.device)  # 報酬モデルを初期化

    def load_rd_functions(self):
        if not os.path.exists(self.rd_save_dir):
            os.makedirs(self.rd_save_dir)

        map_name = self.map_name
        print(f"🔍 [DEBUG] map_name: {map_name}")
        
        seed_value = getattr(self.args, 'seed', 42)
        response_file = os.path.join(self.rd_save_dir, f'response_{self.id}_seed_{seed_value}.npy')
        
        # 新しいAPI呼び出しを強制するために既存ファイルを確認・削除
        force_new_api_call = getattr(self.args, 'force_new_api_call', True)
        
        if os.path.exists(response_file) and force_new_api_call:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = os.path.join(self.rd_save_dir, f'response_{self.id}_seed_{seed_value}_backup_{timestamp}.npy')
            os.rename(response_file, backup_file)
            
            factor_num_file = os.path.join(self.rd_save_dir, f'factor_num_{self.id}_seed_{seed_value}.npy')
            dialog_file = os.path.join(self.rd_save_dir, f'dialog_{self.id}_seed_{seed_value}.npy')
            
            if os.path.exists(factor_num_file):
                os.rename(factor_num_file, factor_num_file.replace('.npy', f'_backup_{timestamp}.npy'))
            if os.path.exists(dialog_file):
                os.rename(dialog_file, dialog_file.replace('.npy', f'_backup_{timestamp}.npy'))
        
        # API呼び出しの実行
        if not os.path.exists(response_file):
            env_name = getattr(self.args, 'env_name', 'drp')
            
            try:
                call_drp_gpt(env_name, self.map_name, self.rd_save_dir, True, self.id, 
                            n=getattr(self.args, 'llm_n', 1), 
                            port=getattr(self.args, 'port', 8080), 
                            seed=seed_value, 
                            agent_num=self.n_agents)
            except Exception as e:
                # 🔧 FIXED: map_nameとagent_numを渡す
                from factor_chat_with_gpt import generate_fallback_drp_response
                fallback_response = generate_fallback_drp_response(
                    map_name=self.map_name,
                    agent_num=self.n_agents
                )
                
                # フォールバック関数が見つからない場合のエラーハンドリング
                if fallback_response is None:
                    raise RuntimeError(
                        f"❌ [FATAL] Fallback function not found for {self.map_name} "
                        f"with {self.n_agents} agents. Please create the file: "
                        f"{self.map_name}_{self.n_agents}agents.py"
                    )
                
                np.save(response_file, [json.dumps(fallback_response)])
                
                factor_num_file = os.path.join(self.rd_save_dir, f'factor_num_{self.id}_seed_{seed_value}.npy')
                np.save(factor_num_file, fallback_response.get('factor_number', 10))

        if os.path.exists(response_file):
            rd_responses = np.load(response_file, allow_pickle=True)
            
            rd_functions = []
            for i in range(len(rd_responses)):
                try:
                    func = json.loads(rd_responses[i])['Functions']
                    rd_functions.append(func)
                except (json.JSONDecodeError, KeyError) as e:
                    continue
            
            if len(rd_functions) > 0:
                self.rd_functions = rd_functions
            else:
                # 🔧 FIXED: map_nameとagent_numを渡す
                from factor_chat_with_gpt import generate_fallback_drp_response
                fallback_response = generate_fallback_drp_response(
                    map_name=self.map_name,
                    agent_num=self.n_agents
                )
                
                if fallback_response is None:
                    raise RuntimeError(
                        f"❌ [FATAL] Fallback function not found for {self.map_name} "
                        f"with {self.n_agents} agents"
                    )
                
                self.rd_functions = [fallback_response['Functions']]
        else:
            # 🔧 FIXED: map_nameとagent_numを渡す
            from factor_chat_with_gpt import generate_fallback_drp_response
            fallback_response = generate_fallback_drp_response(
                map_name=self.map_name,
                agent_num=self.n_agents
            )
            
            if fallback_response is None:
                raise RuntimeError(
                    f"❌ [FATAL] Fallback function not found for {self.map_name} "
                    f"with {self.n_agents} agents"
                )
            
            self.rd_functions = [fallback_response['Functions']]

    def get_factor_num(self):
        """因子数を取得するメソッド（10固定版）"""
        seed_value = getattr(self.args, 'seed', 42)
        factor_num_file = os.path.join(self.rd_save_dir, f'factor_num_{self.id}_seed_{seed_value}.npy')
        
        # 🔧 MODIFIED: 常に10因子に固定（ファイル読み込みをコメントアウト）
        self.factor_num = 10
        print(f"🔧 [FACTOR_NUM] Fixed to 10 factors (file loading disabled)")
        
        # ファイルに10を保存（念のため）
        np.save(factor_num_file, self.factor_num)
        print(f"💾 [DEBUG] Saved factor_num=10 to file: {factor_num_file}")

    def func_forward(self, obs, func_str, device):  # 関数を用いて前方計算を行うメソッド
        if isinstance(obs, torch.Tensor):  # obsがTensor型の場合
            obs = obs.cpu().numpy()  # numpy配列に変換
        bs, n_agents, t, _ = obs.shape  # obsの形状を取得
        array_obs = obs.reshape(-1, obs.shape[-1])  # obsをリシェイプ
        
        func = func_str  # 関数文字列を取得

        import heapq
        import math
        namespace = {
            'np': np,
            'numpy': np,
            'heapq': heapq,
            'math': math
        }  # 名前空間を設定

        try:
            exec(func, namespace)  # 関数を実行
            evaluation_func = namespace['evaluation_func']  # 評価関数を取得
            factor_scores = evaluation_func(array_obs)  # 因子スコアを計算array_obsを引数に
            cat_factor_scores = np.concatenate(factor_scores, axis=-1)  # 因子スコアを結合
            tensor_scores = th.tensor(cat_factor_scores.reshape(bs, n_agents, t, -1)).float().to(device)  # Tensor型に変換
            return tensor_scores  # スコアを返す
        except Exception as e:  # エラーが発生した場合
            fallback_scores = np.random.randn(bs * n_agents * t, getattr(self, 'factor_num', 10))  # フォールバックスコアを生成
            tensor_scores = th.tensor(fallback_scores.reshape(bs, n_agents, t, -1)).float().to(device)  # Tensor型に変換
            return tensor_scores  # スコアを返す

    def forward(self, states, actions, episode_length, next_states=None, return_tensor_scores=False):
        # RD関数が存在しない場合の処理
        if not hasattr(self, 'rd_functions') or len(self.rd_functions) == 0:
            from factor_chat_with_gpt import generate_fallback_drp_response
            fallback_response = generate_fallback_drp_response()
            self.rd_functions = [fallback_response['Functions']]
            if not hasattr(self, 'factor_num'):
                self.factor_num = 10
            
        func = self.rd_functions[0]
        b, na, t, d = states.shape
        
        # LARE用の単一エージェント処理か、通常のマルチエージェント処理かを判定
        if na == 1:
            # 単一エージェントの場合はエラーチェックをスキップ
            pass
        elif na == self.n_agents:
            # 通常のマルチエージェント処理
            pass
        else:
            # 予期しないエージェント数の場合はエラー
            raise ValueError(f"Invalid agent count: got {na}, expected {self.n_agents} or 1")
        
        # 既存のコードと同じ処理
        if self.args.only_s:
            tensor_scores = torch.tensor(states).float().to(self.reward_model.device)
            if getattr(self.args, 'use_next_state', False):
                next_tensor_scores = torch.tensor(next_states).float().to(self.reward_model.device)
        else:
            tensor_scores = self.func_forward(states, func, self.reward_model.device)
            if getattr(self.args, 'use_next_state', False):
                next_tensor_scores = self.func_forward(next_states, func, self.reward_model.device)

        if getattr(self.args, 'use_next_state', False):
            tensor_scores = next_tensor_scores
        
        tensor_scores = tensor_scores.reshape(b*na, t, -1)  # スコアをリシェイプ
        rewards = self.reward_model(tensor_scores)  # 報酬を計算
        rewards = rewards.reshape(b, na, t, -1)  # 報酬をリシェイプ
        tensor_scores = tensor_scores.reshape(b, na, t, -1)  # スコアをリシェイプ
        
        if return_tensor_scores:
            return rewards, tensor_scores
        else:
            return rewards
    
    def get_reward(self, obs, next_obs=None, debug_agent_id=None, debug_step=None):
        """
        観測から報酬を計算する（デバッグ機能付き）
        
        Args:
            obs (np.ndarray): 観測配列 (321,)
            next_obs (np.ndarray, optional): 次の観測配列
            debug_agent_id (int, optional): デバッグ用エージェントID
            debug_step (int, optional): デバッグ用ステップ数
        
        Returns:
            float: 計算された報酬値
        """
        try:
            # 観測を適切な形状に変換
            if obs.ndim == 1:
                # (321,) -> (1, 1, 1, 321) にバッチ次元を追加
                obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).unsqueeze(0).unsqueeze(0)
            else:
                obs_tensor = torch.from_numpy(obs).float()
            
            # 次の観測も同様に処理
            if next_obs is not None:
                if next_obs.ndim == 1:
                    next_obs_tensor = torch.from_numpy(next_obs).float().unsqueeze(0).unsqueeze(0).unsqueeze(0)
                else:
                    next_obs_tensor = torch.from_numpy(next_obs).float()
            else:
                next_obs_tensor = None
            
            # forwardメソッドを呼び出す前に、evaluation_funcを直接呼び出してデバッグ情報を取得
            if debug_step is not None and debug_step <= 3:
                self._debug_evaluation_func(obs, debug_agent_id, debug_step)
            
            # forwardメソッドを呼び出して報酬を計算
            with torch.no_grad():
                # episode_length=1として単一ステップの報酬を計算
                rewards = self.forward(
                    states=obs_tensor,
                    actions=None,  # アクションは不要
                    episode_length=1,
                    next_states=next_obs_tensor
                )
            
            # 報酬テンソルから単一の値を抽出
            if rewards.numel() == 1:
                reward_value = rewards.item()
            else:
                # 複数の報酬がある場合は合計を取る
                reward_value = rewards.sum().item()
            
            return reward_value
            
        except Exception as e:
            error_msg = f"❌ [ERROR] Failed to calculate reward: {e}"
            if debug_step is not None and debug_step <= 3:
                print(error_msg)
                import traceback
                print(f"🔍 [DEBUG] Full traceback: {traceback.format_exc()}")
            else:
                print(error_msg)
            return 0.0  # エラー時はデフォルト報酬を返す

    def _debug_evaluation_func(self, obs, debug_agent_id, debug_step):
        """
        evaluation_funcを直接呼び出してデバッグ情報を取得（エラー修正版）
        """
        try:
            # RD関数が存在するかチェック
            if not hasattr(self, 'rd_functions') or len(self.rd_functions) == 0:
                print(f"⚠️ [WARNING] No RD functions available, using fallback")
                return
            
            # 観測を適切な形状に変換
            if obs.ndim == 1:
                obs_reshaped = obs.reshape(1, -1)
            else:
                obs_reshaped = obs
            
            # evaluation_funcを実行
            func_str = self.rd_functions[0]
            
            # 安全な名前空間を設定
            namespace = {
                'np': np, 
                'numpy': np,
                '__builtins__': __builtins__,
                'print': print,
                'len': len,
                'range': range,
                'enumerate': enumerate,
                'zip': zip,
                'sum': sum,
                'max': max,
                'min': min,
                'abs': abs,
                'round': round,
                'float': float,
                'int': int,
                'str': str,
                'bool': bool,
                'list': list,
                'dict': dict,
                'tuple': tuple,
            }
            
            # 関数を実行
            exec(func_str, namespace)
            evaluation_func = namespace['evaluation_func']
            
            # 因子を計算
            factors = evaluation_func(obs_reshaped)
            
            
                            
        except Exception as e:
            print(f"❌ [ERROR] Failed to debug evaluation_func: {e}")
            import traceback
            print(f"🔍 [DEBUG] Traceback: {traceback.format_exc()}")

# DRP用のエイリアス（後方互換性）
DRPFactorRewardDecomposer = FactorRewardDecomposer