import torch  # torchライブラリをインポート
import numpy as np  # numpyライブラリをnpとしてインポート
import sys, os  # sys.path 操作用
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'epymarl', 'src'))  # MARL4DRP の epymarl/src を import 解決パスに追加
from modules.agents.rnn_agent import RNNAgent  # EPyMARLのRNNAgentをインポート

class DummyArgs:  # DummyArgsクラスの定義
    def __init__(self, hidden_dim=64, n_actions=None, use_rnn=False):  # コンストラクタの定義（デフォルト引数付き）
        self.hidden_dim = hidden_dim  # hidden_dimをインスタンス変数に設定
        self.n_actions = n_actions  # n_actionsをインスタンス変数に設定
        self.use_rnn = use_rnn  # use_rnnをインスタンス変数に設定

class PolicyRunner:  # PolicyRunnerクラスの定義
    def __init__(self, model_path, input_shape, n_actions, agent_num):  # コンストラクタの定義
        self.args = DummyArgs(hidden_dim=64, n_actions=n_actions, use_rnn=False)  # DummyArgsのインスタンスを生成
        self.agent = RNNAgent(input_shape, self.args)  # RNNAgentのインスタンスを生成
        self.agent.load_state_dict(torch.load(model_path, map_location='cpu'))  # モデルの重みデータをロード
        self.agent.eval()  # エージェントを評価モードに設定
        self.hidden_states = [self.agent.init_hidden() for _ in range(agent_num)]  # 各エージェント用の隠れ状態を初期化

    def init_hidden_state(self):  # 初期隠れ状態を生成する関数
        return self.agent.init_hidden()

    def get_action(self, ag_idx, obs, avail_actions):  # アクションを取得するメソッド定義
        # 指定エージェントの隠れ状態が足りなければ拡張する
        if ag_idx >= len(self.hidden_states):
            # 足りない分だけ初期化された隠れ状態を追加
            for _ in range(ag_idx - len(self.hidden_states) + 1):
                self.hidden_states.append(self.init_hidden_state())  # 初期隠れ状態を生成する関数
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)  # 観測値をテンソルに変換し、次元拡張
        # エージェントのfc1レイヤが受け付ける入力次元を取得要素にスライス）
        expected_dim = self.agent.fc1.in_features  
        obs_tensor = obs_tensor[:, :expected_dim]  # 常に期待する次元に合わせてスライス
        h_in = self.hidden_states[ag_idx]  # 指定エージェントの隠れ状態を取得

        q_values, h_out = self.agent(obs_tensor, h_in)  # エージェントの順伝播を実施しQ値と新しい隠れ状態を取得        q_values, h_out = self.agent(obs_tensor, h_in)  # エージェントの順伝播を実施しQ値と新しい隠れ状態を取得
        self.hidden_states[ag_idx] = h_out.detach()  # 新しい隠れ状態を保存（計算グラフから切り離す）

        q_numpy = q_values.squeeze(0).detach().numpy()  # Q値テンソルをnumpy配列に変換        q_numpy = q_values.squeeze(0).detach().numpy()  # Q値テンソルをnumpy配列に変換
        masked_q = [q_numpy[a] if a in avail_actions else -np.inf for a in range(len(q_numpy))]  # 利用可能なアクションのみQ値を残すrange(len(q_numpy))]  # 利用可能なアクションのみQ値を残す
        
        #####################################  # 区切り行（必要に応じて処理を追加）#####################################  # 区切り行（必要に応じて処理を追加）
        return int(np.argmax(masked_q))  # 最大のQ値を持つアクションのインデックスを返す
