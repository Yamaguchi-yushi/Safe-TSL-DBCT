import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

class HAPPOMixer(nn.Module):
    def __init__(self, args):
        super(HAPPOMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        self.embed_dim = args.mixing_embed_dim

        # ハイパーネットワークの層数に応じた構造
        if getattr(args, "hypernet_layers", 1) == 1:
            self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
            self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
        elif getattr(args, "hypernet_layers", 1) == 2:
            hypernet_embed = self.args.hypernet_embed
            self.hyper_w_1 = nn.Sequential(
                nn.Linear(self.state_dim, hypernet_embed),
                nn.ReLU(),
                nn.Linear(hypernet_embed, self.embed_dim * self.n_agents),
            )
            self.hyper_w_final = nn.Sequential(
                nn.Linear(self.state_dim, hypernet_embed),
                nn.ReLU(),
                nn.Linear(hypernet_embed, self.embed_dim),
            )
        else:
            raise Exception("Error setting number of hypernet layers.")

        # 状態依存のバイアス（隠れ層用）
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # 最終層用の状態価値関数
        self.V = nn.Sequential(
            nn.Linear(self.state_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1),
        )

        # HAPPO特有の注意機構
        self.attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=args.attention_heads,
            dropout=args.attention_dropout
        )

    def forward(self, agent_values, states):
        bs = agent_values.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_values = agent_values.view(-1, 1, self.n_agents)

        # 第1層
        w1 = th.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(th.bmm(agent_values, w1) + b1)

        # 注意機構による重み付け
        hidden = hidden.transpose(0, 1)  # 注意機構用にシーケンス次元を入れ替え
        attn_out, _ = self.attention(hidden, hidden, hidden)
        hidden = attn_out.transpose(0, 1)  # 元の形状に戻す

        # 第2層
        w_final = th.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        
        # 状態依存のバイアス
        v = self.V(states).view(-1, 1, 1)
        
        # 最終出力の計算
        y = th.bmm(hidden, w_final) + v
        
        # 形状を整えて返す
        v_tot = y.view(bs, -1, 1)
        return v_tot

    def get_attention_weights(self, agent_values, states):
        """注意の重みを取得するメソッド（可視化用）"""
        states = states.reshape(-1, self.state_dim)
        agent_values = agent_values.view(-1, 1, self.n_agents)

        w1 = th.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(th.bmm(agent_values, w1) + b1)

        hidden = hidden.transpose(0, 1)
        _, attn_weights = self.attention(hidden, hidden, hidden)
        
        return attn_weights