# MARL4DRP - Multi-Agent Reinforcement Learning for Drone Routing Problem

マルチエージェント強化学習（MARL）を用いたドローン経路問題（DRP）のフレームワーク。
[EPyMARL](https://github.com/uoe-agents/epymarl) をベースに、カスタム DRP 環境を統合しています。

## Table of Contents

- [About](#about)
- [Installation](#installation)
- [Environment](#environment)
- [Running Experiments](#running-experiments)
- [Algorithms](#algorithms)
- [Configuration](#configuration)
- [File Structure](#file-structure)
- [Troubleshooting](#troubleshooting)

---

## About

本プロジェクトは、複数のドローンが地図上の目標地点へ衝突を避けながら移動する **Drone Routing Problem (DRP)** を MARL で解くフレームワークです。

- カスタム Gym 環境 (`drp_env`) による DRP のシミュレーション
- QMIX / MAPPO などの MARL アルゴリズムを EPyMARL で実行
- LLM ベースの報酬分解モデル（オプション）

---

## Installation

### 1. リポジトリのクローン

```bash
git clone <this-repo-url>
cd MARL4DRP
```

### 2. Conda 環境の作成

```bash
conda create -n drp_new python=3.9
conda activate drp_new
```

### 3. PyTorch のインストール（CUDA 12.8）

```bash
pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 torchaudio==2.7.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128
```

> CUDA バージョンに合わせて変更してください。CPU のみの場合は公式サイトを参照。

### 4. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

### 5. gym（OpenAI 版）のインストール

```bash
pip install git+https://github.com/openai/gym.git@c755d5c35a25ab118746e2ba885894ff66fb8c43
```

### 6. SMAC のインストール（StarCraft 環境を使う場合）

```bash
pip install git+https://github.com/oxwhirl/smac.git@d6aab33f76abc3849c50463a8592a84f59a5ef84
```

### 7. drp_env パッケージのインストール

```bash
pip install -e .
```

### 8. PAC アルゴリズムの依存関係（オプション）

```bash
pip install -r epymarl/pac_requirements.txt
```

---

## Environment

### Gym への登録と使い方

```python
import gym
import drp_env

env = gym.make("drp_env:drp-5agent_map_10x10-v2", state_repre_flag="onehot")
```

### 環境 ID の形式

```
drp_env:drp-{agent_num}agent_{map_name}-v2
```

| パラメータ | 説明 | 選択肢 |
|---|---|---|
| `agent_num` | エージェント数 | 1〜29 |
| `map_name` | マップ名 | `map_3x3`, `map_5x4`, `map_8x5`, `map_10x6`, `map_10x8`, `map_10x10`, `map_aoba00`, `map_aoba01`, `map_shibuya` |

### 状態表現（`state_repre_flag`）

| フラグ | 説明 |
|---|---|
| `coordinate` | 座標ベース |
| `onehot` | One-hot エンコーディング（全体観測） |
| `onehot_fov` | One-hot + 視野（部分観測） |
| `heu_onehot` | ヒューリスティック + One-hot |
| `heu_onehot_fov` | ヒューリスティック + One-hot + 視野 |

### 行動空間

各エージェントの行動は **ノード番号**（隣接ノードへの移動）。
無効な行動を選択した場合は現在位置に留まる。

### 報酬

| 報酬 | デフォルト値 | 説明 |
|---|---|---|
| `goal` | +100 | 目標地点に到達 |
| `collision` | -10 × speed | 他エージェントと衝突 |
| `wait` | -10 × speed | 同じ位置に停止 |
| `move` | -1 × speed | 移動（通常ペナルティ） |

### Info

| キー | 型 | 説明 |
|---|---|---|
| `distance_from_start` | `List[float]` | 各エージェントの出発地点からの距離 |
| `goal` | `bool` | 全エージェントが目標に到達したか |
| `collision` | `bool` | 衝突が発生したか |
| `timeup` | `bool` | 制限時間に達したか（EPyMARL 用） |

---

## Running Experiments

実験はすべて `epymarl/src/` ディレクトリから実行します。

```bash
cd epymarl/src
```

または conda 環境を指定して直接実行:

```bash
conda run -n drp_new python3 epymarl/src/main.py --config=<alg> --env-config=gymma with <overrides>
```

### QMIX の実行例

```bash
conda run -n drp_new python3 epymarl/src/main.py \
    --config=qmix \
    --env-config=gymma \
    with env_args.key="drp_env:drp-5agent_map_10x10-v2" \
         env_args.state_repre_flag="onehot" \
         env_args.time_limit=200
```

### MAPPO の実行例

```bash
conda run -n drp_new python3 epymarl/src/main.py \
    --config=mappo \
    --env-config=gymma \
    with env_args.key="drp_env:drp-5agent_map_10x10-v2" \
         env_args.state_repre_flag="onehot_fov" \
         env_args.time_limit=200
```

### 学習済みモデルの評価

```bash
conda run -n drp_new python3 epymarl/src/main.py \
    --config=qmix \
    --env-config=gymma \
    with env_args.key="drp_env:drp-5agent_map_10x10-v2" \
         checkpoint_path="epymarl/results/sacred/qmix/<run_id>/models" \
         evaluate=True
```

### 結果の保存先

```
epymarl/results/sacred/{algorithm}/{env_name}/{run_id}/
```

---

## Algorithms

EPyMARL で利用可能なアルゴリズム:

| カテゴリ | アルゴリズム | `--config` |
|---|---|---|
| Value-based | IQL | `iql` |
| Value-based | VDN | `vdn` |
| Value-based | QMIX | `qmix` |
| Value-based | QTRAN | `qtran` |
| Policy Gradient | IPPO | `ippo` |
| Policy Gradient | MAPPO | `mappo` |
| Policy Gradient | IA2C | `ia2c` |
| Actor-Critic | COMA | `coma` |
| Actor-Critic | MADDPG | `maddpg` |
| Pareto | PAC | `pac_ns` |

---

## Configuration

### 主要な設定ファイル

| ファイル | 説明 |
|---|---|
| `epymarl/src/config/default.yaml` | デフォルト設定（全アルゴリズム共通） |
| `epymarl/src/config/algs/qmix.yaml` | QMIX のハイパーパラメータ |
| `epymarl/src/config/algs/mappo.yaml` | MAPPO のハイパーパラメータ |
| `epymarl/src/config/envs/gymma.yaml` | 環境設定 |

### コマンドラインでの上書き

Sacred フレームワークにより、`with` 以降にキー=値形式で設定を上書き可能:

```bash
python3 main.py --config=qmix --env-config=gymma \
    with lr=0.001 buffer_size=2000 env_args.time_limit=300
```

---

## File Structure

```
MARL4DRP/
├── README.md
├── requirements.txt                        # pip 依存パッケージ一覧
├── drp_new_freeze_5080_cu128.txt           # RTX 5080 / CUDA 12.8 環境の完全フリーズ
├── setup.py                                # drp_env パッケージ定義
│
├── drp_env/                                # カスタム DRP 環境パッケージ
│   ├── __init__.py                         # Gym への環境登録
│   ├── drp_env.py                          # メイン環境クラス (DrpEnv)
│   ├── EE_map.py                           # マップ読み込み・グラフ構築 (NetworkX)
│   ├── map/                                # CSV マップファイル
│   ├── state_repre/                        # 状態表現モジュール
│   │   ├── coordinate.py
│   │   ├── onehot.py
│   │   ├── onehot_fov.py
│   │   ├── heu_onehot.py
│   │   └── heu_onehot_fov.py
│   ├── reward_model/                       # 報酬モデル
│   │   ├── base_reward_model.py
│   │   ├── mard/                           # MARD 報酬分解
│   │   ├── arel/                           # AREL 報酬モデル
│   │   └── LLMrd/                          # LLM ベース報酬分解 (OpenAI API)
│   ├── SafeMarlEnv/                        # 安全制約ラッパー
│   └── lare_central_trainer.py             # 中央集権型トレーナー
│
└── epymarl/                                # EPyMARL フレームワーク
    ├── README.md                           # EPyMARL 詳細ドキュメント
    ├── requirements.txt
    ├── pac_requirements.txt                # PAC アルゴリズム追加依存
    └── src/
        ├── main.py                         # エントリーポイント (Sacred)
        ├── run.py                          # 実行ループ
        ├── config/
        │   ├── default.yaml               # デフォルト設定
        │   ├── algs/                      # アルゴリズム別設定
        │   └── envs/                      # 環境別設定
        ├── components/
        │   └── episode_buffer.py          # リプレイバッファ
        ├── controllers/                   # マルチエージェントコントローラ
        ├── learners/                      # 学習アルゴリズム実装
        ├── runners/                       # エピソードランナー
        ├── modules/
        │   ├── agents/                    # RNN エージェント
        │   ├── critics/                   # クリティックネットワーク
        │   └── mixers/                    # 価値関数ミキサー (QMIX 等)
        └── envs/
            └── gymma.py                   # Gym 環境ラッパー
```

---

## Troubleshooting

### CUDA Out of Memory エラー

**症状**: `torch.OutOfMemoryError: CUDA out of memory` が `episode_buffer.py` で発生

**原因**: `qmix.yaml` の `buffer_cpu_only: False` によりリプレイバッファが GPU VRAM に配置される

**解決**: `epymarl/src/config/algs/qmix.yaml` を編集:

```yaml
buffer_cpu_only: True  # False → True に変更
```

VRAM 使用量が大幅に削減される（例: 14.39 GiB → 1.85 GiB）

---

### Sacred / モジュールが見つからない

`epymarl/src/` をカレントディレクトリにするか、パスを指定:

```bash
cd epymarl/src && python3 main.py ...
# または
PYTHONPATH=epymarl/src python3 epymarl/src/main.py ...
```

---

### gym バージョンの競合

本プロジェクトは `gym==0.21.x`（OpenAI git 版）と `gymnasium==1.1.1` の両方を使用します。
インストール順序の問題が生じた場合は再インストールしてください:

```bash
pip uninstall gym gymnasium -y
pip install git+https://github.com/openai/gym.git@c755d5c35a25ab118746e2ba885894ff66fb8c43
pip install gymnasium==1.1.1
```
