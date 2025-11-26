"""
LARE (Large Language Model Reward Engineering) Module

This module provides reward decomposition functionality using large language models
for multi-agent reinforcement learning environments.
"""

__version__ = "0.1.0"
__author__ = "DRP Team"

# 主要なクラスのインポート
try:
    from .factor_reward_decompose import FactorRewardDecomposer
    from .factor_reward_model import FactorRewardModel  # もし存在する場合
    from .prompt_template import PromptTemplate  # もし存在する場合
    
    # パッケージから直接インポート可能にする
    __all__ = [
        'FactorRewardDecomposer',
        'FactorRewardModel',
        'PromptTemplate',
    ]
    
    print("✅ LARE module imported successfully")
    
except ImportError as e:
    print(f"⚠️  Warning: Could not import some LARE components: {e}")
    # 最低限必要なものだけインポート
    try:
        from .factor_reward_decompose import FactorRewardDecomposer
        __all__ = ['FactorRewardDecomposer']
    except ImportError:
        __all__ = []
        print("❌ Critical: Could not import FactorRewardDecomposer")

# モジュールレベルの設定
import os
import sys

# 現在のディレクトリをPythonパスに追加
current_dir = os.path.dirname(__file__)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)