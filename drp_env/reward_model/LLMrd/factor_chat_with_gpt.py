import requests  # HTTPリクエスト用ライブラリ
import os  # OS操作用ライブラリ
import json  # JSON操作用ライブラリ
import sys  # システム操作用ライブラリ
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # 現在のファイルのディレクトリをパスに追加
from prompt_template import *  # プロンプトテンプレートをインポート
import numpy as np  # 数値計算用ライブラリ
import argparse  # コマンドライン引数解析用ライブラリ
import torch as th  # PyTorchライブラリ
from openai import OpenAI  # OpenAI APIクライアント

global_port = 8001  # グローバルポート番号の設定

def inference(model, message, temperature, host, n=1, seed=None, map_name="map_3x3", agent_num=3):
    """GPTモデルとの推論を実行する関数"""
    print(f"🔗 [DEBUG] Starting inference with model: {model}, temperature: {temperature}, n: {n}")  # デバッグ出力
    
    # APIリクエスト用のJSONデータを作成
    data = json.dumps({
        'model': model,  # 使用するモデル名
        'messages': message,  # 送信するメッセージ
        'temperature': temperature,  # 生成の創造性パラメータ
        'response_format': {"type": "json_object"},  # JSON形式での応答を要求
        'n': n  # 生成する応答の数
    })
    correct_inference = False  # 正常な推論が完了したかのフラグ
    count = 0  # 再試行回数のカウンタ
    out_content = []  # 出力内容を格納するリスト
    
    while not correct_inference:  # 正常な推論が完了するまでループ
        count += 1  # 試行回数をカウントアップ
        print(f"🔄 [DEBUG] API call attempt {count}")  # デバッグ出力
        
        if count >= 1:  # 最大試行回数に達した場合
            print("❌ [DEBUG] Max retry attempts reached. Using fallback response.")  # 最大試行回数到達のメッセージ
            # フォールバック応答を生成
            fallback_response = generate_fallback_drp_response(map_name, agent_num)  # DRP用フォールバック応答を生成
            if fallback_response is None:
                print("❌ [FALLBACK] No fallback function available. Cannot proceed.")  # フォールバック関数がない場合のエラーメッセージ
                return None
            
            out_content.append(json.dumps(fallback_response))  # フォールバック応答をリストに追加
            break  # ループを終了
        try:
            # APIキーの存在確認
            if not os.environ.get('OPENAI_API_KEY'):
                print("⚠️  [DEBUG] OPENAI_API_KEY not found in environment variables")  # デバッグ出力
                raise Exception("API key not found")
            
            client = OpenAI()  # OpenAIクライアントを初期化
            print(f"🌐 [DEBUG] Making API call to OpenAI...")  # デバッグ出力
            
            # チャット補完APIを呼び出し
            out = client.chat.completions.create(
                model=model,  # 使用するモデル
                messages=message,  # 送信するメッセージ
                temperature=temperature,  # 創造性パラメータ
                response_format={"type": "json_object"},  # JSON応答形式
                n=n  # 応答数
            )
            
            print(f"✅ [DEBUG] API call successful, processing {len(out.choices)} responses")  # デバッグ出力
            
            for i in range(n):  # 各応答を処理
                out_content.append(out.choices[i].message.content)  # 応答内容をリストに追加
                print(f"📝 [DEBUG] Response {i+1} length: {len(out.choices[i].message.content)} characters")  # デバッグ出力
            correct_inference = True  # 正常完了フラグをTrue
            
        except KeyboardInterrupt:  # キーボード割り込み（Ctrl+C）の場合
            print("⚠️  [DEBUG] Keyboard interrupt detected")  # デバッグ出力
            break  # ループを終了
        except Exception as e:  # その他の例外の場合
            print(f'❌ [DEBUG] Error: {e}. Retrying...')  # エラーメッセージと再試行の通知
            if count < 5:
                print(f"🔄 [DEBUG] Waiting 2 seconds before retry...")  # デバッグ出力
                import time
                time.sleep(2)
            
    print(f"✅ [DEBUG] Inference completed with {len(out_content)} responses")  # デバッグ出力
    return out_content  # 生成された応答内容を返す

def load_fallback_function(map_name, agent_num):
    """
    マップ名とエージェント数に基づいてフォールバック報酬関数をロードする

    Args:
        map_name (str): マップ名（例: 'map_5x4'）
        agent_num (int): エージェント数（例: 3）

    Returns:
        function: フォールバック報酬関数
    """
    import importlib.util
    import os

    fallback_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'fallback_functions'
    )

    filename = f'evaluation_func.py'
    filepath = os.path.join(fallback_dir, filename)

    print(f"🔍 [FALLBACK] Searching for fallback function:")
    print(f"   - Map: {map_name}")
    print(f"   - Agents: {agent_num}")
    print(f"   - Required file: {filename}")
    print(f"   - Search directory: {fallback_dir}")
    
    if not os.path.exists(fallback_dir):
        print(f"❌ [FALLBACK] Directory does not exist: {fallback_dir}")
        print(f"❌ [FALLBACK] Please create the directory and add fallback function files")
        return None
    
    if not os.path.isfile(filepath):
        print(f"❌ [FALLBACK] Required file not found: {filename}")
        print(f"❌ [FALLBACK] Please create the file at: {filepath}")
        print(f"")
        print(f"📝 [FALLBACK] Available files in directory:")

        try:
            # fallback_dirディレクトリ内のファイル一覧を取得し、.pyで終わるファイルのみを抽出
            available_files = [f for f in os.listdir(fallback_dir) if f.endswith('.py')]
            
            if available_files:
                # .pyファイルが存在する場合、アルファベット順にソートして表示
                for available_file in sorted(available_files):
                    print(f"   - {available_file}")
            else:
                # .pyファイルが1つも見つからない場合
                print(f"   (No .py files found)")
                
        except Exception as e:
            # ディレクトリの読み取りに失敗した場合（権限エラーなど）
            print(f"   (Could not list directory: {e})")

        return None  # 必要なファイルが見つからなかったのでNoneを返す
    
    try:
        print(f"✅ [FALLBACK] Found required file: {filename}")

        spec = importlib.util.spec_from_file_location("fallback_module", filepath)
        fallback_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fallback_module)

        if not hasattr(fallback_module, 'evaluation_func'):
            print(f"❌ [FALLBACK] 'evaluation_func' not found in {filename}")
            return None
        
        import inspect
        func_source = inspect.getsource(fallback_module.evaluation_func)

        factor_num = getattr(fallback_module, 'FACTOR_NUMBER', None)
        if factor_num is None:
            print(f"❌ [FALLBACK] 'FACTOR_NUMBER' not defined in {filename}")
            return None
        
        print(f"✅ [FALLBACK] Successfully loaded 'evaluation_func' from {filename}")
        print(f"   - File: {filename}")
        print(f"  - Factor count: {factor_num}")

        return {
            "Functions": func_source,
            "factor_number": factor_num
        }
    
    except Exception as e:
        print(f"❌ [FALLBACK] Error loading fallback function from {filename}: {e}")
        import traceback
        print(f"❌ [FALLBACK] Traceback:\n{traceback.format_exc()}")
        return None

def generate_fallback_drp_response(map_name="map_3x3", agent_num=3):
    """
    DRP環境用のフォールバック応答を生成（ファイルベース、厳格モード）
    
    Args:
        map_name (str): マップ名
        agent_num (int): エージェント数
    
    Returns:
        dict or None: フォールバック応答、失敗時はNone
    """
    print(f"")
    print(f"=" * 80)
    print(f"🔄 [FALLBACK] Generating fallback response")
    print(f"   - Map: {map_name}")
    print(f"   - Agents: {agent_num}")
    print(f"=" * 80)
    
    result = load_fallback_function(map_name, agent_num)
    
    if result is None:
        print(f"")
        print(f"=" * 80)
        print(f"❌ [FALLBACK] FATAL ERROR: Could not load fallback function")
        print(f"=" * 80)
        print(f"")
    else:
        print(f"")
        print(f"=" * 80)
        print(f"✅ [FALLBACK] Fallback function loaded successfully")
        print(f"=" * 80)
        print(f"")
    
    return result


def save_readable_format(out_content, all_message, factor_num, save_dir, id, seed):
    import datetime
    import os
    
    # 読みやすい形式のファイル名を生成
    response_txt_file = os.path.join(save_dir, f'response_{id}_seed_{seed}.txt')
    dialog_json_file = os.path.join(save_dir, f'dialog_{id}_seed_{seed}.json')
    summary_file = os.path.join(save_dir, f'summary_{id}_seed_{seed}.md')
    
    try:
        # 1. 応答内容をテキストファイルに保存
        with open(response_txt_file, 'w', encoding='utf-8') as f:
            f.write(f"# DRP Reward Function Response\n")
            f.write(f"=" * 50 + "\n")
            f.write(f"Seed: {seed}\n")
            f.write(f"Generated at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Factor count: {factor_num}\n")
            f.write(f"Response count: {len(out_content)}\n")
            f.write(f"=" * 50 + "\n\n")
            
            for i, content in enumerate(out_content):
                f.write(f"{'='*20} Response {i+1} {'='*20}\n\n")
                try:
                    parsed = json.loads(content)
                    f.write(f"Functions:\n")
                    f.write(f"{'-'*40}\n")
                    function_code = parsed.get('Functions', 'No functions found')
                    f.write(function_code)
                    f.write(f"\n{'-'*40}\n\n")
                    
                    # その他のキーも保存
                    for key, value in parsed.items():
                        if key != 'Functions':
                            f.write(f"{key}:\n")
                            f.write(f"{'-'*20}\n")
                            f.write(f"{value}\n\n")
                except json.JSONDecodeError:
                    f.write(f"Raw content:\n")
                    f.write(f"{'-'*40}\n")
                    f.write(f"{content}\n")
                    f.write(f"{'-'*40}\n\n")
        
        # 2. 対話履歴をJSONファイルに保存
        with open(dialog_json_file, 'w', encoding='utf-8') as f:
            dialog_data = {
                "metadata": {
                    "seed": seed,
                    "generated_at": datetime.datetime.now().isoformat(),
                    "factor_count": factor_num,
                    "response_count": len(out_content)
                },
                "dialog": all_message
            }
            json.dump(dialog_data, f, ensure_ascii=False, indent=2)
        
        # 3. 概要をMarkdownファイルに保存
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"# DRP Reward Function Summary\n\n")
            f.write(f"## Basic Information\n")
            f.write(f"- **Seed**: {seed}\n")
            f.write(f"- **Generated at**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"- **Factor count**: {factor_num}\n")
            f.write(f"- **Response count**: {len(out_content)}\n\n")
            
            f.write(f"## Generated Functions\n\n")
            for i, content in enumerate(out_content):
                f.write(f"### Function {i+1}\n\n")
                try:
                    parsed = json.loads(content)
                    function_code = parsed.get('Functions', 'No functions found')
                    # 関数の最初の20行を表示
                    lines = function_code.split('\n')
                    f.write(f"```python\n")
                    for j, line in enumerate(lines[:20]):
                        f.write(f"{line}\n")
                    if len(lines) > 20:
                        f.write(f"... (truncated, total {len(lines)} lines)\n")
                    f.write(f"```\n\n")
                    
                    # 関数の統計情報
                    f.write(f"**Function Statistics:**\n")
                    f.write(f"- Total lines: {len(lines)}\n")
                    f.write(f"- Contains factors: {function_code.count('factors.append')}\n")
                    f.write(f"- Contains numpy: {'numpy' in function_code or 'np.' in function_code}\n\n")
                    
                except json.JSONDecodeError:
                    f.write(f"```\n{content[:300]}...\n```\n\n")
            
            f.write(f"## Dialog Summary\n\n")
            f.write(f"- **Total messages**: {len(all_message)}\n")
            
            user_messages = [m for m in all_message if isinstance(m, dict) and m.get('role') == 'user']
            assistant_messages = [m for m in all_message if isinstance(m, dict) and m.get('role') == 'assistant']
            
            f.write(f"- **User messages**: {len(user_messages)}\n")
            f.write(f"- **Assistant messages**: {len(assistant_messages)}\n\n")
            
            # 最後のメッセージの概要
            if all_message:
                f.write(f"### Last Message\n")
                last_msg = all_message[-1]
                if isinstance(last_msg, dict):
                    f.write(f"- **Role**: {last_msg.get('role', 'unknown')}\n")
                    content = last_msg.get('content', '')
                    f.write(f"- **Content preview**: {content[:100]}...\n")
        
        print(f"📝 [DEBUG] Saved readable format files:")
        print(f"   - TXT: {response_txt_file}")
        print(f"   - JSON: {dialog_json_file}")
        print(f"   - MD: {summary_file}")
        
    except Exception as e:
        print(f"❌ [DEBUG] Error saving readable format: {e}")

def call_drp_gpt(env_name, map_name, save_dir, save=True, id=0, use_recheck=False, n=2, port=8000, seed=0):
    """DRP環境用のGPT呼び出し関数"""
    print(f"🚀 [DEBUG] Starting call_drp_gpt with parameters:")
    print(f"   - env_name: {env_name}")
    print(f"   - map_name: {map_name}")
    print(f"   - save_dir: {save_dir}")
    print(f"   - n: {n}")
    print(f"   - seed: {seed}")
    print(f"   - id: {id}")
    
    # シード値を含むファイル名を生成
    response_file = os.path.join(save_dir, f'response_{id}_seed_{seed}.npy')
    dialog_file = os.path.join(save_dir, f'dialog_{id}_seed_{seed}.npy')
    factor_num_file = os.path.join(save_dir, f'factor_num_{id}_seed_{seed}.npy')
    
    print(f"📋 [DEBUG] Target files:")
    print(f"   - response: {response_file}")
    print(f"   - dialog: {dialog_file}")
    print(f"   - factor_num: {factor_num_file}")
    
    if port is None:
        port = global_port
        
    # prompt_template.pyから取得（get_drp_prompt関数をprompt_template.pyに移動）
    print(f"📝 [DEBUG] Generating prompt for {env_name} on {map_name}")  # デバッグ出力
    prompt = get_prompt(env_name, map_name, factor_decomp=True)
    
    message = prompt.get_message()
    all_message = prompt.get_message()
    start_idx = len(all_message)
    print(f"📋 [DEBUG] Initial message length: {len(message)}")  # デバッグ出力
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"📁 [DEBUG] Created save directory: {save_dir}")  # デバッグ出力
    
    # DRP環境用の簡潔な再確認メッセージ
    recheck_message = [
        {
            'role': 'user', 
            'content': "You have generated several evaluation functions. \
Please summarize them and generate a new evaluation function that incorporates all the evaluation factors.\
If there are other important evaluation factors, please include them as well."
        }
    ]
    
    if n == 1:
        recheck_message = []

    # モデル設定（DRP用に調整）
    model = 'gpt-4o'  # 使用するGPTモデル
    init_temperature = 0.8
    check_temperature = 0.3
    host = None
    
    print(f"🧠 [DEBUG] Generating DRP reward functions for {env_name} on {map_name}...")  # DRP報酬関数生成開始のメッセージ
    
    # 初回推論
    print(f"🔄 [DEBUG] Starting initial inference...")  # デバッグ出力
    out_content = inference(model, message, init_temperature, host, n=n, seed=seed, map_name=map_name, agent_num=agent)  # GPTに推論を実行
    message.append({'role': 'assistant', 'content': str(out_content)})  # アシスタントの応答をメッセージに追加
    all_message.append({'role': 'assistant', 'content': str(out_content)})  # 全メッセージにも追加
    print("✅ [DEBUG] Initial response generated:")  # 初回応答生成完了のメッセージ
    print(f"📊 [DEBUG] Response count: {len(out_content)}")  # デバッグ出力
    
    check_phases = len(recheck_message) if len(recheck_message) > 0 else 1  # 確認フェーズ数を決定
    print(f"🔍 [DEBUG] Check phases: {check_phases}")  # デバッグ出力

    for i in range(check_phases):  # 各確認フェーズに対して
        print(f"🔄 [DEBUG] Check phase {i+1}/{check_phases}")  # デバッグ出力
        
        if len(recheck_message) > 0:  # 再確認メッセージが存在する場合
            new_content = recheck_message.pop(0)['content']  # 次の再確認メッセージを取得
            message.append({'role': 'user', 'content': new_content})  # ユーザーメッセージを追加
            all_message.append({'role': 'user', 'content': new_content})  # 全メッセージにも追加
            print(f"🔄 [DEBUG] Starting recheck inference...")  # デバッグ出力
            out_content = inference(model, message, check_temperature, host, seed=seed)  # 再推論を実行
            print("✅ [DEBUG] Recheck response generated:")  # 再確認応答生成完了のメッセージ
            print(f"📊 [DEBUG] Recheck response count: {len(out_content)}")  # デバッグ出力

            message.append({'role': 'assistant', 'content': str(out_content)})  # アシスタント応答を追加
            all_message.append({'role': 'assistant', 'content': str(out_content)})  # 全メッセージにも追加

        # DRP用の関数チェック
        for recheck_count in range(3):  # 最大3回まで再確認
            print(f"🔍 [DEBUG] Function check attempt {recheck_count + 1}/3")  # デバッグ出力
            pass_check, error_idx, error_content, factor_num = prompt.factor_check(out_content)  # 関数チェックを実行
            print(f"📈 [DEBUG] Function check: {'PASS' if pass_check else 'FAIL'}, Factors: {factor_num}")  # チェック結果を表示
            
            if pass_check:  # チェックが通った場合
                print(f"✅ [DEBUG] Function check passed on attempt {recheck_count + 1}")  # デバッグ出力
                break  # ループを終了
                
            print(f"⚠️  [DEBUG] Recheck attempt {recheck_count + 1}: {error_content}")  # 再確認試行のメッセージ
            
            if recheck_count == 0:  # 初回の再確認の場合
                message[-1] = {'role': 'assistant', 'content': out_content[error_idx]}  # 最後のアシスタントメッセージを更新
                message.append({'role': 'user', 'content': error_content})  # エラー内容をユーザーメッセージとして追加
                all_message.append({'role': 'user', 'content': error_content})  # 全メッセージにも追加
            else:  # 2回目以降の再確認の場合
                message[-2] = {'role': 'assistant', 'content': out_content[error_idx]}  # アシスタントメッセージを更新
                message[-1] = {'role': 'user', 'content': error_content}  # ユーザーメッセージを更新
                all_message.append({'role': 'assistant', 'content': out_content[error_idx]})  # 全メッセージに追加
                all_message.append({'role': 'user', 'content': error_content})  # 全メッセージに追加
                
            print(f"🔄 [DEBUG] Starting correction inference...")  # デバッグ出力
            out_content = inference(model, message, check_temperature, host, seed=seed)  # 修正版を再推論
            all_message.append({'role': 'assistant', 'content': str(out_content)})  # 修正応答を全メッセージに追加
            print("✅ [DEBUG] Correction response generated")  # 修正応答のメッセージ

        # 保存処理
        if save and pass_check:  # 保存が有効でチェックが通った場合
            print(f"💾 [DEBUG] Saving successful response with seed {seed}...")  # デバッグ出力
            np.save(response_file, out_content)  # 応答内容を保存
            np.save(dialog_file, all_message)  # 対話履歴を保存
            np.save(factor_num_file, factor_num)  # 因子数を保存
            
            # 読みやすい形式での保存を追加
            save_readable_format(out_content, all_message, factor_num, save_dir, id, seed)
            
            print(f"✅ [DEBUG] Successfully saved DRP reward functions with {factor_num} factors (seed: {seed})")  # 保存成功のメッセージ
            break  # ループを終了
        elif not pass_check:  # チェックが失敗した場合
            print(f"⚠️  [DEBUG] Warning: Function check failed after all attempts. Using fallback (seed: {seed}).")  # チェック失敗の警告
            # フォールバック保存
            fallback_content = [json.dumps(generate_fallback_drp_response())]  # フォールバック内容を生成
            if save:  # 保存が有効な場合
                print(f"💾 [DEBUG] Saving fallback response with seed {seed}...")  # デバッグ出力
                np.save(response_file, fallback_content)  # フォールバック内容を保存
                np.save(dialog_file, all_message)  # 対話履歴を保存
                np.save(factor_num_file, 8)  # フォールバックは8因子
                
                # 読みやすい形式での保存を追加
                save_readable_format(fallback_content, all_message, 8, save_dir, id, seed)
                
                print(f"✅ [DEBUG] Fallback response saved with 8 factors (seed: {seed})")  # デバッグ出力

def test_drp_reward_generation():  # DRP報酬生成のテスト関数
    """DRP報酬生成のテスト"""
    print("🧪 [DEBUG] Testing DRP reward function generation...")  # テスト開始のメッセージ
    
    # テスト用ディレクトリ
    test_dir = "/tmp/drp_test_rewards"  # テスト用の保存ディレクトリ
    
    # DRP用報酬関数生成
    call_drp_gpt(
        env_name="drp",  # 環境名
        map_name="map_3x3",  # マップ名
        save_dir=test_dir,  # 保存ディレクトリ
        save=True,  # 保存フラグ
        id=0,  # ID
        n=1,  # 生成数
        seed=42  # 乱数シード
    )
    
    # 生成された関数をテスト
    response_file = os.path.join(test_dir, f"response_0_seed_42.npy")  # 応答ファイルのパス
    print(f"🔍 [DEBUG] Checking for response file: {response_file}")  # デバッグ出力
    
    if os.path.exists(response_file):  # 応答ファイルが存在する場合
        print(f"✅ [DEBUG] Response file found, loading...")  # デバッグ出力
        response = np.load(response_file, allow_pickle=True)  # 応答を読み込み
        print("🔍 [DEBUG] Generated function test:")  # 生成関数テストのメッセージ
        print(f"📊 [DEBUG] Response type: {type(response)}, length: {len(response)}")  # デバッグ出力
        
        # 実際に実行してみる
        try:
            func_data = json.loads(response[0])  # JSON形式の応答を解析
            func_str = func_data['Functions']  # 関数文字列を取得
            print(f"📝 [DEBUG] Function string length: {len(func_str)}")  # デバッグ出力
            
            namespace = {'np': np, 'numpy': np}  # 実行環境を設定
            exec(func_str, namespace)  # 関数を定義
            
            test_obs = np.array([[0.0, 0.0, 0, 5], [1.0, 1.0, 1, 4]])  # テスト観測を作成
            print(f"🧪 [DEBUG] Test observation shape: {test_obs.shape}")  # デバッグ出力
            
            result = namespace['evaluation_func'](test_obs)  # 関数を実行
            print(f"✅ [DEBUG] Test execution successful: {len(result)} factors")  # 実行成功のメッセージ
            
            for i, factor in enumerate(result):  # 各因子の情報を表示
                print(f"📈 [DEBUG] Factor {i} shape: {factor.shape}, sample values: {factor[:3]}")  # 因子の形状とサンプル値
                
        except Exception as e:  # 実行に失敗した場合
            print(f"❌ [DEBUG] Test execution failed: {e}")  # 失敗メッセージ
    else:
        print("❌ [DEBUG] No response file generated")  # 応答ファイルが生成されなかった場合のメッセージ