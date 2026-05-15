import gym  # gymモジュールをインポート
import time  # 時間計測用モジュールをインポート
import json  # JSON操作用モジュールをインポート
from datetime import datetime  # 日付時刻操作のためdatetimeをインポート

import policy.policy as submitted  # 提出用ポリシーとしてモジュールをインポート
import problem.problems as problems  # 問題定義用モジュールをインポート

### Parameters  # パラメータ設定
TEST_EPI_NUM = 10  # テストエピソード数を10に設定
###############  # 区切り

def calculate_cost(instances, policy):  # 関数calculate_costを定義（インスタンスとポリシーを引数）
    """
    Once your policy has been developed,
    you can run this file without any edition.

    It outputs the following 6 criteria for each problem in a json file
    1. runtime: mean of the runtime per each episode
    2. distance: mean of total moving distance of every drones
    3. time_step: mean of the termination time step of each episode
    4. goal_rate: goal rate
    5. goal_count: number of steps which 'all' drones reach their goals
    6. subtotal_cost: the average of the cost at the same pattern

    However, we only take "final cost"(,which is the sum of subtotal_cost) as the final evaluation index.
    """  # 関数の説明（英語コメント）
    cost = []  # 各インスタンスのスコアを格納するリスト
    for instance in instances:  # 各問題インスタンスに対してループ
        map_name = instance["map"]  # マップ名を取得
        drone_num = instance["drone_num"]  # ドローン台数を取得
        start_arr = instance["start"]  # 開始位置の配列を取得
        goal_arr = instance["goal"]  # 目標位置の配列を取得

        # make environment  # 環境の作成
        env = gym.make(
            "drp_env:drp_safe-" + str(drone_num) + "agent_" + map_name + "-v2",  # 環境名を生成
            state_repre_flag="onehot_fov",  # 状態表現フラグを設定
            goal_array=goal_arr,  # 目標位置配列を設定
            start_ori_array=start_arr,  # 開始位置配列を設定
        )  # 環境を作成

        # recore runtime of each episode  # 各エピソードの実行時間を記録するためのリスト
        time_record = []  # 実行時間記録用リストの初期化
        goal_rate_list = []  # 各エピソードの目標到達率記録リストの初期化
        subtotal_cost_same_environment = []  # 同一環境内での部分コスト記録リストの初期化
        # run environment with submitted policy  # 提出済みポリシーで環境を実行
        for epi in range(TEST_EPI_NUM):  # 指定エピソード数分ループを実施
            start_time = time.time()  # エピソード開始時刻を記録
            n_obs = env.reset()  # 環境をリセットして初期状態を取得
            goal_checker = False  # 目標到達のチェックフラグを初期化
            goal_step = [None] * drone_num  # 各ドローンの目標到達ステップのリストを初期化
            while not goal_checker:  # 全ドローンが終了状態になるまでループ
                actions = policy(n_obs, env)  # ポリシー関数から行動を取得
                n_obs, reward, done, info = env.step(actions)  # 行動を実行し次の状態や報酬等を取得
                goal_checker = all(done)  # 全ドローンが終了状態かをチェック
                for i in range(drone_num):  # 全ドローンについてループ
                    if reward[i] == 100:  # 目標到達の場合
                        goal_step[i] = info["step"]  # 到達ステップを記録
                    elif reward[i] == -50:  # 衝突の場合
                        if goal_step[i] == None:  # まだ到達記録がなければ
                            goal_step[i] = 100  # 失敗として100を記録
            for i in range(drone_num):  # 各ドローンについて最終確認
                if goal_step[i] == None:  # 到達記録が無い場合
                    goal_step[i] = 100  # 100を記録
            pos = env.get_pos_list()  # ドローンの位置リストを取得
            goal_drone = 0  # 目標到達したドローン数のカウンタを初期化
            for i in range(len(pos)):  # 各位置情報についてループ
                if (
                    pos[i]["type"] == "n" and pos[i]["pos"] == goal_arr[i]
                ):  # ドローンがタイプ"n"で目標位置にいる場合
                    goal_drone += 1  # 目標到達カウンタをインクリメント
            goalrate = goal_drone / drone_num  # エピソード内の目標到達率を計算
            end_time = time.time()  # エピソード終了時刻を記録
            time_record.append(end_time - start_time)  # 実行時間を記録リストに追加
            goal_rate_list.append(goalrate)  # 目標到達率を記録リストに追加
            subtotal_cost_same_environment.append(sum(goal_step))  # コスト（合計ステップ）を記録リストに追加

        # calculate score from logs in environment  # 環境ログからスコアを計算
        score = {"instance_id": instance["id"]}  # インスタンスIDを含むスコア辞書を作成

        mean_runtime = 0.0  # 平均実行時間の初期化
        sum_runtime = 0.0  # 総実行時間の初期化

        mean_distance = 0.0  # 平均移動距離の初期化
        sum_distance = 0.0  # 総移動距離の初期化

        mean_timestep = 0.0  # 平均タイムステップの初期化
        sum_timestep = 0.0  # 総タイムステップの初期化

        goalrate = 0.0  # 平均目標到達率の初期化
        # goalcount = 0  # ゴールカウントの初期化（現状未使用）

        for epi in range(TEST_EPI_NUM):  # 各エピソードについてループ
            log = env.get_log(epi + 1)  # エピソードログを取得
            sum_runtime += time_record[epi]  # 実行時間の合計を更新
            sum_distance += sum(log["distance_from_start"])  # 走行距離の合計を更新
            sum_timestep += log["termination_time"]  # 終了タイムステップの合計を更新
            # if log["result"] == "goal": # もし結果が目標到達なら（コメントアウト済み）
            #     goalcount += 1  # ゴールカウントをインクリメント（コメントアウト済み）
        mean_runtime = sum_runtime / TEST_EPI_NUM  # 平均実行時間を計算
        mean_distance = sum_distance / TEST_EPI_NUM  # 平均移動距離を計算
        mean_timestep = sum_timestep / TEST_EPI_NUM  # 平均タイムステップを計算
        goalrate = sum(goal_rate_list) / TEST_EPI_NUM  # 平均目標到達率を計算
        mean_goal_step = sum(subtotal_cost_same_environment) / TEST_EPI_NUM  # 平均部分コストを計算
        score["runtime"] = mean_runtime  # スコアに平均実行時間を設定
        score["distance"] = mean_distance  # スコアに平均移動距離を設定
        score["time_step"] = mean_timestep  # スコアに平均タイムステップを設定
        score["goal_rate"] = goalrate  # スコアに平均目標到達率を設定
        score["goal_count"] = goal_rate_list.count(1.0)  # スコアに全ドローン完全到達エピソード数を設定
        score["subtotal_cost"] = mean_goal_step  # スコアに平均部分コストを設定
        cost.append(score)  # 全体スコアリストに今回のスコアを追加
        # delete env  # 環境オブジェクトの削除
        del env  # 環境オブジェクトを削除してメモリ解放

    subtotal_costs = [score["subtotal_cost"] for score in cost]  # 各スコアから部分コストのみ抽出
    final_cost = sum(subtotal_costs)  # 総最終コストを計算
    score_dict = {
        "Author": submitted.TEAM_NAME,  # スコア結果に著者（チーム名）を記録
        "Scored time": str(datetime.now().strftime("%Y-%m-%H-%M-%S")),  # スコア算出時刻を記録
        "Score": cost,  # 各エピソードのスコアリストを記録
        "final cost": final_cost,  # 総最終コストを記録
    }  # 結果をまとめた辞書を作成

    json_filename = submitted.TEAM_NAME + ".json"  # 出力するJSONファイル名を設定
    with open(json_filename, "w") as f:  # JSONファイルを書き込みモードでオープン
        json.dump(score_dict, f, indent=4)  # 結果をインデント付きJSON形式でファイルに書き込む
    return cost, final_cost  # 計算したスコアリストと最終コストを返却

if __name__ == "__main__":  # このファイルが実行された場合のエントリーポイント
    cost, final_score = calculate_cost(problems.instances, submitted.policy)  # コスト計算関数を実行
