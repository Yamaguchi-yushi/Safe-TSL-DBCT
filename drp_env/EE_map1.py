import csv  # CSVファイル操作用
import networkx as nx  # グラフ構造用
import numpy as np  # 数値計算用
import matplotlib.pyplot as plt  # グラフ描画用
import copy  # オブジェクトのコピー用
import math  # 数学関数用
import sys  # システム関連
import os  # OS操作用
sys.path.append(os.path.join(os.path.dirname(__file__), ''))  # カレントディレクトリをパスに追加

MAP_PARENT_DIR = "./map/"  # マップファイルの親ディレクトリ

class MapMake():  # マップ生成クラス

	def __init__(self, agent_num, start_ori_array, goal_array, map_name):  # 初期化
		print('MapMake initialized')  # 初期化メッセージ
		self.agent_num = agent_num  # エージェント数

		base_nodes = [0,2]  # 基準ノード
		self.base_nodes = base_nodes  # 基準ノード保存
		self.fig1 = plt.figure()  # 図の作成
		self.ax3 = self.fig1.add_subplot(111)  # サブプロット作成
		map_dir = MAP_PARENT_DIR+map_name  # マップディレクトリ
		node_file_name, edge_file_name = map_dir+'/node', map_dir+'/edge'  # ノード・エッジファイル名
		csv_nodes_number,csv_nodes_pos,csv_edges, csv_edges_weights = self.read_nodes_csv(node_file_name, edge_file_name)  # CSV読み込み
		self.G, self.pos, self.edge_labels = self.Graph_initial(csv_nodes_number, csv_nodes_pos, csv_edges, csv_edges_weights)  # グラフ初期化
		# self.n_nodes = len(self.G.nodes())
		if self.agent_num > len(csv_nodes_number) :  # エージェント数がノード数を超える場合
			print('Error: The number of agents exceeds the maximum numberof nodes on the graph')  # エラー出力
			self.sta_code = 0  # ステータスコード
			self.close()  # 終了処理
			sys.exit(0)  # プログラム終了
		else:
			print('Mal Environment initialized')  # 正常メッセージ

		print('Agent numbers', self.agent_num)  # エージェント数表示

		# ユーザー指定
		self.input_start_ori_array = copy.deepcopy(start_ori_array)  # スタートノード配列
		self.input_goal_array = copy.deepcopy(goal_array)  # ゴールノード配列
		
		self.start_ori_array = copy.deepcopy(self.input_start_ori_array)  # スタートノード配列
		self.goal_array = copy.deepcopy(self.input_goal_array)  # ゴールノード配列
		
		if self.input_start_ori_array==[]:  # スタート未指定時
			self.random_start()  # ランダムスタート
		if self.input_goal_array==[]:  # ゴール未指定時
			self.random_goal()  # ランダムゴール
		
		self.base_nodes = base_nodes  # 基準ノード再設定

		print('Start node for each agent', self.start_ori_array)  # スタートノード表示
		print('Goal node for each agent', self.goal_array)  # ゴールノード表示

		#self.start1_ori,self.goal1=0,5
		#self.start2_ori,self.goal2=3,4

	def random_start(self):  # ランダムスタートノード割当
		self.G_nodes_copy = copy.deepcopy(list(self.G.nodes()))  # ノードリストコピー
		self.start_ori_array = []  # スタートノード配列初期化
		# G_nodes_copy=copy.deepcopy(list(self.G.nodes()))
		for i in range(self.agent_num):  # 各エージェントに対して
			random_node = np.random.choice(self.G_nodes_copy)  # ランダムノード選択
			self.start_ori_array.append(random_node)  # 配列に追加
			self.G_nodes_copy.remove(random_node)  # 選択ノードをリストから削除
  
	def random_goal(self):  # ランダムゴールノード割当
		self.goal_array = []  # ゴールノード配列初期化
		# G_nodes_copy=copy.deepcopy(list(self.G.nodes()))
		for i in range(self.agent_num):  # 各エージェントに対して
			random_node = np.random.choice(self.G_nodes_copy)  # ランダムノード選択
			self.goal_array.append(random_node)  # 配列に追加
			self.G_nodes_copy.remove(random_node)  # 選択ノードをリストから削除

	def read_nodes_csv(self, node, edge):  # ノード・エッジCSV読み込み
		csv_nodes_source = []  # ノードCSVデータ
		csv_edges_source = []  # エッジCSVデータ
		current_path = os.path.join(os.path.dirname(__file__), '')  # カレントパス
		with open(current_path+node+'.csv') as f:  # ノードCSVを開く
			reader = csv.reader(f)  # リーダー作成
			for row in reader:  # 各行
				csv_nodes_source.append(row)  # 配列に追加

		with open(current_path+edge+'.csv') as f:  # エッジCSVを開く
			reader = csv.reader(f)  # リーダー作成
			for row in reader:  # 各行
				csv_edges_source.append(row)  # 配列に追加

		csv_nodes_source.remove(csv_nodes_source[0])#タイトル行削除
		csv_edges_source.remove(csv_edges_source[0])#タイトル行削除
		#ノード情報
		csv_nodes_number = [int(i[0]) for i in csv_nodes_source]  # ノード番号
		csv_nodes_pos = dict()  # ノード座標辞書
		for node in csv_nodes_source:  # 各ノード
			#csv_nodes_pos[int(node[0])]=[ float(node[1]),float(node[2]) ]
			csv_nodes_pos[int(node[0])] = [round(float(node[1]),2), round(float(node[2]),2)]  # 座標を格納
		
		#エッジ情報
		csv_edges = []  # エッジリスト
		csv_edges_weights = []  # エッジ重みリスト
		for i in range(len(csv_edges_source)):  # 各エッジ
			row = csv_edges_source[i]  # 行データ
			source = int(row[0])  # 始点
			distention = int(row[1])  # 終点
			#error
			distance = np.sqrt(((csv_nodes_pos[source][0]-csv_nodes_pos[distention][0])**2)+((csv_nodes_pos[source][1]-csv_nodes_pos[distention][1])**2))  # 距離計算
			csv_edges.append((source, distention))  # エッジ追加
			csv_edges_weights.append((source, distention, distance))  # エッジ重み追加

		#print(csv_nodes_number)  #[0, 1, 2, 3, 4, 5, 6]
		#print(csv_nodes_pos)     #{0: (0.0, 5.0), 1: (2.0, 10.0), 2: (2.0, 0.0),....
		#print(csv_edges)         #[(0, 1), (0, 2), (1, 2), (1, 3), (2, 4),....
		#print(csv_edges_weights) #[(0, 1, 5.4), (0, 2, 5.4), (1, 2, 10.0),....
		return csv_nodes_number, csv_nodes_pos, csv_edges, csv_edges_weights  # 結果を返す

	def Graph_initial(self, csv_nodes_number ,csv_nodes_pos ,csv_edges ,csv_edges_weights):  # グラフ初期化
		G = nx.Graph()  # 無向グラフ作成
		G_g = nx.Graph()  # 予備グラフ               
		G.add_nodes_from(csv_nodes_number)  # ノード追加                                 
		G.add_edges_from(csv_edges)  # エッジ追加
		self.pos = csv_nodes_pos  # ノード座標
		G.add_weighted_edges_from(csv_edges_weights) #(始点，終点，重み)でエッジを設定
		self.edge_labels = {(i, j): int(w['weight']) for i, j, w in G.edges(data=True)} #エッジラベル辞書
		self.G=G  # グラフ保存

		return self.G, self.pos, self.edge_labels  # 結果を返す

	def draw_weighted_graph(self, G ,pos):  # グラフ描画
		nx.draw_networkx_nodes(G, pos, node_size=500, node_color='skyblue',edgecolors='skyblue') #ノード描画
		nx.draw_networkx_edges(G, pos, width=1) #エッジ描画
		nx.draw_networkx_labels(G, pos) #ノードラベル描画
		nx.draw_networkx_edge_labels(G, pos, edge_labels=self.edge_labels) #エッジラベル描画
		#nx.draw_networkx_node_labels(G, pos, node_labels=node_labels) #ノードラベル描画
		nx.draw_networkx(self.G, with_labels = True,pos=self.pos,alpha=0.2, node_size=170, node_color='lightblue')  # 追加描画

	def plot_map_dynamic(self, delay ,obs_old, obs, goal_array, agent_num, joint_action_old, reach_account, step, episode):# 動的マップ描画
		self.agent_num = agent_num  # エージェント数設定
		
		#for i in range(self.goods):
			#ax3.scatter(  self.pos[self.base_nodes[0]][0]-0.5, self.pos[self.base_nodes[0]][1]-i*1.3 , alpha=1, s=500, marker='*',c='grey')
		for base_node in self.base_nodes:  # 基準ノード描画
			self.ax3.scatter(self.pos[base_node][0], self.pos[base_node][1], alpha=1, s=1500, marker='1', c='steelblue')
		
		c = [(i+1)/self.agent_num  for i in range(self.agent_num)]  # 色リスト
		#print("c",c)

		for i in range(self.agent_num):  # 各エージェント
			sc = self.ax3.scatter(obs[i][0], obs[i][1] ,alpha=1, s=600 ,c=c[i], cmap='rainbow', vmin=0, vmax=1)  # エージェント描画
			self.ax3.annotate(str(joint_action_old[i]), (obs[i][0], obs[i][1] ), size = 12, color = "white")  # 行動ラベル

			if self.pos[goal_array[i]]!=7777:  # ゴール描画
				self.ax3.scatter(self.pos[obs[i][3]][0], self.pos[obs[i][3]][1], alpha=1, s=700, c=c[i], cmap='rainbow', vmin=0, vmax=1)
			
			#if s[i][0] == self.pos[ s_old[i][2] ] and self.agent_carry_goods_array[i]==1:
			
			#ax3.scatter(  s[i][0][0], s[i][0][1] , alpha=1, s=500, marker='*',c='grey')
			#self.agent_carry_goods_array[i]=0
		
			#print( s_old[i], s[i])
			if [obs_old[i][0], obs_old[i][1]]==[ obs[i][0], obs[i][1]]: #位置変化なし
				if [obs[i][0], obs[i][1]]==self.pos[obs[i][3]] : #ゴール
					self.ax3.annotate('reach1', (obs[i][0]-0.2, obs[i][1]+0.2))     
				else:
					self.ax3.annotate('wait1', (obs[i][0]-0.2, obs[i][1]+0.2))
				
				
			# 矢印描画（コメントアウト）
			"""
			if  [obs_old[i][0], obs_old[i][1]]!=self.pos[obs[i][3]]:

			arrow_length=1
			self.ax3.annotate('', xy=[obs_old[i][0]+arrow_length*np.cos(angle[i]/57 ),obs_old[i][1]+arrow_length*np.sin(angle[i]/57 )] , xytext=[obs_old[i][0], obs_old[i][1]],
						arrowprops=dict(shrink=0, width=3, headwidth=8, 
										headlength=10, connectionstyle='arc3',
										facecolor='gray', edgecolor='gray')
						)
			
			"""

		#plt.xlim(-40,160) #x軸範囲指定
		#plt.ylim(-10,185) #y軸範囲指定

		#plt.gcf().text(0.02, 0.5, "reach_n:"+str(reach_account), fontsize=10)
		self.ax3.text(-5, 0, "reach_n:"+str(reach_account), fontsize=10)  # ゴール到達数表示
		self.ax3.text(-5, 3, "step_n:"+str(step), fontsize=10)  # ステップ数表示
		self.ax3.text(-5, 6, "episode_n:"+str(episode), fontsize=10)  # エピソード数表示

		self.draw_weighted_graph(self.G, self.pos)  # グラフ描画
		plt.grid() #グリッド
		#xtick=np.arange(-1,12, 1)
		#plt.xticks(xtick)
		plt.pause(delay)  #描画遅延
		plt.cla() #軸クリア

	def get_avail_action_fun(self, obs_i, current_start, current_goal, goal_i):  # 利用可能な行動取得
		#if s==self.pos[goal_i] and goal_i==0:
		if [obs_i[0],obs_i[1]]==self.pos[goal_i]:  # ゴール到達時
			#return ['null']
			return [goal_i]  # ゴールのみ

		action_set = []  # 行動セット
		#print(s,pos.values())
		#print("[obs_i[0],obs_i[1]] pos.values()",[obs_i[0],obs_i[1]],self.pos.values())
		if str([obs_i[0],obs_i[1]]) in [str(ele) for ele in self.pos.values()]: #ノード上
			#print("it currently at node")
			node = [k for k, v in self.pos.items() if str(v) == str([obs_i[0],obs_i[1]])][0]  # 現在ノード
			#print("current node",node)
			for edge in self.G.edges():  # エッジ探索
				if node in edge:
					if list(edge)[0] not in action_set and list(edge)[0]!=node:
						action_set.append(list(edge)[0])

					if list(edge)[1] not in action_set and list(edge)[1]!=node :
						#action_set.append(list(edge)[1])
						action_set.append(list(edge)[1])
			action_set.append(node)  # 現在ノードも追加

		else:
			#print("it currently NOT at node")
			# action_set=[current_start ,current_goal]
			action_set = [current_goal]  # エッジ上はcurrent_goalのみ

		return action_set  # 行動セット返却

	def collision_detect(self, obs_prepare):  # 衝突判定
		collision_flag = 0  # 衝突フラグ
		for i in range(self.agent_num-1):  # 全エージェントペア
			pos_i = [obs_prepare[i][0], obs_prepare[i][1]]  # エージェントi位置
			#print("pos_i",i,pos_i)
			for j in range(i+1, self.agent_num):  # エージェントj
				pos_j = [obs_prepare[j][0], obs_prepare[j][1]]  # エージェントj位置
				#print("pos_j",j,pos_j)
				distance_ij = math.dist(pos_i, pos_j)  # 距離計算
				#print( "distance i j",distance_ij)

				if distance_ij<5:  # 距離が5未満なら衝突
					collision_flag = 1  # 衝突フラグ
					print('!!!collision!!! with agent',i,j)  # 衝突メッセージ
		
		return collision_flag  # 衝突フラグ返却

"""
if __name__ == '__main__':
    Map=MapMake()
    csv_nodes_number,csv_nodes_pos,csv_edges,csv_edges_weights=Map.read_nodes_csv('node','edge')
    G, pos, edge_labels = Map.Graph_initial(csv_nodes_number,csv_nodes_pos,csv_edges,csv_edges_weights)
    
    Map.plot_map(pos) # a must be a angle !!!list!!!
"""

