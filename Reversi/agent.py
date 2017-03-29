#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import copy

import numpy as np
np.random.seed(0)
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda
from chainer import optimizers
from rlglue.agent.Agent import Agent
from rlglue.agent import AgentLoader as AgentLoader
from rlglue.types import Action
from rlglue.types import Observation
from rlglue.utils import TaskSpecVRLGLUE3

# QNet
# ニューラルネットワークのクラス
class QNet(chainer.Chain):
    
    # __init__( n_in, n_units, n_out)
    #       n_in: 入力層サイズ
    #    n_units: 中間層サイズ
    #      n_out: 出力層サイズ
    def __init__(self, n_in, n_units, n_out):
        super(QNet, self).__init__(
            l1=L.Linear(n_in, n_units),
            l20=L.Linear(n_units, n_units),
            l21=L.Linear(n_units, n_units),
            l22=L.Linear(n_units, n_units),
            l23=L.Linear(n_units, n_units),
            l24=L.Linear(n_units, n_units),
            l25=L.Linear(n_units, n_units),
            l26=L.Linear(n_units, n_units),
            l27=L.Linear(n_units, n_units),
            l3=L.Linear(n_units, n_out),
        )

    #value(x)
    #       x: 入力層の値
    #ニューラルネットワークによる計算
    #Return: 出力層の結果
    def value(self, x):
        h = F.relu(self.l1(x))
        h = F.relu(self.l20(h))
        h = F.relu(self.l21(h))
        h = F.relu(self.l22(h))
        h = F.relu(self.l23(h))
        h = F.relu(self.l24(h))
        h = F.relu(self.l25(h))
        h = F.relu(self.l26(h))
        h = F.relu(self.l27(h))
        return self.l3(h)

    #__call__(s_data, a_data, y_data)
    #    s_data: 状態
    #    a_data: アクション
    #    y_data: 教師データ(次の行動の最大Q値)
    #学習用コールバック。
    #1. s_data を Forward 伝播する(Q,Q_Data)
    #2. t_data に Q_Dataをコピー
    #3．t_data の a_data[i] の値を y_data[i]の Q 値で置き換え教師データ作成(t)
    #4. Q,t の二乗誤差を算出
    #
    #Return: 二乗誤差計算結果
    def __call__(self, s_data, a_data, y_data):
        self.loss = None
        
        s = chainer.Variable(self.xp.asarray(s_data))
        Q = self.value(s)
        
        Q_data = copy.deepcopy(Q.data)
        
        if type(Q_data).__module__ != np.__name__:
            Q_data = self.xp.asnumpy(Q_data)
        
        t_data = copy.deepcopy(Q_data)
        for i in range(len(y_data)):
            t_data[i, a_data[i]] = y_data[i]
        
        t = chainer.Variable(self.xp.asarray(t_data))
        self.loss = F.mean_squared_error(Q, t)
        
        print('Loss:', self.loss.data)
        
        return self.loss
        
# エージェントクラス
class KmoriReversiAgent(Agent):
    
    #__init__(gpu, size)
    #    gpu: GPU 番号(0以上、CPU 使用の場合 -1)
    #    size: 正方形ボードの 1 辺の長さ(6 以上の偶数)
    # エージェントの初期化、学習の内容を定義する
    def __init__(self, gpu, size):
        # サイズは 6 以上の偶数で。
        if size<6 and size%2 != 0 : print("size must be even number and 6 or above!") ; exit()
        # 盤の情報(オセロは8)
        self.n_rows = int(size)
        self.n_cols = self.n_rows
        
        # 学習のInputサイズ
        self.dim = self.n_rows * self.n_cols # ボードサイズ=出力層のサイズ
        self.bdim = self.dim * 4  # 学習用データのサイズ
        
        self.gpu = gpu
        
        # 学習を開始させるステップ数
        self.learn_start = 5 * 10**3
        
        # 保持するデータ数(changed)
        self.capacity = 2 * 10**4
        
        # eps = ランダムに○を決定する確率
        self.eps_start = 1.0
        self.eps_end = 0.001
        self.eps = self.eps_start
        
        # 学習時にさかのぼるAction数
        self.n_frames = 9
        
        # 一度の学習で使用するデータサイズ
        self.batch_size = 128
        
        self.replay_mem = []
        self.last_state = None
        self.last_action = None
        self.reward = None
        self.state = np.zeros((1, self.n_frames, self.bdim)).astype(np.float32)
        
        self.step_counter = 0
        
        self.update_freq = 1 * 10**4
        
        self.r_win = 1.0
        self.r_draw = -0.5
        self.r_lose = -1.0
        
        self.frozen = False
        
        self.win_or_draw = 0
        self.stop_learning = 200
    
    #agent_init(task_spec_str)
    #    task_spec_str: RL_Glue から渡されるタスク情報
    # ゲーム情報の初期化
    def agent_init(self, task_spec_str):
        task_spec = TaskSpecVRLGLUE3.TaskSpecParser(task_spec_str)
        
        if not task_spec.valid:
            raise ValueError(
                'Task spec could not be parsed: {}'.format(task_spec_str))
        
        self.gamma = task_spec.getDiscountFactor() #　割引率
        #　DQN　作成
        #　Arg1: 入力層サイズ
        #　Arg2:　隠れ層ノード数
        #　Arg3：　出力層サイズ
        self.Q = QNet(self.bdim*self.n_frames, self.bdim*self.n_frames, self.dim)
        
        if self.gpu >= 0:
            cuda.get_device(self.gpu).use()
            self.Q.to_gpu()
        self.xp = np if self.gpu < 0 else cuda.cupy
        
        self.targetQ = copy.deepcopy(self.Q)
        
        self.optimizer = optimizers.RMSpropGraves(lr=0.00025, alpha=0.95,
                                                  momentum=0.0)
        self.optimizer.setup(self.Q)
    
    #agent_start(observation)
    #    observation: ゲーム状態(ボード状態など)
    #environment.py の env_startの次に呼び出される。
    #1手目 Action を決定し、実行する。
    #実行した Action をエージェントへの情報として RL_Glue に渡す。
    def agent_start(self, observation):
        # stepを1増やす
        self.step_counter += 1
        
        #　kmori： 独自のobservationを使用して、状態をアップデート。
        # 一部サンプルに合わせ、残りは別の方法で作成した。
        self.update_state(observation)
        self.update_targetQ()
        
        # 自分が打つ手を決定する。
        int_action = self.select_int_action()
        action = Action()
        action.intArray = [int_action]
        
        # eps を更新する。epsはランダムに○を打つ確率
        self.update_eps()
        
        # state = 盤の状態 と action = ○を打つ場所 を退避する
        self.last_state = copy.deepcopy(self.state)
        self.last_action = copy.deepcopy(int_action)
        
        return action
    
    #agent_step(reward, observation)
    #    reward: 報酬
    #    observation: ゲーム状態(ボード状態など)
    #エージェントの二手目以降、ゲームが終わるまで呼ばれる。
    #(Reversi の場合、報酬は常にゼロとなる)
    def agent_step(self, reward, observation):
        # ステップを1増加
        self.step_counter += 1
        
        self.update_state(observation)
        self.update_targetQ()
        
        # 自分が打つ手を決定する。
        int_action = self.select_int_action() # 戻り値が -1 ならパス。
        action = Action()
        action.intArray = [int_action]
        self.reward = reward
        
        # epsを更新
        self.update_eps()
        
        # データを保存 (状態、アクション、報酬、結果)
        self.store_transition(terminal=False)
        
        if not self.frozen:
            # 学習実行
            if self.step_counter > self.learn_start:
                self.replay_experience()
        
        self.last_state = copy.deepcopy(self.state)
        self.last_action = copy.deepcopy(int_action)
        
        # ○の位置をエージェントへ渡す
        return action
    
    #agent_end(reward)
    #    reward: 報酬
    # ゲームが終了した時点で呼ばれる
    def agent_end(self, reward):
        # 環境から受け取った報酬
        self.reward = reward
        
        if not self.frozen:
            if self.reward >= self.r_draw:
                self.win_or_draw += 1
            else:
                self.win_or_draw = 0
            
            if self.win_or_draw == self.stop_learning:
                self.frozen = True
                f = open('result.txt', 'a')
                f.writelines('Agent frozen\n')
                f.close()
        
        # データを保存 (状態、アクション、報酬、結果)
        self.store_transition(terminal=True)
        
        if not self.frozen:
            # 学習実行
            if self.step_counter > self.learn_start:
                self.replay_experience()
    
    def agent_cleanup(self):
        # (今後実装)
        # RL_Cleanup により呼ばれるはず。
        # ここでモデルをセーブすればきっといい。
        pass
    
    def agent_message(self, message):
        pass
    
    #update_state(observation=None)
    #    observation: ゲーム状態(ボード状態など)
    #ゲーム状態を state に格納する。
    def update_state(self, observation=None):
        # 学習用の状態保存。
        if observation is None:
            frame = np.zeros(1, 1, self.bdim).astype(np.float32)
        else:
            # observation の内容から、学習用データを作成。
            observation_binArray=[]
            pageSize=self.n_rows*self.n_cols
            # コマの位置
            for i in range(0,pageSize):
                observation_binArray.append(int(observation.intArray[i]))
                observation_binArray.append(int(observation.intArray[pageSize+i]))
            # コマを置ける場所
            for i in range(0,pageSize):
                observation_binArray.append(int(observation.intArray[2*pageSize+i]))
                observation_binArray.append(int(observation.intArray[3*pageSize+i]))
            
            
            frame = (np.asarray(observation_binArray).astype(np.float32)
                                                     .reshape(1, 1, -1))
        self.state = np.hstack((self.state[:, 1:], frame))
    
    #update_eps()
    #ゲームの手数合計に基づき、ε-Greedy 法の ε を更新。
    def update_eps(self):
        if self.step_counter > self.learn_start:
            if len(self.replay_mem) < self.capacity:
                self.eps -= ((self.eps_start - self.eps_end) /
                             (self.capacity - self.learn_start + 1))
    
    #update_targetQ()
    #update_freq 毎に、現時点の Q 値を、targetQ(Q 値推測用 Network) にコピー。
    def update_targetQ(self):
        if self.step_counter % self.update_freq == 0:
            self.targetQ = copy.deepcopy(self.Q)
    
    #select_int_action()
    #現在のボード状態から、DQN を用いて打つ手を決める。
    #コマを置く場所を決める。
    def select_int_action(self):
        bits = self.state[0, -1]  #　ここでは stateの最後の要素つまり現時点の情報を得ている。
        
        # ここでは、空きマスを取得している。
        # このオセロでは、コマを置ける場所は Observation に含まれるのでそれを使用する。
        free=[]
        freeInBoard=bits[(2*self.n_rows*self.n_cols):]
        for i in range(0, len(freeInBoard), 2) :
            if int(freeInBoard[i]) == 1 :
                free.append(i//2)
                
        # 置ける場所がなければここでパス
        if len(free)==0:
            # pass...
            return -1
        
        #　Q値を求める
        s = chainer.Variable(self.xp.asarray(self.state))
        Q = self.Q.value(s)
        
        # Follow the epsilon greedy strategy
        if np.random.rand() < self.eps:
            int_action = free[np.random.randint(len(free))]
        else:
            #　先頭のQ値
            Qdata = Q.data[0]
            if type(Qdata).__module__ != np.__name__:
                Qdata = self.xp.asnumpy(Qdata)
            
            #　アクションを決定します。
            #     石を置けるマスの中から、Q値の最も高いものを行動として選択しています。
            for i in np.argsort(-Qdata):
                if i in free:
                    int_action = i
                    break
                    
        
        return int_action
    
    def store_transition(self, terminal=False):
        if len(self.replay_mem) < self.capacity:
            self.replay_mem.append(
                (self.last_state, self.last_action, self.reward,
                 self.state, terminal))
        else:
            #　self.replay_mem[1:]　で先頭つまり最古の要素を除く配列に、新しいものを追加。
            # これにより FIFO　でリストが回転する。
            self.replay_mem = (self.replay_mem[1:] +
                [(self.last_state, self.last_action, self.reward, 
                  self.state, terminal)])
    
    def replay_experience(self):
        #　replay_memory　から　バッチサイズ分の要素をランダムに取得する。
        indices = np.random.randint(0, len(self.replay_mem), self.batch_size)
        samples = np.asarray(self.replay_mem)[indices]
        
        s, a, r, s2, t = [], [], [], [], []
        
        for sample in samples:
            s.append(sample[0])
            a.append(sample[1])
            r.append(sample[2])
            s2.append(sample[3])
            t.append(sample[4])
        
        
        s = np.asarray(s).astype(np.float32)
        a = np.asarray(a).astype(np.int32)
        r = np.asarray(r).astype(np.float32)
        s2 = np.asarray(s2).astype(np.float32)
        t = np.asarray(t).astype(np.float32)
        
        #Q 値推測用ネットワーク targetQ を取得し、s2の Q 値を求める
        s2 = chainer.Variable(self.xp.asarray(s2))
        Q = self.targetQ.value(s2)
        Q_data = Q.data
        
        if type(Q_data).__module__ == np.__name__:
            max_Q_data = np.max(Q_data, axis=1)
        else:
            max_Q_data = np.max(self.xp.asnumpy(Q_data).astype(np.float32), axis=1)
        
        #targetQで推測した Q 値を使用して 教師データ t 作成
        t = np.sign(r) + (1 - t)*self.gamma*max_Q_data
        
        self.optimizer.update(self.Q, s, a, t)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deep Q-Learning')
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--size', '-s', default=6, type=int,
                        help='Reversi board size')
    args = parser.parse_args()
    
    AgentLoader.loadAgent(KmoriReversiAgent(args.gpu,args.size))
