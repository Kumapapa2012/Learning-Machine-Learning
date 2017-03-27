#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse

import numpy as np
np.random.seed(0)
from rlglue.environment.Environment import Environment
from rlglue.environment import EnvironmentLoader as EnvironmentLoader
from rlglue.types import Observation
from rlglue.types import Reward_observation_terminal

import Game_Reversi as game_base

class KmoriReversiEnvironment(Environment):
    game=None
    map=[]
    
    # 敵プレイヤーが正常に打つ確率
    opp = 0.75
    
    #__init__(size)
    #    size: ボードサイズ
    #今回は、正方形のボードのみ対応することとする。
    #指定されたボードサイズでクラスを作成。
    def __init__(self, size):
        self.game= game_base.Game_Reversi(size,size)     
        
        self.n_rows = self.game.n_rows
        self.n_cols = self.game.n_cols
        self.history = []
    
    #build_map_from_game()
    #    size: ボードサイズ
    #現在のボードについて、以下を1次元のリストで表現する。
    #   0： 現在の盤面(Agent=1　のコマの位置)
    #   1： 現在の盤面(Environment=-1　のコマの位置)
    #   2： Agentがコマを置ける場所   
    #   3： Environmentがコマを置ける場所
    #Return: Map Data
    def build_map_from_game(self):
        
        map_data=[]
        
        #   0： 現在の盤面(Agent=1　のコマの位置)
        board_data=(self.game.g_board.reshape(-1)== 1).astype(int)
        map_data.extend(board_data)
        
        #   1： 現在の盤面(Environment=-1　のコマの位置)
        board_data=(self.game.g_board.reshape(-1)==-1).astype(int)
        map_data.extend(board_data)
        
        #   2： Agent が置ける場所。"turn" の正負でAgent と Environment を表現している。
        #      正なら Agent となる。
        pos_available=np.zeros_like(self.game.g_board)
        l_available=self.game.getPositionAvail(self.game.turn)
        for avail in l_available:
            pos_available[tuple(avail)]=1
        map_data.extend(pos_available.reshape(-1).tolist())
        
        #   3： Environmentの置ける場所
        pos_available=np.zeros_like(self.game.g_board)
        l_available=self.game.getPositionAvail(-self.game.turn)
        for avail in l_available:
            pos_available[tuple(avail)]=1
        map_data.extend(pos_available.reshape(-1).tolist())
        
        return map_data
    
    #env_init()
    #RL_Glueの設定を行う。
    def env_init(self):
        # OBSERVATONS INTS = 盤の状態 (-1 または 1 の値が self.n_rows*self.n_cols*4次元(詳細は、build_map_from_game()))
        # ACTIONS INTS = ○を打つ場所を指定 (-1 ~ (self.n_rows*self.n_cols-1))
        # REWARDS = 報酬 (-1.0 ~ 1.0)   ex) 勝 1, 引分 -0.5, 負 -1
        return 'VERSION RL-Glue-3.0 PROBLEMTYPE episodic DISCOUNTFACTOR 0.99 OBSERVATIONS INTS ('+str(self.n_rows*self.n_cols*4)+' 0 1) ACTIONS INTS (-1 '+str(self.n_rows*self.n_cols-1)+') REWARDS (-1.0 1.0)'
    
    #env_start()
    #Episodeの開始
    def env_start(self):
        #　plan:Reversi ボード初期化
        self.game.resetBoard()
        
        # map データの作成
        self.map=self.build_map_from_game()
        
        # 盤の状態を保持し、最後に確認するためのリスト
        # kmori： self.history に現在の盤面をテキストで追記します。
        self.history = []
        current_map = ''
        for i in range(0, len(self.map), self.n_cols):
            current_map += ' '.join(map(str, self.map[i:i+self.n_cols])) + '\n'
        self.history.append(current_map)
        
        # 盤の状態をRL_Glueを通してエージェントに渡す
        observation = Observation()
        observation.intArray = self.map
        
        return observation
        
    #env_step(action)
    #    action: エージェントの手(コマを置く場所)
    #エージェントから受け取ったコマを置く場所を、Game_Reversi.py に渡す。
    #その結果を RL_glue に渡す。
    #Return: Reward_observation_terminal
    def env_step(self, action):
        # エージェントから受け取った○を打つ場所
        int_action_agent = action.intArray[0]
        
        # 相手(Agent)　の手を実行し、勝敗を確認する
        # 勝敗がつかなければ、自身の手を考え、実行する。
        # ゲームの報酬などをまとめて エージェントにおくる。
        #　パスの場合は、(-1,-1)を使用する
        if int_action_agent==-1 :
            step_raw_col=(-1,-1)
        else :
            step_raw_col=(int_action_agent//self.n_cols,int_action_agent%self.n_cols)
        
        # step　実行
        step_o, step_r, step_done = self.game.step(step_raw_col)
        
        rot = Reward_observation_terminal()
        
        #　build_map_from_game()でマップを作成する。
        self.map=self.build_map_from_game()
        observation = Observation()
        observation.intArray = self.map
        rot.o = observation
        
        #　step_r　は報酬、step_done　は継続の有無
        rot.r=step_r
        rot.terminal = step_done
        
        # ボード情報保存用
        current_map = ''
        for i in range(0, len(self.map), self.n_cols):
            current_map += ' '.join(map(str, self.map[i:i+self.n_cols])) + '\n'
        self.history.append(current_map)
        
        # 試合の様子を記録
        if rot.r == self.game.r_lose:
            f = open('history.txt', 'a')
            history = '\n'.join(self.history)
            f.writelines('# START\n' + history + '# END\n\n')
            f.close()
        
        # 決着がついた場合は agentのagent_end
        # 決着がついていない場合は agentのagent_step に続く
        return rot
        
    def env_cleanup(self):
        pass
    
    def env_message(self, message):
        pass
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deep Q-Learning')
    parser.add_argument('--size', '-s', default=6, type=int,
                        help='Reversi board size')
    args = parser.parse_args()
    EnvironmentLoader.loadEnvironment(KmoriReversiEnvironment(args.size))
