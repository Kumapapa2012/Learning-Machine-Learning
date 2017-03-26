#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import sys
import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(0)
import rlglue.RLGlue as RLGlue


with open('result.txt', 'a') as f:
    f.writelines('START: ' + str(datetime.datetime.now()) + '\n')

which_episode = 0
total_win = 0
total_draw = 0
total_lose = 0

ns_epoch = []
pcts_win = []
pcts_win_or_draw = []
pcts_lose = []

# 強化学習の主処理
def runEpisode(step_limit):
    global which_episode
    global total_win
    global total_draw
    global total_lose
    global ns_epoch
    global pcts_win
    global pcts_win_or_draw
    global pcts_lose

    which_episode += 1

    # ゲーム1回 開始
    terminal = RLGlue.RL_episode(step_limit)

    # 勝負がつくまでのステップ数と報酬を取得
    total_steps = RLGlue.RL_num_steps()
    total_reward = RLGlue.RL_return()

    # 今回の結果を表示
    r_win = 1.0
    r_draw = -0.5
    r_lose = -1.0

    if total_reward == r_win:
        total_win += 1
    elif total_reward == r_draw:
        total_draw += 1
    elif total_reward == r_lose:
        total_lose += 1

    print("Episode "+str(which_episode)+"\t "+str(total_steps)+ " steps \t" + str(total_reward) + " total reward\t " + str(terminal) + " natural end")

    # 100回毎に勝敗を集計
    record_interval = 100

    if which_episode % record_interval == 0:
        line = 'Episode: {}, {} wins, {} draws, {} loses'.format(which_episode, total_win, total_draw, total_lose)
        print('---------------------------------------------------------------')
        print(line)
        print('---------------------------------------------------------------')

        # 集計結果をファイルに出力
        with open('result.txt', 'a') as f:
            f.writelines(line + '\n')

        ns_epoch.append(which_episode)
        pcts_win.append(float(total_win) / record_interval * 100)
        pcts_win_or_draw.append(float(total_win + total_draw) / record_interval * 100)
        pcts_lose.append(float(total_win) / record_interval * 100)

        total_win = 0
        total_draw = 0
        total_lose = 0

# RL_Glue初期化
print("\n\nExperiment starting up!")
task_spec = RLGlue.RL_init()
print("RL_init called, the environment sent task spec: " + task_spec)

# 50,000回実行
for _ in range(0, 5 * 10**4):
    runEpisode(0)

# 学習結果をグラフで出力
plt.plot(np.asarray(ns_epoch), np.asarray(pcts_win_or_draw))
plt.xlabel('episode')
plt.ylabel('percentage')
plt.title('Average win or draw rate')
plt.grid(True)
plt.savefig("percentages.png")

RLGlue.RL_cleanup()

# 終了時刻をファイルに書出し
with open('result.txt', 'a') as f:
    f.writelines('END: ' + str(datetime.datetime.now()) + '\n')
