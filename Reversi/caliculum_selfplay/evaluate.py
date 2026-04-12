#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
評価スクリプト（カリキュラム学習・セルフプレイ版）。

使い方:
    python evaluate.py                              # ヒューリスティック相手に評価
    python evaluate.py --opponent reversi_ppo       # 別モデル相手に評価
    python evaluate.py --episodes 500 --render      # 盤面表示しながら評価
"""
from __future__ import annotations

import argparse

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from reversi_env import ReversiEnv

MODEL_PATH = "reversi_curriculum_ppo"
N_EPISODES = 1000
BOARD_SIZE  = 6


def evaluate(
    model_path: str  = MODEL_PATH,
    n_episodes: int  = N_EPISODES,
    board_size: int  = BOARD_SIZE,
    opponent_path: str | None = None,
    deterministic: bool = True,
    render: bool = False,
) -> dict[str, float]:
    """
    Args:
        opponent_path: None のときヒューリスティック、
                       モデルパスを指定するとそのモデルと対戦。
    """
    # 対戦相手の準備
    opponent = None
    if opponent_path:
        opponent = MaskablePPO.load(opponent_path)
        print(f"対戦相手: {opponent_path}")
    else:
        print("対戦相手: ヒューリスティック AI")

    render_mode = "human" if render else None
    env = ReversiEnv(
        n_rows=board_size, n_cols=board_size,
        render_mode=render_mode, opponent=opponent,
    )
    env = ActionMasker(env, lambda e: e.action_masks())

    model = MaskablePPO.load(model_path, env=env)

    wins = draws = losses = 0
    total_steps = 0

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_steps = 0

        while not done:
            masks = env.action_masks()
            action, _ = model.predict(obs, action_masks=masks, deterministic=deterministic)
            obs, reward, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated
            ep_steps += 1

        total_steps += ep_steps

        if reward >= 1.0:
            wins += 1
        elif reward == -0.5:
            draws += 1
        else:
            losses += 1

        if (ep + 1) % 100 == 0:
            n = ep + 1
            print(f"  [{n:4d}/{n_episodes}] "
                  f"win={wins/n*100:.1f}%  "
                  f"draw={draws/n*100:.1f}%  "
                  f"lose={losses/n*100:.1f}%")

    env.close()

    return {
        "win_rate":        wins   / n_episodes,
        "draw_rate":       draws  / n_episodes,
        "lose_rate":       losses / n_episodes,
        "avg_episode_len": total_steps / n_episodes,
    }


def print_stats(stats: dict, n_episodes: int) -> None:
    print()
    print(f"=== Results over {n_episodes} episodes ===")
    print(f"  Wins:        {stats['win_rate']*100:5.1f}%")
    print(f"  Draws:       {stats['draw_rate']*100:5.1f}%")
    print(f"  Losses:      {stats['lose_rate']*100:5.1f}%")
    print(f"  Win or draw: {(stats['win_rate']+stats['draw_rate'])*100:.1f}%")
    print(f"  Avg length:  {stats['avg_episode_len']:.1f} steps")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Reversi agent")
    parser.add_argument("--model",      "-m", type=str,            default=MODEL_PATH)
    parser.add_argument("--opponent",   "-o", type=str,            default=None,
                        help="Opponent model path (default: heuristic AI)")
    parser.add_argument("--episodes",   "-n", type=int,            default=N_EPISODES)
    parser.add_argument("--board",      "-b", type=int,            default=BOARD_SIZE)
    parser.add_argument("--render",     "-r", action="store_true")
    parser.add_argument("--stochastic",        action="store_true")
    args = parser.parse_args()

    stats = evaluate(
        model_path    = args.model,
        n_episodes    = args.episodes,
        board_size    = args.board,
        opponent_path = args.opponent,
        deterministic = not args.stochastic,
        render        = args.render,
    )
    print_stats(stats, args.episodes)
