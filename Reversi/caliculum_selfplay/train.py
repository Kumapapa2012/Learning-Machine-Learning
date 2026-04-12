#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
カリキュラム学習 + セルフプレイ版 学習スクリプト。

Phase 1: ヒューリスティック AI と対戦 → 勝率 60% 以上で Phase 2 へ移行
Phase 2: 過去モデルのプールとセルフプレイ

使い方:
    python train.py
    python train.py --timesteps 2000000 --threshold 0.6 --pool 5
    python train.py --check
"""
from __future__ import annotations

import argparse
import datetime

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from reversi_env import ReversiEnv
from selfplay_callback import CurriculumSelfPlayCallback

# ------------------------------------------------------------------
# デフォルトハイパーパラメータ
# ------------------------------------------------------------------
BOARD_SIZE      = 6
N_ENVS          = 4
TOTAL_TIMESTEPS = 2_000_000   # セルフプレイ分が加わるので多めに
MODEL_PATH      = "reversi_curriculum_ppo"
LOG_DIR         = "./reversi_tb/"
CHECKPOINT_DIR  = "./checkpoints/"
SNAPSHOT_DIR    = "./snapshots/"


def make_env(board_size: int = BOARD_SIZE, seed: int = 0):
    """DummyVecEnv 用ファクトリ。初期対戦相手は None（ヒューリスティック）。"""
    def _init():
        env = ReversiEnv(n_rows=board_size, n_cols=board_size, opponent=None)
        env = ActionMasker(env, lambda e: e.action_masks())
        return env
    return _init


def build_model(vec_env) -> MaskablePPO:
    return MaskablePPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        policy_kwargs=dict(net_arch=[256, 256]),
        learning_rate=3e-4,
        n_steps=512,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        device="cuda",
        tensorboard_log=LOG_DIR,
    )


def main(args: argparse.Namespace) -> None:
    print(f"[{datetime.datetime.now()}] Training start (curriculum + self-play)")
    print(f"  board={args.board}x{args.board}, envs={args.envs}, "
          f"timesteps={args.timesteps:,}")
    print(f"  Phase1 threshold={args.threshold:.0%}, pool_size={args.pool}")

    if args.check:
        print("Checking environment...")
        check_env(ReversiEnv(n_rows=args.board, n_cols=args.board), warn=True)
        print("Environment OK")

    # 学習環境
    vec_env = DummyVecEnv([make_env(args.board, seed=i) for i in range(args.envs)])
    vec_env = VecMonitor(vec_env)

    # モデル
    model = build_model(vec_env)

    # コールバック
    curriculum_cb = CurriculumSelfPlayCallback(
        phase1_threshold=args.threshold,
        eval_window=200,
        pool_size=args.pool,
        update_freq=10_000,
        snapshot_dir=SNAPSHOT_DIR,
        verbose=1,
    )
    checkpoint_cb = CheckpointCallback(
        save_freq=100_000,
        save_path=CHECKPOINT_DIR,
        name_prefix="reversi_curriculum",
    )

    # 学習
    model.learn(
        total_timesteps=args.timesteps,
        callback=[curriculum_cb, checkpoint_cb],
        progress_bar=True,
    )

    model.save(MODEL_PATH)
    print(f"[{datetime.datetime.now()}] Model saved → {MODEL_PATH}.zip")
    print(f"  最終フェーズ: {'セルフプレイ (Phase 2)' if curriculum_cb.current_phase == 2 else 'ヒューリスティック (Phase 1)'}")
    print(f"  最終勝率（直近 200 ep）: {curriculum_cb.current_win_rate():.1%}")

    vec_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Curriculum + Self-play Reversi training")
    parser.add_argument("--board",      "-b", type=int,   default=BOARD_SIZE,
                        help="Board size (default: 6)")
    parser.add_argument("--envs",       "-e", type=int,   default=N_ENVS,
                        help="Number of parallel environments (default: 4)")
    parser.add_argument("--timesteps",  "-t", type=int,   default=TOTAL_TIMESTEPS,
                        help="Total training timesteps (default: 2,000,000)")
    parser.add_argument("--threshold",  "-T", type=float, default=0.6,
                        help="Win rate threshold to switch to self-play (default: 0.6)")
    parser.add_argument("--pool",       "-p", type=int,   default=5,
                        help="Self-play model pool size (default: 5)")
    parser.add_argument("--check",      action="store_true",
                        help="Run environment check before training")
    args = parser.parse_args()
    main(args)
