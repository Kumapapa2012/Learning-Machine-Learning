# -*- coding: utf-8 -*-
"""
カリキュラム学習 + セルフプレイ コールバック。

Phase 1: ヒューリスティック AI と対戦。勝率が閾値を超えたら Phase 2 へ。
Phase 2: 過去モデルのプール（pool_size 個）からランダムに選んだモデルと対戦。
         update_freq ステップごとに現在のモデルをスナップショット保存してプールを更新。
"""
from __future__ import annotations

import os
from collections import deque

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib import MaskablePPO


class CurriculumSelfPlayCallback(BaseCallback):
    """
    Args:
        phase1_threshold: Phase 1 → 2 に移行する勝率の閾値（デフォルト 0.6）
        eval_window:      勝率を計算するエピソード数のウィンドウ（デフォルト 200）
        pool_size:        保持する過去モデルの最大数（デフォルト 5）
        update_freq:      Phase 2 でモデルプールを更新するステップ間隔（デフォルト 10,000）
        snapshot_dir:     スナップショットの保存先ディレクトリ
        verbose:          ログ出力レベル
    """

    PHASE_HEURISTIC = 1
    PHASE_SELFPLAY  = 2

    def __init__(
        self,
        phase1_threshold: float = 0.6,
        eval_window: int = 200,
        pool_size: int = 5,
        update_freq: int = 10_000,
        snapshot_dir: str = "./snapshots/",
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.phase1_threshold = phase1_threshold
        self.eval_window      = eval_window
        self.pool_size        = pool_size
        self.update_freq      = update_freq
        self.snapshot_dir     = snapshot_dir

        self.phase            = self.PHASE_HEURISTIC
        self.ep_rewards: deque[float] = deque(maxlen=eval_window)
        self.model_pool: deque        = deque(maxlen=pool_size)
        self._last_update_step        = 0

    # ------------------------------------------------------------------
    # SB3 コールバック API
    # ------------------------------------------------------------------

    def _on_training_start(self) -> None:
        os.makedirs(self.snapshot_dir, exist_ok=True)
        if self.verbose:
            print(f"[Curriculum] Phase 1 開始: ヒューリスティック相手に学習 "
                  f"(閾値={self.phase1_threshold:.0%}, ウィンドウ={self.eval_window})")

    def _on_step(self) -> bool:
        # VecMonitor が付加した episode 情報からエピソード報酬を取得
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.ep_rewards.append(info["episode"]["r"])

        # ウィンドウが溜まっていなければスキップ
        if len(self.ep_rewards) < self.eval_window:
            return True

        win_rate = sum(r >= 1.0 for r in self.ep_rewards) / len(self.ep_rewards)

        if self.phase == self.PHASE_HEURISTIC:
            self._check_phase1_transition(win_rate)

        elif self.phase == self.PHASE_SELFPLAY:
            self._check_pool_update(win_rate)

        return True

    # ------------------------------------------------------------------
    # フェーズ移行・プール更新
    # ------------------------------------------------------------------

    def _check_phase1_transition(self, win_rate: float) -> None:
        if win_rate >= self.phase1_threshold:
            self.phase = self.PHASE_SELFPLAY
            if self.verbose:
                print(f"\n[Curriculum] ★ Phase 2 移行: 勝率 {win_rate:.1%} "
                      f"≥ 閾値 {self.phase1_threshold:.1%} "
                      f"(step={self.num_timesteps:,})")
            self._update_pool_and_assign()

    def _check_pool_update(self, win_rate: float) -> None:
        if self.num_timesteps - self._last_update_step >= self.update_freq:
            if self.verbose:
                print(f"\n[Curriculum] プール更新 "
                      f"(step={self.num_timesteps:,}, 直近勝率={win_rate:.1%}, "
                      f"pool={len(self.model_pool)}/{self.pool_size})")
            self._update_pool_and_assign()

    def _update_pool_and_assign(self) -> None:
        """現在のモデルをプールに追加し、各環境の対戦相手をプールからランダム割り当て。"""
        # ディスクに保存してから load で読み直す（deepcopy は mappingproxy で失敗するため）
        path = os.path.join(self.snapshot_dir, f"snapshot_{self.num_timesteps}")
        self.model.save(path)
        model_snapshot = MaskablePPO.load(path)

        self.model_pool.append(model_snapshot)
        self._last_update_step = self.num_timesteps

        # 各環境に対戦相手を割り当て
        pool_list = list(self.model_pool)
        n_envs    = self.training_env.num_envs
        for i in range(n_envs):
            opponent = pool_list[np.random.randint(len(pool_list))]
            # ActionMasker → ReversiEnv の set_opponent() を呼ぶ
            self.training_env.env_method("set_opponent", opponent, indices=[i])

    # ------------------------------------------------------------------
    # 状態参照用プロパティ
    # ------------------------------------------------------------------

    @property
    def current_phase(self) -> int:
        return self.phase

    def current_win_rate(self) -> float:
        if not self.ep_rewards:
            return 0.0
        return sum(r >= 1.0 for r in self.ep_rewards) / len(self.ep_rewards)
