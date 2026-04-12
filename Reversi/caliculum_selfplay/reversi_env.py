# -*- coding: utf-8 -*-
"""
Gymnasium 環境ラッパー for Reversi（カリキュラム学習・セルフプレイ対応版）。

親フォルダの reversi_env.py との違い:
  - opponent 引数でヒューリスティックとモデル対戦を切り替え可能
  - set_opponent() で学習中に対戦相手を動的に更新できる
  - game.step() を使わず手を分割処理（対戦相手ロジックを差し替えるため）
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from Game_Reversi import Game_Reversi


class ReversiEnv(gym.Env):
    """
    Observation space:
        Box shape (4 * n_rows * n_cols,), dtype=float32
          [0   .. n-1 ] 自分のコマ位置
          [n   .. 2n-1] 相手のコマ位置
          [2n  .. 3n-1] 自分の合法手
          [3n  .. 4n-1] 相手の合法手

    Action space:
        Discrete(n_rows * n_cols + 1)
          0 .. n-1 : (action // n_cols, action % n_cols) にコマを置く
          n        : パス

    Args:
        opponent: None のときヒューリスティック AI（Phase 1）。
                  MaskablePPO モデルを渡すとそのモデルと対戦（Phase 2）。
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        n_rows: int = 6,
        n_cols: int = 6,
        render_mode: Optional[str] = None,
        opponent=None,
    ) -> None:
        super().__init__()
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.render_mode = render_mode
        self.opponent = opponent  # None: ヒューリスティック / model: セルフプレイ

        self.game = Game_Reversi(n_rows, n_cols)
        self.game.render = False

        n = n_rows * n_cols
        self._pass_action = n

        self.action_space = spaces.Discrete(n + 1)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(4 * n,), dtype=np.float32
        )

    # ------------------------------------------------------------------
    # 対戦相手の差し替え（CurriculumSelfPlayCallback から呼ばれる）
    # ------------------------------------------------------------------

    def set_opponent(self, opponent) -> None:
        """学習中に対戦相手モデルを更新する。None でヒューリスティックに戻す。"""
        self.opponent = opponent

    # ------------------------------------------------------------------
    # Gymnasium 必須 API
    # ------------------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        self.game.resetBoard()
        return self._get_obs_for(self.game.turn), {}

    def step(self, action: int):
        pos = self._action_to_pos(action)

        # 不正アクションのガード（マスクなし評価時など）
        if pos != (-1, -1) and not self.game.isValidMove(pos[0], pos[1], self.game.turn):
            return self._get_obs_for(self.game.turn), 0.0, False, False, {"valid_action": False}

        # 1. エージェントのコマを置く
        if pos != (-1, -1):
            self.game.putStone(pos[0], pos[1], self.game.turn, False)

        # 2. 対戦相手のコマを置く
        opp_pos = self._get_opponent_move()
        if opp_pos is not None:
            self.game.putStone(opp_pos[0], opp_pos[1], -self.game.turn, False)

        # 3. 終了判定
        agent_moves = self.game.getPositionAvail(self.game.turn)
        env_moves   = self.game.getPositionAvail(-self.game.turn)
        terminated  = not agent_moves and not env_moves

        reward = 0.0
        if terminated:
            num_agent = int(np.sum(self.game.g_board == self.game.turn))
            num_opp   = int(np.sum(self.game.g_board == -self.game.turn))
            if num_agent > num_opp:
                reward = self.game.r_win
            elif num_agent < num_opp:
                reward = self.game.r_lose
            else:
                reward = self.game.r_draw

        if self.render_mode == "human":
            self.render()

        return self._get_obs_for(self.game.turn), reward, terminated, False, {"valid_action": True}

    def render(self):
        s = str(self.game.g_board)
        if self.render_mode == "human":
            print(s)
        elif self.render_mode == "ansi":
            return s

    # ------------------------------------------------------------------
    # Action masking（MaskablePPO / ActionMasker 用）
    # ------------------------------------------------------------------

    def action_masks(self) -> np.ndarray:
        return self._action_masks_for(self.game.turn)

    # ------------------------------------------------------------------
    # 内部ヘルパー
    # ------------------------------------------------------------------

    def _get_opponent_move(self):
        """ヒューリスティックまたはモデルで対戦相手の手を決定する。"""
        opp = -self.game.turn
        if self.opponent is None:
            # ヒューリスティック（Phase 1）
            return self.game.getPosition(opp)
        else:
            # モデル対戦（Phase 2）
            valid = self.game.getPositionAvail(opp)
            if not valid:
                return None  # 強制パス
            obs   = self._get_obs_for(opp)
            masks = self._action_masks_for(opp)
            action, _ = self.opponent.predict(obs, action_masks=masks, deterministic=False)
            return self._action_to_pos(int(action))

    def _get_obs_for(self, player: int) -> np.ndarray:
        """指定プレイヤー視点の観測ベクトルを返す。"""
        board      = self.game.g_board.reshape(-1)
        my_pieces  = (board == player).astype(np.float32)
        opp_pieces = (board == -player).astype(np.float32)

        n = self.n_rows * self.n_cols
        my_valid  = np.zeros(n, dtype=np.float32)
        opp_valid = np.zeros(n, dtype=np.float32)

        for p in self.game.getPositionAvail(player):
            my_valid[p[0] * self.n_cols + p[1]] = 1.0
        for p in self.game.getPositionAvail(-player):
            opp_valid[p[0] * self.n_cols + p[1]] = 1.0

        return np.concatenate([my_pieces, opp_pieces, my_valid, opp_valid])

    def _action_masks_for(self, player: int) -> np.ndarray:
        """指定プレイヤーの合法手マスクを返す。"""
        mask  = np.zeros(self.action_space.n, dtype=bool)
        valid = self.game.getPositionAvail(player)
        if valid:
            for p in valid:
                mask[p[0] * self.n_cols + p[1]] = True
        else:
            mask[self._pass_action] = True
        return mask

    def _action_to_pos(self, action: int):
        if action == self._pass_action:
            return (-1, -1)
        return (action // self.n_cols, action % self.n_cols)
