#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
学習済みモデルと人間が対戦するスクリプト。
モデルが先手（M）、人間が後手（H）。

使い方:
    python play.py
    python play.py --model checkpoints/reversi_curriculum_400000_steps.zip
"""
from __future__ import annotations

import argparse
import numpy as np
from sb3_contrib import MaskablePPO
from Game_Reversi import Game_Reversi

MODEL_PATH  = "reversi_curriculum_ppo"
BOARD_SIZE  = 6
MODEL_TURN  =  1   # モデル（先手）
HUMAN_TURN  = -1   # 人間（後手）


def display_board(game: Game_Reversi) -> None:
    symbols = {MODEL_TURN: "M", HUMAN_TURN: "H", 0: "."}
    print()
    print("  " + " ".join(str(c) for c in range(game.n_cols)))
    for r in range(game.n_rows):
        row = " ".join(symbols[game.g_board[r, c]] for c in range(game.n_cols))
        print(f"{r} {row}")
    print()


def get_obs(game: Game_Reversi, player: int) -> np.ndarray:
    n     = game.n_rows * game.n_cols
    board = game.g_board.reshape(-1)
    my_pieces  = (board == player).astype(np.float32)
    opp_pieces = (board == -player).astype(np.float32)
    my_valid   = np.zeros(n, dtype=np.float32)
    opp_valid  = np.zeros(n, dtype=np.float32)
    for p in game.getPositionAvail(player):
        my_valid[p[0] * game.n_cols + p[1]] = 1.0
    for p in game.getPositionAvail(-player):
        opp_valid[p[0] * game.n_cols + p[1]] = 1.0
    return np.concatenate([my_pieces, opp_pieces, my_valid, opp_valid])


def get_masks(game: Game_Reversi, player: int) -> np.ndarray:
    n     = game.n_rows * game.n_cols
    mask  = np.zeros(n + 1, dtype=bool)
    valid = game.getPositionAvail(player)
    if valid:
        for p in valid:
            mask[p[0] * game.n_cols + p[1]] = True
    else:
        mask[n] = True
    return mask


def do_model_turn(model: MaskablePPO, game: Game_Reversi) -> None:
    """モデルが手を選んで石を置く。"""
    valid = game.getPositionAvail(MODEL_TURN)
    if not valid:
        print("モデル: パス")
        return

    obs   = get_obs(game, MODEL_TURN)
    masks = get_masks(game, MODEL_TURN)
    action, _ = model.predict(obs, action_masks=masks, deterministic=True)
    action = int(action)

    if action == game.n_rows * game.n_cols:
        print("モデル: パス")
        return

    row, col = action // game.n_cols, action % game.n_cols
    game.putStone(row, col, MODEL_TURN, False)
    print(f"モデル: ({row}, {col})")


def do_human_turn(game: Game_Reversi) -> None:
    """人間が手を入力して石を置く。"""
    valid = game.getPositionAvail(HUMAN_TURN)
    if not valid:
        print("あなたの合法手がありません → パス")
        return

    valid_list = [(int(p[0]), int(p[1])) for p in valid]
    print(f"合法手: {valid_list}")

    while True:
        try:
            raw = input("手を入力 (行,列) / パスは '-1,-1': ").strip()
            row, col = map(int, raw.split(","))
            if (row, col) == (-1, -1):
                print("パス")
                return
            if not game.isValidMove(row, col, HUMAN_TURN):
                print("その手は置けません。")
                continue
            game.putStone(row, col, HUMAN_TURN, False)
            return
        except (ValueError, IndexError):
            print("形式が正しくありません。例: 2,3")


def check_end(game: Game_Reversi) -> bool:
    """終了判定。終わっていれば結果を表示して True を返す。"""
    if game.getPositionAvail(MODEL_TURN) or game.getPositionAvail(HUMAN_TURN):
        return False

    num_model = int(np.sum(game.g_board == MODEL_TURN))
    num_human = int(np.sum(game.g_board == HUMAN_TURN))

    print("=" * 32)
    print("ゲーム終了")
    print(f"  モデル (M) : {num_model} コマ")
    print(f"  あなた (H) : {num_human} コマ")
    if num_model > num_human:
        print("  結果 → モデルの勝ち")
    elif num_human > num_model:
        print("  結果 → あなたの勝ち！")
    else:
        print("  結果 → 引き分け")
    print("=" * 32)
    return True


def main(model_path: str, board_size: int) -> None:
    print(f"モデル読み込み中: {model_path}")
    model = MaskablePPO.load(model_path)

    game = Game_Reversi(board_size, board_size)
    game.render = False

    print(f"\n{board_size}×{board_size} リバーシ開始")
    print("  M = モデル（先手）  H = あなた（後手）  . = 空")
    display_board(game)

    while True:
        print("--- モデルのターン ---")
        do_model_turn(model, game)
        display_board(game)
        if check_end(game):
            break

        print("--- あなたのターン ---")
        do_human_turn(game)
        display_board(game)
        if check_end(game):
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play against trained Reversi model")
    parser.add_argument("--model", "-m", type=str, default=MODEL_PATH,
                        help=f"モデルパス (default: {MODEL_PATH})")
    parser.add_argument("--board", "-b", type=int, default=BOARD_SIZE,
                        help=f"盤面サイズ (default: {BOARD_SIZE})")
    args = parser.parse_args()
    main(args.model, args.board)
