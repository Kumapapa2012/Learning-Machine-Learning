# -*- coding: utf-8 -*-
import numpy as np

# 全方向ベクトル（isValidMove, putStone で共用）
DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


class Game_Reversi:
    # 報酬
    r_win = 1.0
    r_draw = -0.5
    r_lose = -1.0

    # 表示
    render = True

    def __init__(self, n_rows, n_cols):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.turn = 1  # agent=1, environment=-1
        self.g_board = np.zeros([self.n_rows, self.n_cols], dtype=np.int16)
        self._init_board()

    def _init_board(self):
        """盤面を初期状態（中央4コマ配置）に設定する。"""
        r, c = self.n_rows // 2, self.n_cols // 2
        self.g_board[r - 1, c - 1] = 1
        self.g_board[r - 1, c] = -1
        self.g_board[r, c - 1] = -1
        self.g_board[r, c] = 1

    def resetBoard(self):
        """ボードをゲーム初期状態に戻す。"""
        self.g_board[:] = 0
        self._init_board()

    def isValidMove(self, row, column, c):
        """
        指定した row, col が c にとって合法手かを判定する。
        1 方向でも相手のコマを返せるなら True。
        """
        if self.g_board[row, column] != 0:
            return False

        pos = np.array([row, column])
        for direction in DIRECTIONS:
            d = np.array(direction)
            cpos = pos + d
            if not (0 <= cpos[0] < self.n_rows and 0 <= cpos[1] < self.n_cols):
                continue
            if self.g_board[tuple(cpos)] != -c:
                continue
            # 相手コマの先に自コマがあるか探索
            cpos = cpos + d
            while 0 <= cpos[0] < self.n_rows and 0 <= cpos[1] < self.n_cols:
                cell = self.g_board[tuple(cpos)]
                if cell == 0:
                    break
                if cell == c:
                    return True
                cpos = cpos + d
        return False

    def putStone(self, row, column, c, sim):
        """
        コマを置いて相手のコマを返す。
        sim=True の場合はコピーのボードで処理し、実際のボードはタッチしない。
        戻り値: 返した相手コマの数
        """
        board = np.copy(self.g_board) if sim else self.g_board
        pos = np.array([row, column])
        board[tuple(pos)] = c
        num_flip = 0

        for direction in DIRECTIONS:
            d = np.array(direction)
            cpos = pos + d
            can_flip = False
            while 0 <= cpos[0] < self.n_rows and 0 <= cpos[1] < self.n_cols:
                cell = board[tuple(cpos)]
                if cell == -c:
                    cpos = cpos + d
                elif cell == c:
                    can_flip = True
                    break
                else:  # cell == 0
                    break

            if can_flip:
                cpos = cpos - d
                while not np.array_equal(cpos, pos):
                    board[tuple(cpos)] = c
                    num_flip += 1
                    cpos = cpos - d

        return num_flip

    def getPositionAvail(self, c):
        """現在の盤面で c がコマを置ける場所のリストを返す。"""
        empty_cells = np.argwhere(self.g_board == 0)
        return [p for p in empty_cells if self.isValidMove(p[0], p[1], c)]

    def getPosition(self, c):
        """
        コマを置く場所を以下の優先順位で決定して返す。
        1. 四隅に置ける場合は 90% の確率でそこを取る
        2. それ以外は 80% の確率で最も返せる数が多い場所を取る
        3. 決まらない場合はランダム
        """
        can_put = self.getPositionAvail(c)
        if not can_put:
            return None

        corner_positions = []
        best_pos = None
        best_count = 0

        for p in can_put:
            if (p[0] in (0, self.n_rows - 1)) and (p[1] in (0, self.n_cols - 1)):
                corner_positions.append(p)
            flip_count = self.putStone(p[0], p[1], c, True)
            if flip_count > best_count:
                best_count = flip_count
                best_pos = p

        t_rnd = np.random.random()

        if corner_positions and t_rnd < 0.9:
            return corner_positions[np.random.randint(len(corner_positions))]
        if best_pos is not None and t_rnd < 0.8:
            return best_pos
        return can_put[np.random.randint(len(can_put))]

    def step(self, player_pos):
        """
        1. player_pos にエージェントのコマを置く。
        2. 環境側のコマを置く。
        3. 勝敗判定と報酬を返す。
        戻り値: (board, reward, done)
        """
        # エージェントの手
        if player_pos == (-1, -1):
            if self.render:
                print("PASS!")
        elif self.isValidMove(player_pos[0], player_pos[1], self.turn):
            self.putStone(player_pos[0], player_pos[1], self.turn, False)
        else:
            print("Bad Move...%i, %i, %i" % (player_pos[0], player_pos[1], self.turn))
            return self.g_board, 0, False

        if self.render:
            print("Your turn ends...")
            print(self.g_board)

        # 環境の手
        env_pos = self.getPosition(-self.turn)
        if env_pos is None:
            if self.render:
                print("PASS!")
        else:
            self.putStone(env_pos[0], env_pos[1], -self.turn, False)

        if self.render:
            print("My turn ends...")
            print(self.g_board)

        # 終了判定
        agent_moves = self.getPositionAvail(self.turn)
        env_moves = self.getPositionAvail(-self.turn)
        done = not agent_moves and not env_moves

        reward = 0.0
        if done:
            if self.render:
                print("****************Finish****************")
            num_agent = np.sum(self.g_board == self.turn)
            num_env = np.sum(self.g_board == -self.turn)
            if self.render:
                print("you:%i/environment:%i" % (num_agent, num_env))
            if num_agent > num_env:
                reward = self.r_win
                if self.render:
                    print("you win!")
            elif num_agent < num_env:
                reward = self.r_lose
                if self.render:
                    print("you lose!")
            else:
                reward = self.r_draw
                if self.render:
                    print("Draw!")

        return self.g_board, reward, done
