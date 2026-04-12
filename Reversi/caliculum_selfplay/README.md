# Reversi RL — カリキュラム学習 + セルフプレイ

Gymnasium + MaskablePPO（stable-baselines3）によるリバーシ強化学習の実装です。  
ヒューリスティック AI との対戦（Phase 1）から、過去モデルとのセルフプレイ（Phase 2）へと段階的に学習します。

---

## ファイル構成

| ファイル | 内容 |
|---|---|
| `Game_Reversi.py` | リバーシのゲームロジック（盤面管理・合法手判定・ヒューリスティック AI） |
| `reversi_env.py` | Gymnasium カスタム環境。対戦相手を差し替え可能 |
| `selfplay_callback.py` | カリキュラム学習・セルフプレイを制御する SB3 コールバック |
| `train.py` | 学習スクリプト |
| `evaluate.py` | 評価スクリプト |
| `play.py` | 人間 vs モデルの対戦スクリプト |
| `requirements.txt` | 依存パッケージ |

---

## セットアップ

```bash
pip install -r requirements.txt
```

GPU を使う場合は PyTorch を CUDA 版に入れ替えてください。

```bash
# CUDA バージョンを確認
nvidia-smi

# 例: CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## 学習

```bash
# デフォルト設定で学習（200万ステップ）
python train.py

# 環境チェックあり
python train.py --check

# パラメータ指定
python train.py --timesteps 2000000 --threshold 0.6 --pool 5 --envs 4
```

### 主なオプション

| オプション | デフォルト | 内容 |
|---|---|---|
| `--timesteps` | 2,000,000 | 総学習ステップ数 |
| `--threshold` | 0.6 | Phase 1 → 2 移行の勝率閾値 |
| `--pool` | 5 | セルフプレイの過去モデルプールサイズ |
| `--envs` | 4 | 並列環境数 |
| `--board` | 6 | 盤面サイズ |
| `--check` | - | 学習前に環境チェックを実行 |

### 学習フェーズ

```
Phase 1: ヒューリスティック AI と対戦
  └─ 直近 200 ゲームの勝率 ≥ 閾値 になったら...
        ↓
Phase 2: セルフプレイ
  └─ 過去モデルのプール（最大 pool 個）からランダムに対戦相手を選択
  └─ 10,000 ステップごとにプールを更新
```

### 学習の確認（TensorBoard）

```bash
tensorboard --logdir ./reversi_tb/
```

`ep_rew_mean` が Phase 2 移行後に一時的に下がり、その後上昇するのが正常な挙動です。

---

## 評価

```bash
# ヒューリスティック AI 相手に 1000 ゲーム評価
python evaluate.py --model checkpoints/reversi_curriculum_400000_steps.zip

# 別モデル相手に評価
python evaluate.py --model <モデルパス> --opponent <対戦相手モデルパス>

# 盤面を表示しながら評価
python evaluate.py --render --episodes 10
```

---

## 人間との対戦

```bash
python play.py --model checkpoints/reversi_curriculum_400000_steps.zip
```

- モデルが先手（`M`）、人間が後手（`H`）
- 手の入力形式: `行,列`（例: `2,3`）
- パスは `-1,-1`

```
  0 1 2 3 4 5
0 . . . . . .
1 . . . . . .
2 . . M H . .
3 . . H M . .
4 . . . . . .
5 . . . . . .

合法手: [(2, 3), (3, 2), (4, 4)]
手を入力 (行,列) / パスは '-1,-1':
```

---

## 出力ファイル

| パス | 内容 |
|---|---|
| `checkpoints/` | 100,000 ステップごとのチェックポイント |
| `snapshots/` | セルフプレイ用の過去モデルスナップショット |
| `reversi_tb/` | TensorBoard ログ |
| `reversi_curriculum_ppo.zip` | 最終モデル |

---

## 動作環境

- Python 3.10+
- gymnasium >= 0.29
- stable-baselines3 >= 2.3
- sb3-contrib >= 2.3
- torch >= 2.2
