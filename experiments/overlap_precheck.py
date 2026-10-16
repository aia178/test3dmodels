"""実験: 復元の前に、写真だけから「2 枚が重なっているか」を安い計算で見積もれるか。
正解: 復元結果から出した本当の重なり（写真 i の点のうち、写真 j の点が 5cm 以内にある割合）。
見積もり: 縮小した写真（幅 48。ブラウザで 1 秒程度で計算できる大きさ）どうしを横にずらしながら重ね、明るさの模様が最もよく合う位置での一致度。
使い方（server フォルダで）: uv run python ../experiments/overlap_precheck.py
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parent / "server"))
from roomfit.reconstruct import load_raw  # noqa: E402

OUT = here / "out" / "home_1012"
raw = load_raw(OUT / "raw.npz")
P, ci = raw["points"].astype(float), raw["cam_index"]
N = len(raw["cam2world"])
rng = np.random.default_rng(0)
sub = [P[ci == v][rng.choice((ci == v).sum(), 8000, replace=False)] for v in range(N)]
trees = [cKDTree(s) for s in sub]
true = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        if i != j:
            d, _ = trees[j].query(sub[i])
            true[i, j] = (d < 0.05).mean()
true = np.minimum(true, true.T)

W, H = 48, 64


def feat(path):
    g = np.asarray(Image.open(path).convert("L").resize((W, H), Image.BILINEAR)).astype(float)
    gx = np.abs(np.diff(g, axis=1, prepend=g[:, :1]))
    gy = np.abs(np.diff(g, axis=0, prepend=g[:1, :]))
    return gx + gy          # 明るさの変化の強さ（照明の違いに強い）


F = [feat(p) for p in sorted((OUT / "images").glob("*.jpg"))]


def score(a, b):
    """横 ±70%・縦 ±20% の範囲でずらし、重なった部分の相関が最大になる所の値。重なりが小さいずれは不利にする。"""
    best = -1.0
    for dy in range(-12, 13, 4):
        for dx in range(-33, 34, 2):
            ax0, ax1 = max(0, dx), min(W, W + dx)
            ay0, ay1 = max(0, dy), min(H, H + dy)
            if ax1 - ax0 < 12 or ay1 - ay0 < 30:
                continue
            A = a[ay0:ay1, ax0:ax1]
            B = b[ay0 - dy:ay1 - dy, ax0 - dx:ax1 - dx]
            A, B = A - A.mean(), B - B.mean()
            den = np.sqrt((A * A).sum() * (B * B).sum())
            if den < 1e-6:
                continue
            c = float((A * B).sum() / den)
            best = max(best, c)
    return best


est = np.zeros((N, N))
for i in range(N):
    for j in range(i + 1, N):
        est[i, j] = est[j, i] = score(F[i], F[j])
iu = np.triu_indices(N, 1)
t, e = true[iu], est[iu]
print(f"写真の組 {len(t)} 通り: 本当の重なりと見積もりの相関 {np.corrcoef(t, e)[0, 1]:.2f}")
for lo, hi in ((0, 0.1), (0.1, 0.4), (0.4, 0.7), (0.7, 1.01)):
    m = (t >= lo) & (t < hi)
    if m.any():
        print(f"  本当の重なり {lo:.1f}〜{hi:.1f}: {m.sum():2d} 組、見積もり 中央値 {np.median(e[m]):.2f}（{e[m].min():.2f}〜{e[m].max():.2f}）")
# 使い方の想定: 各写真について「いちばんよく重なる相手」との値を見る
best_true, best_est = true.max(axis=1), est.max(axis=1)
print("各写真の、最もよく重なる相手との値（本当 / 見積もり）:", [(round(a, 2), round(b, 2)) for a, b in zip(best_true, best_est)])
# しきい値で「重なる相手がいるか」を当てられるか
for th in (0.25, 0.3, 0.35, 0.4):
    pred, truth = e >= th, t >= 0.4
    tp, fp, fn = (pred & truth).sum(), (pred & ~truth).sum(), (~pred & truth).sum()
    print(f"  見積もり {th} 以上を「重なっている」とすると: 当たり {tp}、見逃し {fn}、誤って重なり扱い {fp}（本当に重なっている組は {truth.sum()}）")

print("\n組ごとに、各写真の「最もよく合う相手」との見積もりの最小値・中央値（＝いちばん孤立した写真がどれだけ合うか）")
SETS = {"13 枚すべて（良好）": list(range(13)), "8 枚（良好）": [0, 2, 4, 5, 7, 9, 11, 12], "6 枚（良好）": [0, 2, 5, 8, 11, 12],
        "選んだ 4 枚（良好）": [0, 5, 11, 12], "3 枚（要注意）": [0, 11, 12], "2 枚（要注意）": [11, 12],
        "間引き 0,3,6,9,12（要注意）": [0, 3, 6, 9, 12], "間引き 1,4,7,10（要注意）": [1, 4, 7, 10],
        "間引き 2,5,8,11（要注意）": [2, 5, 8, 11], "間引き 0,4,8,12（要注意）": [0, 4, 8, 12]}
for name, idx in SETS.items():
    b_est = [max(est[i, j] for j in idx if j != i) for i in idx]
    b_true = [max(true[i, j] for j in idx if j != i) for i in idx]
    print(f"  {name}: 見積もり 最小 {min(b_est):.2f}・中央値 {np.median(b_est):.2f} ｜ 本当の重なり 最小 {min(b_true):.2f}・中央値 {np.median(b_true):.2f}")
