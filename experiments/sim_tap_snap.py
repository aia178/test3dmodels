"""実験: 写真をタップして測るとき、「タップした近くの端へ吸い付ける」処理で精度が上がるかを、タップのずれを乱数で作って確かめる。
アプリが使うのと同じファイル（views/NNN.jpg と NNN.xyz）を使う。
使い方（server フォルダで）: uv run python ../experiments/sim_tap_snap.py
"""
from pathlib import Path

import numpy as np
from PIL import Image

OUT = Path(__file__).resolve().parent / "out" / "home_1012"
INVALID = -32768


def load(n):
    img = np.asarray(Image.open(OUT / "views" / f"{n:03d}.jpg").convert("RGB")).astype(float)
    h, w = img.shape[:2]
    xyz = np.frombuffer((OUT / "views" / f"{n:03d}.xyz").read_bytes(), dtype="<i2").reshape(h, w, 3)
    return img @ np.array([0.299, 0.587, 0.114]), xyz


def lookup(xyz, u, v):
    """いまのアプリの方法: タップした画素のまわり 5×5 の中央値。"""
    h, w = xyz.shape[:2]
    ui, vi = int(round(u)), int(round(v))
    win = xyz[max(0, vi - 2):vi + 3, max(0, ui - 2):ui + 3].reshape(-1, 3)
    win = win[win[:, 0] != INVALID]
    return None if len(win) < 5 else np.median(win, axis=0) / 1000.0


def bilinear(L, u, v):
    h, w = L.shape
    u, v = min(max(u, 0), w - 1.001), min(max(v, 0), h - 1.001)
    i, j = int(u), int(v)
    a, b = u - i, v - j
    return L[j, i] * (1 - a) * (1 - b) + L[j, i + 1] * a * (1 - b) + L[j + 1, i] * (1 - a) * b + L[j + 1, i + 1] * a * b


def snap(L, xyz, tap, other, reach=12.0):
    """タップ位置から、もう一方の点へ向かう直線に沿って ±reach 画素の範囲で、明るさが最も急に変わる所（端）を探す。
    3D の位置は、端の内側（もう一方の点の側）の画素から直線で外挿する。端が見つからなければ None。"""
    d = np.array(other, float) - np.array(tap, float)
    n = np.linalg.norm(d)
    if n < 8:
        return None
    d /= n
    ts = np.arange(-reach, reach + 0.01, 0.5)
    prof = np.array([bilinear(L, tap[0] + t * d[0] - 0.5, tap[1] + t * d[1] - 0.5) for t in ts])
    g = np.abs(prof[4:] - prof[:-4])          # 2 画素離れた差（ぼけた端でも拾えるように）
    # 範囲内で最もはっきりした端を選ぶ（「タップに最も近い端」も試したが、細い枠の継ぎ目へ飛び移って悪化した）
    k = int(np.argmax(g))
    if g[k] < 12:
        return None
    off = 0.0
    if 0 < k < len(g) - 1:
        den = g[k - 1] - 2 * g[k] + g[k + 1]
        off = 0.5 * (g[k - 1] - g[k + 1]) / den if abs(den) > 1e-9 else 0.0
    t_edge = ts[k + 2] + off * 0.5
    e = np.array(tap, float) + t_edge * d
    ks, P = [], []
    for kk in np.arange(1.5, 5.6, 1.0):  # 細い枠の中に収まるよう、内側 1.5〜5.5 画素だけを使う
        q = e + kk * d
        ui, vi = int(np.floor(q[0])), int(np.floor(q[1]))
        if 0 <= ui < xyz.shape[1] and 0 <= vi < xyz.shape[0] and xyz[vi, ui, 0] != INVALID:
            ks.append(kk)
            P.append(xyz[vi, ui] / 1000.0)
    if len(ks) < 3:
        return None
    coef = np.polyfit(ks, np.array(P), 1)
    return coef[1], e


def run(n, left, right, label, truth_cm, trials=400, jitter=10.0):
    L, xyz = load(n)
    rng = np.random.default_rng(0)
    raw, snp, fb = [], [], 0
    for _ in range(trials):
        f = rng.uniform(0.15, 0.85)
        a = np.array(left[0]) + f * (np.array(left[1]) - np.array(left[0]))
        b = np.array(right[0]) + f * (np.array(right[1]) - np.array(right[0]))
        ta, tb = a + rng.uniform(-jitter, jitter, 2), b + rng.uniform(-jitter, jitter, 2)
        pa, pb = lookup(xyz, ta[0] - 0.5, ta[1] - 0.5), lookup(xyz, tb[0] - 0.5, tb[1] - 0.5)
        if pa is not None and pb is not None:
            raw.append(np.linalg.norm(pa - pb) * 100)
        sa, sb = snap(L, xyz, ta, tb), snap(L, xyz, tb, ta)
        if sa is None or sb is None:
            fb += 1
            continue
        snp.append(np.linalg.norm(sa[0] - sb[0]) * 100)
    raw, snp = np.array(raw), np.array(snp)
    print(f"{label}（丁寧に測った値 {truth_cm} cm、タップのずれ ±{jitter:.0f} 画素）")
    print(f"  そのまま: 中央値 {np.median(raw):.1f} cm、ばらつき（10〜90%） {np.percentile(raw, 10):.1f}〜{np.percentile(raw, 90):.1f}、標準偏差 {raw.std():.2f}")
    print(f"  端へ吸い付け: 中央値 {np.median(snp):.1f} cm、ばらつき（10〜90%） {np.percentile(snp, 10):.1f}〜{np.percentile(snp, 90):.1f}、標準偏差 {snp.std():.2f} ｜ 端が見つからなかった割合 {fb / trials * 100:.0f}%")


# 写真 12（引き戸の正面）: 左の縦枠の外側と戸の右端（measure_home_1012.py と同じ位置）
run(12, ((14, 40), (39, 480)), ((217, 35), (212.5, 495)), "引き戸の幅・写真 12", 89.5)
# 写真 5（引き戸を斜めから）
run(5, ((59, 60), (56, 490)), ((256.5, 40), (227.5, 505)), "引き戸の幅・写真 5", 87.3)
# 写真 0（クローゼットを斜めから）: 枠の外側〜外側
run(0, ((20, 50), (23.5, 500)), ((248.5, 60), (231, 475)), "クローゼットの幅・写真 0", 106.3)
