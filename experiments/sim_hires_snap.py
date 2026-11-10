"""実験: 端を探す画像を、モデルが見た大きさ（392×518）から 2 倍・4 倍にすると、タップ計測の精度が上がるか。
3D の位置は元の画素格子のまま（内側の画素から直線で外挿）。端の位置だけを細かい画像で探す。
使い方（server フォルダで）: uv run python ../experiments/sim_hires_snap.py
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

here = Path(__file__).resolve().parent
src = (here / "sim_tap_snap.py").read_text(encoding="utf-8")
ns = {"__file__": str(here / "sim_tap_snap.py")}
exec(compile(src[: src.index("# 写真 12（引き戸の正面）")], "sim_tap_snap", "exec"), ns)
load, bilinear, OUT, INVALID = ns["load"], ns["bilinear"], ns["OUT"], ns["INVALID"]


def lum_at_scale(n, k):
    """前処理済みの写真（768×1024）から、モデル画像の k 倍の大きさ・同じ切り抜きの明るさ画像を作る。"""
    im = Image.open(OUT / "images" / f"{n:03d}.jpg").convert("L")
    W = 392 * k
    Hfull = round(im.height * W / im.width)
    im = im.resize((W, Hfull), Image.LANCZOS)
    top = (Hfull - 518 * k) // 2
    return np.asarray(im.crop((0, top, W, top + 518 * k))).astype(float)


def snap_k(Lk, k, xyz, tap, other, reach=12.0):
    d = np.array(other, float) - np.array(tap, float)
    n = np.linalg.norm(d)
    if n < 8:
        return None
    d /= n
    step = 0.5 / k
    ts = np.arange(-reach, reach + 1e-9, step)
    prof = np.array([bilinear(Lk, (tap[0] + t * d[0]) * k - 0.5, (tap[1] + t * d[1]) * k - 0.5) for t in ts])
    w = 4 * k                                   # 2 画素（モデル画像での）離れた差
    g = np.abs(prof[w:] - prof[:-w])
    i = int(np.argmax(g))
    if g[i] < 12:
        return None
    off = 0.0
    if 0 < i < len(g) - 1:
        den = g[i - 1] - 2 * g[i] + g[i + 1]
        off = 0.5 * (g[i - 1] - g[i + 1]) / den if abs(den) > 1e-9 else 0.0
    e = np.array(tap, float) + (ts[i + w // 2] + off * step) * d
    ks, P = [], []
    for kk in np.arange(1.5, 5.6, 1.0):
        q = e + kk * d
        ui, vi = int(np.floor(q[0])), int(np.floor(q[1]))
        if 0 <= ui < xyz.shape[1] and 0 <= vi < xyz.shape[0] and xyz[vi, ui, 0] != INVALID:
            ks.append(kk)
            P.append(xyz[vi, ui] / 1000.0)
    if len(ks) < 3:
        return None
    return np.polyfit(ks, np.array(P), 1)[1]


def run(n, left, right, label, careful, trials=300, jitter=10.0):
    _, xyz = load(n)
    out = []
    for k in (1, 2, 4):
        Lk = lum_at_scale(n, k)
        rng = np.random.default_rng(0)
        ws = []
        for _ in range(trials):
            f = rng.uniform(0.15, 0.85)
            a = np.array(left[0]) + f * (np.array(left[1]) - np.array(left[0])) + rng.uniform(-jitter, jitter, 2)
            b = np.array(right[0]) + f * (np.array(right[1]) - np.array(right[0])) + rng.uniform(-jitter, jitter, 2)
            pa, pb = snap_k(Lk, k, xyz, a, b), snap_k(Lk, k, xyz, b, a)
            if pa is not None and pb is not None:
                ws.append(np.linalg.norm(pa - pb) * 100)
        ws = np.array(ws)
        out.append(f"{k} 倍: 中央値 {np.median(ws):.1f}・標準偏差 {ws.std():.2f}（{len(ws)}/{trials}）")
    print(f"{label}（丁寧に測った値 {careful}）: " + " ｜ ".join(out))


run(12, ((14, 40), (39, 480)), ((217, 35), (212.5, 495)), "引き戸・写真 12", 89.5)
run(5, ((59, 60), (56, 490)), ((256.5, 40), (227.5, 505)), "引き戸・写真 5", 87.3)
run(0, ((20, 50), (23.5, 500)), ((248.5, 60), (231, 475)), "クローゼット・写真 0", 106.3)
