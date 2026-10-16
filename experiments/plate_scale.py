"""実験: スイッチ・コンセントのプレート（日本の標準的なものは高さ 120mm と仮定）から、部屋の大きさを合わせられるか。

プレートは小さく、モデルが見る画像（392×518）では高さ 25 画素ほどしかない。そこで端の位置は元の写真（3072×4096）で探し、
3D の位置は、モデルの画素格子の 3D 位置を小数の画素位置で補間して求める（プレートは平らな壁の上なので補間できる）。
注意: この家のプレートの実寸は測っていない。120mm は一般的な規格値で、仮定である。
使い方（server フォルダで）: uv run python ../experiments/plate_scale.py
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parent / "server"))
sys.path.insert(0, str(here))
from gridtools import recover_grid  # noqa: E402
from roomfit.reconstruct import load_raw  # noqa: E402

OUT = here / "out" / "home_1012"
raw = load_raw(OUT / "raw.npz")
imgs = np.load(OUT / "model_images.npy")
photos = sorted(glob.glob(str(here / "data" / "home_1012" / "photos" / "*.jpg")))
PLATE_M = 0.120
# 元の写真 → モデル画像の画素: 1024 に縮小（×0.25）→ 幅 392 に合わせて縮小（×392/768）→ 上下を同じだけ切り落とす
S = 0.25 * 392 / 768
V_OFF = (1024 * 392 / 768 - 518) / 2

# プレートのおおよその位置（モデル画像の画素: 左, 上, 右, 下）。目盛りつき画像から読んだ。
PLATES = [
    (0, "スイッチ", (301, 271, 313, 296)), (0, "コンセント", (289, 455, 299, 481)),
    (11, "スイッチ", (287, 264, 296, 291)), (11, "コンセント", (270, 459, 280, 490)),
    (5, "スイッチ", (36, 257, 48, 285)), (5, "コンセント", (35, 452, 48, 475)),
    (12, "コンセント", (9, 440, 23, 462)),
]


def bilinear_xyz(xyz, u, v):
    """モデル画像の連続座標 (u, v)（画素 i の中心が i+0.5）での 3D 位置。"""
    x, y = u - 0.5, v - 0.5
    i, j = int(np.floor(x)), int(np.floor(y))
    a, b = x - i, y - j
    P = xyz[j:j + 2, i:i + 2]
    if P.shape != (2, 2, 3) or not np.isfinite(P).all():
        return None
    return P[0, 0] * (1 - a) * (1 - b) + P[0, 1] * a * (1 - b) + P[1, 0] * (1 - a) * b + P[1, 1] * a * b


def edge_rows(lum, c0, c1, r_guess, half):
    """列 c0..c1 の平均の明るさで、r_guess の近く（±half）にある最も強い横の端（小数の行）を返す。"""
    r0, r1 = int(r_guess - half), int(r_guess + half)
    prof = lum[r0:r1, c0:c1].mean(axis=1)
    k = 3
    sm = np.convolve(prof, np.ones(k) / k, mode="same")
    g = np.abs(sm[4:] - sm[:-4])
    i = int(np.argmax(g[2:-2])) + 2
    den = g[i - 1] - 2 * g[i] + g[i + 1]
    off = 0.5 * (g[i - 1] - g[i + 1]) / den if abs(den) > 1e-9 else 0.0
    return r0 + i + 2 + off + 0.5, float(g[i])


est = []
for v, kind, (x0, y0, x1, y1) in PLATES:
    g = recover_grid(raw, v, imgs[v])
    lum = np.asarray(Image.open(photos[v]).convert("L")).astype(float)
    to_orig = lambda u, w: (u / S, (w + V_OFF) / S)
    cx0, _ = to_orig(x0 + 0.3 * (x1 - x0), 0)
    cx1, _ = to_orig(x1 - 0.3 * (x1 - x0), 0)
    _, top_o = to_orig(0, y0)
    _, bot_o = to_orig(0, y1)
    h_o = bot_o - top_o
    rt, gt = edge_rows(lum, int(cx0), int(cx1), top_o, 0.22 * h_o)
    rb, gb = edge_rows(lum, int(cx0), int(cx1), bot_o, 0.22 * h_o)
    uc = (x0 + x1) / 2
    Pt, Pb = bilinear_xyz(g["xyz"], uc, rt * S - V_OFF), bilinear_xyz(g["xyz"], uc, rb * S - V_OFF)
    if Pt is None or Pb is None:
        print(f"写真 {v} {kind}: 3D 位置が取れず")
        continue
    h = float(np.linalg.norm(Pt - Pb))
    est.append((v, kind, h, PLATE_M / h, rb - rt, min(gt, gb)))
    print(f"写真 {v:2d} {kind}: 元の写真での高さ {rb - rt:.0f} 画素、端のはっきりさ {min(gt, gb):.0f} ｜ 復元での高さ {h*1000:.1f} mm → 縮尺の見積もり {PLATE_M / h:.3f}")

s = np.array([e[3] for e in est])
print(f"\nプレートからの縮尺の見積もり: 中央値 {np.median(s):.3f}、平均 {s.mean():.3f}、範囲 {s.min():.3f}〜{s.max():.3f}（{len(s)} か所）")
print("メジャーの実測 4 か所からの縮尺: 0.958〜0.964（平均 0.962）")
strong = np.array([e[3] for e in est if e[5] >= 8])
if len(strong):
    print(f"端がはっきり写ったものだけ（{len(strong)} か所）: 中央値 {np.median(strong):.3f}、範囲 {strong.min():.3f}〜{strong.max():.3f}")
(OUT / "plate_scale.json").write_text(json.dumps([{"view": e[0], "kind": e[1], "height_mm": round(e[2] * 1000, 1), "scale": round(e[3], 4)} for e in est], ensure_ascii=False, indent=1), encoding="utf-8")
