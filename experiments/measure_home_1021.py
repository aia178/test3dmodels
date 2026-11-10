"""評価: 自宅の一角を 2026-10-21 に撮影アプリ（vivo、AR の位置つき）で記録し、サーバーの AR の流れで復元した結果を、メジャーの実測と比べる。

測る 4 か所と端の取り方は measure_home_1012.py と同じ（端の色の境目を自動で探し、物の側の 3D 点を直線で外挿する）。
違いは 2 つ。写真と端のおおよその位置がこの記録のもの。床は形状抽出が使えない場合もあるので、畳の画素の 3D 点に平面を当てて求める。
この復元では写真どうしの床が食い違った（写真 0・1・5 の畳が、写真 3・4・7 の畳より 8〜11cm 高い）ので、
高さはその写真に写る畳に当てた平面から測る（写真の中の形だけで決まり、写真どうしの重ね合わせのずれが入らない）。
縮尺はモデルの出力のまま（AR の位置を入力に渡しただけで、実測での補正なし）。

使い方（server フォルダで）: uv run python ../experiments/measure_home_1021.py <ジョブのフォルダ>
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
from roomfit.reconstruct import load_raw  # noqa: E402
from roomfit.worker import view_grids  # noqa: E402

OUT = Path(sys.argv[1])
MEASURED = {"door_width": 85.0, "door_height": 219.0, "closet_width": 102.0, "closet_panel": 226.0}
# 畳だけが写る範囲（写真番号: (x0, y0, x1, y1)、モデル画像の画素）。椅子と座布団を避けた
FLOOR_BOXES = {4: (60, 380, 240, 445), 0: (130, 470, 290, 515), 3: (150, 380, 290, 440)}
VIEW_FLOOR = {0: [(130, 470, 290, 515), (0, 470, 110, 515)], 5: [(20, 485, 280, 515)], 1: [(10, 420, 60, 470), (60, 480, 140, 515)]}

raw = load_raw(OUT / "raw.npz")
imgs = raw["view_images"]
grids = view_grids(raw)


def fit_floor(boxes, label="床"):
    P = np.concatenate([grids[v][1][y0:y1, x0:x1].reshape(-1, 3) for v, (x0, y0, x1, y1) in boxes])
    P = P[np.isfinite(P[:, 0])].astype(np.float64)
    rng = np.random.default_rng(0)
    best = None
    for _ in range(500):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        inl = np.abs((P - a) @ n) < 0.01
        if best is None or inl.sum() > best.sum():
            best = inl
    Q = P[best]
    c = Q.mean(0)
    n = np.linalg.svd(Q - c)[2][2]
    cams = raw["cam2world"][:, :3, 3]
    if np.median((cams - c) @ n) < 0:   # カメラのある側を上にする
        n = -n
    res = (Q - c) @ n
    print(f"{label}: 点 {len(P)} のうち平面上 {best.mean() * 100:.0f}% ｜ 平面からのずれ {np.std(res) * 1000:.1f}mm ｜ カメラの高さ {np.median((cams - c) @ n) * 100:.0f}cm")
    return c, n


FLOOR_C, FLOOR_N = fit_floor(list(FLOOR_BOXES.items()))
OWN_FLOOR = {v: fit_floor([(v, b) for b in bs], f"写真 {v} の床") for v, bs in VIEW_FLOOR.items()}
_g = {}


def grid(v):
    if v not in _g:
        xyz = grids[v][1].astype(np.float64)
        _g[v] = {"room": xyz, "h": (xyz - FLOOR_C) @ FLOOR_N,
                 "wood": imgs[v][..., 0].astype(float) - imgs[v][..., 2].astype(float),
                 "lum": imgs[v].astype(float) @ np.array([0.299, 0.587, 0.114])}
    return _g[v]


def crossing(profile, rising):
    n = len(profile)
    k = max(2, n // 4)
    a, b = np.median(profile[:k]), np.median(profile[-k:])
    lo, hi = (a, b) if rising else (b, a)
    if hi - lo < 30:
        return None
    mid = (lo + hi) / 2
    s = (profile - mid) if rising else (mid - profile)
    idx = np.flatnonzero((s[:-1] < 0) & (s[1:] >= 0))
    if len(idx) != 1:
        return None
    i = idx[0]
    return i + (-s[i]) / (s[i + 1] - s[i])


def extrapolate(xyz_line, f, inside, n_fit):
    if inside > 0:
        ks = np.arange(int(np.ceil(f + n_fit[0])), int(np.floor(f + n_fit[1])) + 1)
    else:
        ks = np.arange(int(np.floor(f - n_fit[0])), int(np.ceil(f - n_fit[1])) - 1, -1)
    ks = ks[(ks >= 0) & (ks < len(xyz_line))]
    P = xyz_line[ks]
    ok = np.isfinite(P[:, 0])
    if ok.sum() < 3:
        return None
    coef = np.polyfit(ks[ok], P[ok], 1)
    return coef[0] * f + coef[1]


def edge_points(v, axis, lines, p0, p1, hw, rising, inside, n_fit=(1.5, 7.0)):
    """measure_home_1012.py と同じ。p0, p1 は端のおおよその位置（モデル画像の画素 (x, y)）。"""
    g = grid(v)
    pts, pix = [], []
    for t in lines:
        if axis == "row":
            if not (min(p0[1], p1[1]) <= t <= max(p0[1], p1[1])):
                continue
            e = p0[0] + (p1[0] - p0[0]) * (t - p0[1]) / (p1[1] - p0[1])
            prof, xyz = g["wood"][t, :], g["room"][t, :, :]
        else:
            if not (min(p0[0], p1[0]) <= t <= max(p0[0], p1[0])):
                continue
            e = p0[1] + (p1[1] - p0[1]) * (t - p0[0]) / (p1[0] - p0[0])
            prof, xyz = g["wood"][:, t], g["room"][:, t, :]
        a, b = int(round(e - hw)), int(round(e + hw))
        if a < 0 or b >= len(prof):
            continue
        f = crossing(prof[a:b + 1], rising)
        if f is None:
            continue
        f += a
        p = extrapolate(xyz, f, inside, n_fit)
        if p is None:
            continue
        pts.append(p)
        pix.append((f, t) if axis == "row" else (t, f))
    return np.array(pts), np.array(pix)


def dark_line(v, cols, e_of_c, below, hw=5):
    """横に走る暗い線（扉の上のすき間・下の敷居との境目）を列ごとに探し、below=True なら線の下側（板の上端）、
    False なら線の上側（板の下端）の 3D 点を、板の側の点から外挿して返す。"""
    g = grid(v)
    out, pix = [], []
    for c in cols:
        e = e_of_c(c)
        a, b = int(round(e - hw)), int(round(e + hw))
        seg = g["lum"][a:b + 1, c]
        m = int(np.argmin(seg))
        if m in (0, len(seg) - 1) or np.median(seg) - seg[m] < 12:
            continue
        side = seg[m + 1:] if below else seg[:m][::-1]
        mid = (seg[m] + np.median(side[1:4])) / 2
        k = 0
        while k < len(side) - 1 and side[k] < mid:
            k += 1
        f = a + m + (k + 0.5) * (1 if below else -1)
        p = extrapolate(g["room"][:, c, :], f, +1 if below else -1, (1.0, 6.0))
        if p is not None:
            out.append(p)
            pix.append((c, f))
    return np.array(out), np.array(pix)


def height(P, v=None):
    c, n = OWN_FLOOR.get(v, (FLOOR_C, FLOOR_N))
    return (P - c) @ n


def line_fit(P):
    c = P.mean(0)
    d = np.linalg.svd(P - c)[2][0]
    return c, d if d @ FLOOR_N >= 0 else -d


def width_between(PL, PR, name):
    """左右の端の直線を、床からの同じ高さで比べた水平の距離。"""
    (cl, dl), (cr, dr) = line_fit(PL), line_fit(PR)

    def at(c, d, y):
        return c + d * (y - height(c)) / (d @ FLOOR_N)
    ws = []
    for y in (0.4, 1.0, 1.6):
        dv = at(cl, dl, y) - at(cr, dr, y)
        ws.append(float(np.linalg.norm(dv - (dv @ FLOOR_N) * FLOOR_N)))
    tilt = [float(np.degrees(np.arccos(min(1.0, abs(d @ FLOOR_N))))) for d in (dl, dr)]
    print(f"  {name}: 幅 {ws[1] * 100:.1f} cm（高さ 40cm で {ws[0] * 100:.1f} / 160cm で {ws[2] * 100:.1f}） ｜ 端の傾き 左 {tilt[0]:.1f}° 右 {tilt[1]:.1f}° ｜ 点数 {len(PL)}/{len(PR)}")
    return ws[1] * 100


def height_of(P, name, v):
    y = height(P, v)
    print(f"  {name}: 高さ 中央値 {np.median(y) * 100:.1f} cm（{np.percentile(y, 10) * 100:.1f}〜{np.percentile(y, 90) * 100:.1f}） ｜ 点数 {len(P)}")
    return float(np.median(y)) * 100


overlay, found = {}, {k: [] for k in ("door_width", "door_height", "closet_width", "panel_top", "panel_bottom")}


def mark(v, pix, color):
    if len(pix):
        overlay.setdefault(v, []).append((pix, color))


print(f"フォルダ: {OUT} ｜ 写真 {len(imgs)} 枚 ｜ 縮尺: モデルの出力のまま")
print("== 引き戸（写真 5: 正面）")
L, pl = edge_points(5, "row", range(40, 461, 5), (31, 30), (37, 470), 8, True, +1, (1.0, 5.0))
R, pr = edge_points(5, "row", range(40, 461, 5), (222, 30), (218, 470), 8, False, -1)
mark(5, pl, (255, 0, 0)); mark(5, pr, (255, 0, 0))
if len(L) > 10 and len(R) > 10:
    found["door_width"].append(width_between(L, R, "幅（左の縦枠の外側〜戸の右端）"))
Tp, pt = edge_points(5, "col", range(40, 221, 4), (30, 8), (230, 3), 5, True, +1, (1.0, 5.0))
mark(5, pt, (0, 0, 255))
if len(Tp) > 10:
    found["door_height"].append(height_of(Tp, "上枠の上端", 5))
print("== 引き戸（写真 1: 斜め）")
Tp, pt = edge_points(1, "col", range(60, 151, 3), (55, 50.3), (150, 16.5), 6, True, +1, (1.0, 5.0))
mark(1, pt, (0, 0, 255))
if len(Tp) > 10:
    found["door_height"].append(height_of(Tp, "上枠の上端", 1))
print("== クローゼット（写真 0: 斜め）")
L, pl = edge_points(0, "row", range(100, 431, 5), (39, 60), (40, 440), 7, True, +1, (1.0, 5.0))
R, pr = edge_points(0, "row", range(100, 431, 5), (217, 80), (201, 430), 8, False, -1, (1.0, 5.0))
mark(0, pl, (255, 0, 0)); mark(0, pr, (255, 0, 0))
if len(L) > 10 and len(R) > 10:
    found["closet_width"].append(width_between(L, R, "幅（枠の外側〜外側）"))
Tp, pt = dark_line(0, [c for c in range(48, 206, 3) if not 124 <= c <= 134], lambda c: 54 + (c - 45) * (72 - 54) / (210 - 45), below=True)
mark(0, pt, (0, 0, 255))
if len(Tp) > 5:
    found["panel_top"].append(height_of(Tp, "扉の板の上端", 0))
Bp, pb = dark_line(0, [c for c in range(42, 176, 3) if not (108 <= c <= 128 or 152 <= c <= 172)], lambda c: 452 + (c - 40) * (435 - 452) / (180 - 40), below=False)
mark(0, pb, (0, 0, 255))
if len(Bp) > 5:
    found["panel_bottom"].append(height_of(Bp, "扉の板の下端", 0))

for v, items in overlay.items():
    im = Image.fromarray(imgs[v]).resize((imgs[v].shape[1] * 2, imgs[v].shape[0] * 2), Image.LANCZOS)
    d = ImageDraw.Draw(im)
    for pix, color in items:
        for x, y in pix:
            d.ellipse([(x + 0.5) * 2 - 3, (y + 0.5) * 2 - 3, (x + 0.5) * 2 + 3, (y + 0.5) * 2 + 3], outline=color, width=2)
    im.save(OUT / f"measured_view{v:02d}.jpg", quality=88)

values = {
    "door_width": np.mean(found["door_width"]) if found["door_width"] else None,
    "door_height": np.mean(found["door_height"]) if found["door_height"] else None,
    "closet_width": np.mean(found["closet_width"]) if found["closet_width"] else None,
    "closet_panel": (np.mean(found["panel_top"]) - np.mean(found["panel_bottom"])) if found["panel_top"] and found["panel_bottom"] else None,
}
summary = {"photos": len(imgs)}
print("== まとめ（復元 / 実測 / 差）")
for k, val in values.items():
    if val is None:
        print(f"  {k}: 測れず")
        summary[k] = None
        continue
    m = MEASURED[k]
    summary[k] = {"recon_cm": round(float(val), 1), "measured_cm": m, "diff_cm": round(float(val - m), 1), "pct": round(float(val / m - 1) * 100, 1)}
    print(f"  {k}: {val:.1f} / {m:.0f} / {val - m:+.1f} cm ({(val / m - 1) * 100:+.1f}%)")
ok = [v for v in summary.values() if isinstance(v, dict)]
if ok:
    summary["mean_abs_cm"] = round(float(np.mean([abs(v["diff_cm"]) for v in ok])), 2)
    s = [v["measured_cm"] / v["recon_cm"] for v in ok]
    summary["scale_factors"] = [round(x, 4) for x in s]
    print(f"  差の絶対値の平均 {summary['mean_abs_cm']} cm ｜ 実測÷復元: {[round(x, 3) for x in s]}")
(OUT / "measure_results.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
