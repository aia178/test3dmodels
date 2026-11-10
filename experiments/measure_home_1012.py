"""評価: 自宅の一角（2026-10-12 撮影・写真 13 枚）の復元結果を、メジャーの実測と比べる。

これは評価用の台本で、アプリの処理ではない。「写真のどの端を測るか」は人が画像を見ておおよその位置を与え、
正確な端の位置は色の境目（木の色と白い壁、扉の上下の暗いすき間）から自動で求める。
3D の位置は、端の内側（物の側）の画素の 3D 点を直線で外挿して求める（端そのものの画素は奥行きが混ざるため）。
高さは、部屋全体の床の平面（structure.py が当てたもの）からの高さで測る。

縮尺は復元モデルの推定のまま（実測での補正なし）。つまり「利用者が何も測らない場合」の誤差を見ている。

写真を減らした復元（run_subset.py の出力）にも使える。その場合、元の 13 枚での番号 → 減らした後の番号の対応を
viewmap.json から読む。測る写真が含まれていない項目は飛ばす。

使い方（server フォルダで）: uv run python ../experiments/measure_home_1012.py [出力フォルダ] [縮尺]
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gridtools import model_input_images, recover_grid  # noqa: E402
from roomfit.reconstruct import load_raw  # noqa: E402

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "out" / "home_1012"
SCALE = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0  # 1.0 = モデルの推定のまま
MEASURED = {"door_width": 85.0, "door_height": 219.0, "closet_width": 102.0, "closet_panel": 226.0}  # cm（利用者の実測）

raw = load_raw(OUT / "raw.npz")
n_views = len(raw["cam2world"])
vm_path = OUT / "viewmap.json"
VIEWMAP = {int(k): v for k, v in json.loads(vm_path.read_text(encoding="utf-8")).items()} if vm_path.exists() else {i: i for i in range(n_views)}
if (OUT / "model_images.npy").exists():
    imgs = np.load(OUT / "model_images.npy")
else:
    imgs = np.stack(model_input_images(sorted((OUT / "images").glob("*.jpg"))))
    np.save(OUT / "model_images.npy", imgs)
if not (OUT / "room.json").exists():
    print("room.json がありません（形状抽出に失敗した復元です）。高さの基準が無いため測れません。")
    sys.exit(2)
room = json.loads((OUT / "room.json").read_text(encoding="utf-8"))
T = np.array(room["T_room_from_world"])

_grids = {}


def grid(v):
    """v は元の 13 枚での番号。"""
    if v not in _grids:
        s = VIEWMAP[v]
        g = recover_grid(raw, s, imgs[s])
        g["room"] = (g["xyz"] @ T[:3, :3].T + T[:3, 3]) * SCALE   # 部屋座標（y が床からの高さ）
        g["wood"] = imgs[s][..., 0].astype(float) - imgs[s][..., 2].astype(float)  # 木の色らしさ = 赤 - 青
        g["lum"] = imgs[s].astype(float) @ np.array([0.299, 0.587, 0.114])
        _grids[v] = g
    return _grids[v]


def crossing(profile: np.ndarray, rising: bool):
    """1 次元の並びで、低い側と高い側の中間を横切る位置（小数の添字）を返す。はっきりした境目が無ければ None。"""
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
    """端の位置 f（小数の添字）の 3D 点を、物の側の画素の 3D 点から直線で外挿する。"""
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
    """axis='row': 各行を左右に走査して縦の端を探す。axis='col': 各列を上下に走査して横の端を探す。
    p0,p1: 端のおおよその位置（モデル画像の画素 (x,y)）を結ぶ線分。hw: 探す幅の半分。
    inside: +1 なら端の添字が大きい側が物、-1 なら小さい側が物。戻り値: 端の 3D 点（部屋座標）と画素位置。"""
    g = grid(v)
    pts, pix = [], []
    for t in lines:
        if axis == "row":
            if not (min(p0[1], p1[1]) <= t <= max(p0[1], p1[1])):
                continue
            e = p0[0] + (p1[0] - p0[0]) * (t - p0[1]) / (p1[1] - p0[1])
            prof_full, xyz_full = g["wood"][t, :], g["room"][t, :, :]
        else:
            if not (min(p0[0], p1[0]) <= t <= max(p0[0], p1[0])):
                continue
            e = p0[1] + (p1[1] - p0[1]) * (t - p0[0]) / (p1[0] - p0[0])
            prof_full, xyz_full = g["wood"][:, t], g["room"][:, t, :]
        a, b = int(round(e - hw)), int(round(e + hw))
        if a < 0 or b >= len(prof_full):
            continue
        f = crossing(prof_full[a:b + 1], rising)
        if f is None:
            continue
        f += a
        p = extrapolate(xyz_full, f, inside, n_fit)
        if p is None:
            continue
        pts.append(p)
        pix.append((f, t) if axis == "row" else (t, f))
    return np.array(pts), np.array(pix)


def panel_top(v, cols, e_of_c):
    """クローゼットの枠の上端 → その下の暗いすき間 → 扉の板の上端、の順に探して、板の上端の 3D 点を返す。"""
    g = grid(v)
    out, pix = [], []
    for c in cols:
        e, prof = e_of_c(c), g["wood"][:, c]
        a = int(round(e - 7))
        if a < 0:
            continue
        f = crossing(prof[a:int(round(e + 7)) + 1], True)
        if f is None:
            continue
        f += a
        j0, j1 = int(np.ceil(f + 3)), int(np.floor(f + 14))
        m = j0 + int(np.argmin(prof[j0:j1 + 1]))
        if m <= int(np.ceil(f + 1)):
            continue
        panel, frame = np.median(prof[m + 4:m + 10]), np.max(prof[int(np.ceil(f + 1)):m])
        if panel - prof[m] < 20 or frame - prof[m] < 15:
            continue
        mid, k = (prof[m] + panel) / 2, m
        while k < m + 6 and prof[k + 1] < mid:
            k += 1
        fp = k + (mid - prof[k]) / (prof[k + 1] - prof[k])
        p = extrapolate(g["room"][:, c, :], fp, +1, (1.0, 6.0))
        if p is not None:
            out.append(p)
            pix.append((c, fp))
    return np.array(out), np.array(pix)


def panel_bottom(v, cols, e_of_c):
    """扉の板の下端（板と敷居の間の暗い線の上側）の 3D 点を返す。"""
    g = grid(v)
    out, pix = [], []
    for c in cols:
        e = e_of_c(c)
        a, b = int(round(e - 4)), int(round(e + 4))
        seg = g["lum"][a:b + 1, c]
        m = int(np.argmin(seg))
        if m in (0, len(seg) - 1) or np.median(seg) - seg[m] < 12:
            continue
        y0, y1, y2 = seg[m - 1], seg[m], seg[m + 1]
        fb = a + m + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2) - 0.5
        p = extrapolate(g["room"][:, c, :], fb, -1, (1.5, 7.0))
        if p is not None:
            out.append(p)
            pix.append((c, fb))
    return np.array(out), np.array(pix)


def line_fit(P):
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c)
    d = vt[0] if vt[0][1] >= 0 else -vt[0]
    return c, d


def at_height(c, d, y):
    return c + d * (y - c[1]) / d[1]


SEGMENTS = {}   # アプリの「測定」として表示するための、両端の部屋座標（x, 床からの高さ, z）


def width_between(PL, PR, name, key=None):
    cl, dl = line_fit(PL)
    cr, dr = line_fit(PR)
    if key:
        SEGMENTS.setdefault(key, []).append((at_height(cl, dl, 1.0).tolist(), at_height(cr, dr, 1.0).tolist()))
    for side, P, c, d in (("左", PL, cl, dl), ("右", PR, cr, dr)):   # 端がまっすぐか（直線からのずれ）を、高さごとに見る
        off = P - (c + np.outer((P - c) @ d, d))
        dev = np.linalg.norm(off[:, [0, 2]], axis=1) * np.sign(off[:, 0] + off[:, 2])
        order = np.argsort(P[:, 1])
        bands = np.array_split(order, 5)
        print(f"    {side}の端の、直線からのずれ（下→上、cm）: " + " ".join(f"{np.median(dev[b_]) * 100:+.1f}" for b_ in bands) + f"  最大 {np.abs(dev).max() * 100:.1f}")
    w = [float(np.linalg.norm((at_height(cl, dl, y) - at_height(cr, dr, y))[[0, 2]])) for y in (0.4, 1.0, 1.6)]
    tilt = [float(np.degrees(np.arccos(min(1.0, abs(d[1]))))) for d in (dl, dr)]
    print(f"  {name}: 幅 {w[1]*100:.1f} cm（高さ 40cm で {w[0]*100:.1f} / 160cm で {w[2]*100:.1f}） ｜ 端の傾き 左 {tilt[0]:.1f}° 右 {tilt[1]:.1f}° ｜ 点数 {len(PL)}/{len(PR)}")
    return w[1] * 100


def height_of(P, name, key=None):
    y = P[:, 1]
    if key:
        c = np.median(P, axis=0)
        SEGMENTS.setdefault(key, []).append(c.tolist())
    print(f"  {name}: 高さ 中央値 {np.median(y)*100:.1f} cm（{np.percentile(y, 10)*100:.1f}〜{np.percentile(y, 90)*100:.1f}） ｜ 点数 {len(P)}")
    return float(np.median(y)) * 100


overlay = {}
found = {"door_width": [], "door_height": [], "closet_width": [], "panel_top": [], "panel_bottom": []}


def mark(v, pix, color):
    if len(pix):
        overlay.setdefault(v, []).append((pix, color))


print(f"フォルダ: {OUT.name} ｜ 写真 {n_views} 枚 ｜ 縮尺: モデルの推定 × {SCALE}")
ROWS = range(60, 471, 5)
if 12 in VIEWMAP:
    print("== 引き戸（写真 12: 正面）")
    L, pl = edge_points(12, "row", ROWS, (14, 40), (39, 480), 8, True, +1, (1.0, 5.0))
    R, pr = edge_points(12, "row", ROWS, (217, 35), (212.5, 495), 8, False, -1)
    mark(12, pl, (255, 0, 0)); mark(12, pr, (255, 0, 0))
    if len(L) > 10 and len(R) > 10:
        found["door_width"].append(width_between(L, R, "幅（左の縦枠の外側〜戸の右端）", "door_width"))
    Tp, pt = edge_points(12, "col", range(30, 381, 5), (20, 15), (390, 12), 7, True, +1)
    mark(12, pt, (0, 0, 255))
    if len(Tp) > 10:
        found["door_height"].append(height_of(Tp, "上枠の上端", "door_top"))
if 5 in VIEWMAP:
    print("== 引き戸（写真 5: 斜め）")
    L, pl = edge_points(5, "row", range(70, 481, 5), (59, 60), (56, 490), 8, True, +1, (1.0, 5.0))
    R, pr = edge_points(5, "row", range(70, 481, 5), (256.5, 40), (227.5, 505), 8, False, -1)
    mark(5, pl, (255, 0, 0)); mark(5, pr, (255, 0, 0))
    if len(L) > 10 and len(R) > 10:
        found["door_width"].append(width_between(L, R, "幅（左の縦枠の外側〜戸の右端）"))
    Tp, pt = edge_points(5, "col", range(70, 361, 5), (56.5, 44), (392, 2.5), 7, True, +1)
    mark(5, pt, (0, 0, 255))
    if len(Tp) > 10:
        found["door_height"].append(height_of(Tp, "上枠の上端"))
if 0 in VIEWMAP:
    print("== クローゼット（写真 0: 斜め）")
    L, pl = edge_points(0, "row", range(80, 441, 5), (20, 50), (23.5, 500), 7, True, +1, (1.0, 5.0))
    R, pr = edge_points(0, "row", range(80, 441, 5), (248.5, 60), (231, 475), 8, False, -1, (1.0, 5.0))
    mark(0, pl, (255, 0, 0)); mark(0, pr, (255, 0, 0))
    if len(L) > 10 and len(R) > 10:
        found["closet_width"].append(width_between(L, R, "幅（枠の外側〜外側）", "closet_width"))
    Tp, pt = panel_top(0, range(150, 241, 3), lambda c: (c - 50) * 56.5 / 198)
    mark(0, pt, (0, 0, 255))
    if len(Tp) > 5:
        found["panel_top"].append(height_of(Tp, "扉の板の上端"))
if 11 in VIEWMAP:
    print("== クローゼット（写真 11: 正面寄り）")
    Tp, pt = panel_top(11, range(60, 231, 3), lambda c: 1.5 + (c - 15) * 44 / 225)
    mark(11, pt, (0, 0, 255))
    if len(Tp) > 5:
        found["panel_top"].append(height_of(Tp, "扉の板の上端", "panel_top"))
    Bp, pb = panel_bottom(11, [c for c in range(20, 201, 4) if not (106 <= c <= 122)], lambda c: 497 + (c - 10) * (471 - 497) / (207 - 10))
    mark(11, pb, (0, 0, 255))
    if len(Bp) > 5:
        found["panel_bottom"].append(height_of(Bp, "扉の板の下端", "panel_bottom"))

for v, items in overlay.items():
    im = Image.fromarray(imgs[VIEWMAP[v]]).resize((392 * 2, 518 * 2), Image.LANCZOS)
    d = ImageDraw.Draw(im)
    for pix, color in items:
        for x, y in pix:
            d.ellipse([(x + 0.5) * 2 - 3, (y + 0.5) * 2 - 3, (x + 0.5) * 2 + 3, (y + 0.5) * 2 + 3], outline=color, width=2)
    im.save(OUT / f"measured_view{v:02d}.jpg", quality=88)

summary = {"photos": n_views, "ceiling_cm": None if room["ceiling_height"] is None else round(room["ceiling_height"] * 100 * SCALE, 1)}
values = {
    "door_width": np.mean(found["door_width"]) if found["door_width"] else None,
    "door_height": np.mean(found["door_height"]) if found["door_height"] else None,
    "closet_width": np.mean(found["closet_width"]) if found["closet_width"] else None,
    "closet_panel": (np.mean(found["panel_top"]) - np.mean(found["panel_bottom"])) if found["panel_top"] and found["panel_bottom"] else None,
}
print("== まとめ（復元 / 実測 / 差）")
for k, val in values.items():
    if val is None:
        print(f"  {k}: 測れず")
        summary[k] = None
        continue
    m = MEASURED[k]
    summary[k] = {"recon_cm": round(float(val), 1), "measured_cm": m, "diff_cm": round(float(val - m), 1), "pct": round(float(val / m - 1) * 100, 1)}
    print(f"  {k}: {val:.1f} / {m:.0f} / {val - m:+.1f} cm ({(val / m - 1) * 100:+.1f}%)")
ok = [v for k, v in summary.items() if isinstance(v, dict)]
if len(ok) >= 2:
    s = [v["measured_cm"] / v["recon_cm"] for v in ok]
    summary["scale_factors"] = [round(x, 4) for x in s]
    summary["scale_spread_pct"] = round((max(s) - min(s)) / np.mean(s) * 100, 2)
    print(f"  実測÷復元（縮尺のずれ）: {[round(x, 3) for x in s]} ｜ ばらつき {summary['scale_spread_pct']}%（小さいほど形が正確）")
(OUT / "measure_segments.json").write_text(json.dumps(SEGMENTS), encoding="utf-8")
(OUT / "measure_results.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
