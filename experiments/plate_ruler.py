"""スイッチ・コンセントの枠（日本で一般的な 1 口の枠は縦 120mm × 横 70mm）を物差しにして、復元の大きさのずれを見積もる（再推論なし）。

枠は 2m 先だとモデルに入る画像（縦 518 画素）では小さすぎるので、元の大きな写真で輪郭を拾う。
1. 元の写真の枠のまわりを切り出し、輪郭（Canny）から一番大きい長方形を枠とみなす。各辺を、辺に垂直な明るさの変化が最大の位置で
   画素以下まで合わせ、直線を当てて 4 つの角を出す（角の丸みは使わない）。
2. 元の写真の画素 → モデルに入った画像の画素（preprocess_inputs の縮小・切り抜き）→ モデルのカメラ座標の視線。
   視線の向きは、そのコマのモデルの点（カメラ座標）に当てた内部パラメータで決める（モデル自身の点と同じ幾何になる）。
3. 枠のまわりの壁の点（モデルの点）に平面を当て、4 つの角の視線と交わる点を出す。縦と横の長さがモデルの単位での枠の大きさ。
規格の大きさ ÷ モデルでの大きさ が、物差しから見た「実寸 ÷ 復元」。メジャーの 4 か所での値（10/12 は 0.958〜0.964、10/21 は 0.958〜0.965）と比べる。

使い方（server フォルダで）: uv run python ../experiments/plate_ruler.py
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parent / "server"))
from roomfit.ar_views import _rotated  # noqa: E402
from roomfit.reconstruct import MAX_SIDE, load_raw  # noqa: E402
from roomfit.worker import view_grids  # noqa: E402

PLATE_H, PLATE_W = 0.120, 0.070
OUT = here / "ext" / "plate_ruler"
# 外側の輪郭を正しく拾えたと絵で確かめたもの（2026-10-21 の 1 回目の実行。内側のスイッチの縁や斜めの箱を拾ったものは除く）
CLEAN = {"写真 1 のswitch", "写真 2 のswitch", "写真 10 のswitch", "写真 11 のswitch", "写真 5 のoutlet", "コマ 011 のswitch", "コマ 009 のoutlet"}
TAPE = {"home_1012": (0.958, 0.964), "home_n4": None, "home_n6": None, "home_n8": None, "home_1021_ar": (0.958, 0.965)}


def preprocess_map(w, h):
    """w×h の画像がモデルに入るときの、画素の写し方 (sx, tx, sy, ty)。内部パラメータの変わり方から求める。"""
    from mapanything.utils.image import preprocess_inputs

    K = np.array([[1000.0, 0, w / 2], [0, 1000.0, h / 2], [0, 0, 1]])
    v = preprocess_inputs([{"img": torch.zeros((h, w, 3), dtype=torch.uint8), "intrinsics": torch.tensor(K, dtype=torch.float32)}])[0]
    Kp = v["intrinsics"][0].numpy().astype(np.float64)
    sx, sy = Kp[0, 0] / K[0, 0], Kp[1, 1] / K[1, 1]
    return sx, Kp[0, 2] - sx * K[0, 2], sy, Kp[1, 2] - sy * K[1, 2], tuple(v["img"].shape[-2:])


def fit_K(xyz, mask):
    """カメラ座標の点の格子から、モデルが使った内部パラメータを当てる（u = fx x/z + cx）。"""
    v, u = np.nonzero(mask & np.isfinite(xyz[..., 2]) & (xyz[..., 2] > 0.05))
    P = xyz[v, u].astype(np.float64)
    a, b = P[:, 0] / P[:, 2], P[:, 1] / P[:, 2]
    fx, cx = np.linalg.lstsq(np.c_[a, np.ones_like(a)], u + 0.5, rcond=None)[0]
    fy, cy = np.linalg.lstsq(np.c_[b, np.ones_like(b)], v + 0.5, rcond=None)[0]
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])


def detect_plate(gray, guess, half=200, edge="outer"):
    """guess のまわりで枠の 4 つの角（元の写真の画素、左上・右上・右下・左下）を返す。
    edge="outer": 辺に垂直な明るさの変化が最大の所（縁の暗い線の外側になりやすい）。"inner": 暗い線の内側、枠の面が始まる所。"""
    x0, y0 = int(guess[0]) - half, int(guess[1]) - half
    if x0 < 0 or y0 < 0 or x0 + 2 * half > gray.shape[1] or y0 + 2 * half > gray.shape[0]:
        return None
    crop = gray[y0:y0 + 2 * half, x0:x0 + 2 * half].astype(np.float32)
    blur = cv2.GaussianBlur(crop, (0, 0), 1.5)
    edges = cv2.Canny(blur.astype(np.uint8), 10, 30)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    best = None
    for c in cnts:
        (cx, cy), (w, h), ang = cv2.minAreaRect(c)
        if min(w, h) < 15 or abs(cx - half) > half * 0.7 or abs(cy - half) > half * 0.7:
            continue
        hh, ww = max(w, h), min(w, h)
        if not 1.3 < hh / ww < 3.5 or cv2.contourArea(cv2.convexHull(c)) < 0.8 * w * h:
            continue
        if best is None or w * h > best[1]:
            best = (c, w * h)
    if best is None:
        return None
    box = cv2.boxPoints(cv2.minAreaRect(best[0]))
    box = box[np.argsort(box[:, 1])]
    top, bot = box[:2][np.argsort(box[:2, 0])], box[2:][np.argsort(box[2:, 0])]
    rough = np.array([top[0], top[1], bot[1], bot[0]])   # 左上・右上・右下・左下
    gy, gx = np.gradient(blur)
    lines = []
    for side in range(4):
        p, q = rough[side], rough[(side + 1) % 4]
        d = (q - p) / np.linalg.norm(q - p)
        nrm = np.array([-d[1], d[0]])   # 枠の内側を向く
        pts = []
        for t in np.linspace(0.2, 0.8, 40):
            c0 = p + t * (q - p)
            offs = np.arange(-6, 6.01, 0.5)
            samp = (c0[None] + offs[:, None] * nrm[None]).astype(np.float32)
            g = np.abs(cv2.remap(gx, samp[:, 0][None], samp[:, 1][None], cv2.INTER_LINEAR)[0] * nrm[0]
                       + cv2.remap(gy, samp[:, 0][None], samp[:, 1][None], cv2.INTER_LINEAR)[0] * nrm[1])
            i = int(np.argmax(g))
            if not 0 < i < len(g) - 1:
                continue
            den = g[i - 1] - 2 * g[i] + g[i + 1]
            off = offs[i] + (0.5 * (g[i - 1] - g[i + 1]) / den * 0.5 if den != 0 else 0)
            if edge == "inner":
                # 縁の暗い線（影か、枠の角の面）の内側、枠の面が始まる所
                o2 = np.arange(off - 2, off + 8.01, 0.25)
                s2 = (c0[None] + o2[:, None] * nrm[None]).astype(np.float32)
                lum = cv2.remap(blur, s2[:, 0][None], s2[:, 1][None], cv2.INTER_LINEAR)[0]
                m = int(np.argmin(lum[:24]))
                face = np.median(lum[m + 12:m + 24]) if m + 24 <= len(lum) else np.median(lum[-8:])
                mid = (lum[m] + face) / 2
                j = m
                while j < len(lum) - 1 and lum[j] < mid:
                    j += 1
                if face - lum[m] >= 4 and 0 < j < len(lum) - 1:   # 暗い線が無い辺（上から光が当たる上の辺など）は外側のまま
                    off = o2[j - 1] + 0.25 * (mid - lum[j - 1]) / (lum[j] - lum[j - 1])
            pts.append(c0 + off * nrm)
        if len(pts) < 5:
            return None
        pts = np.array(pts)
        c = pts.mean(0)
        dd = np.linalg.svd(pts - c)[2][0]
        lines.append((c, dd))
    corners = []
    for k in range(4):
        (c1, d1), (c2, d2) = lines[k - 1], lines[k]
        t = np.linalg.solve(np.c_[d1, -d2], c2 - c1)
        corners.append(c1 + t[0] * d1)
    return np.array(corners) + np.array([x0, y0])


def plate_size(corners_full, to_model, xyz_cam, mask, K):
    q = np.c_[corners_full[:, 0] * to_model[0] + to_model[1], corners_full[:, 1] * to_model[2] + to_model[3]]
    rays = np.c_[(q[:, 0] - K[0, 2]) / K[0, 0], (q[:, 1] - K[1, 2]) / K[1, 1], np.ones(4)]
    lo, hi = q.min(0), q.max(0)
    size = hi - lo
    H, W = mask.shape
    vv, uu = np.mgrid[0:H, 0:W]
    near = (uu + 0.5 > lo[0] - 1.5 * size[0]) & (uu + 0.5 < hi[0] + 1.5 * size[0]) & (vv + 0.5 > lo[1] - 0.6 * size[1]) & (vv + 0.5 < hi[1] + 0.6 * size[1])
    inside = (uu + 0.5 > lo[0] - 1) & (uu + 0.5 < hi[0] + 1) & (vv + 0.5 > lo[1] - 1) & (vv + 0.5 < hi[1] + 1)
    P = xyz_cam[near & ~inside & mask].astype(np.float64)
    P = P[np.isfinite(P).all(1)]
    for _ in range(3):   # 枠の近くの家具の縁などを外す
        c = P.mean(0)
        n = np.linalg.svd(P - c)[2][2]
        r = np.abs((P - c) @ n)
        P = P[r < max(np.percentile(r, 80), 0.003)]
    c = P.mean(0)
    n = np.linalg.svd(P - c)[2][2]
    X = rays * ((c @ n) / (rays @ n))[:, None]
    h = (np.linalg.norm(X[0] - X[3]) + np.linalg.norm(X[1] - X[2])) / 2
    w = (np.linalg.norm(X[0] - X[1]) + np.linalg.norm(X[3] - X[2])) / 2
    tilt = np.degrees(np.arccos(abs(n @ np.array([0, 0, 1.0]))))
    return h, w, float(np.median(X[:, 2])), tilt, len(P)


def cases():
    """(名前, 写真, 枠のおおよその中心（元の写真の画素）, モデルのカメラ座標の点の格子とマスク, 元の写真 → モデル画像の写し方)。"""
    # 10/12: 写真だけの復元（13 枚の experiments/out/home_1012_avg と、4・6・8 枚に減らした復元）。点の格子は scale_cases.py が保存したもの。
    photos = sorted((here / "data" / "home_1012" / "photos").glob("*.jpg"))
    sx, tx, sy, ty, _ = preprocess_map(768, 1024)
    to_model = (sx / 4, tx, sy / 4, ty)   # 元の写真 3072×4096 → 768×1024 → モデル
    guesses = {
        0: [("switch", (2406, 2232)), ("outlet", (2273, 3655))], 1: [("switch", (1781, 2201))],
        2: [("switch", (1505, 2201)), ("outlet", (1433, 3584))], 3: [("switch", (1177, 2181)), ("outlet", (1136, 3563))],
        4: [("switch", (880, 2170)), ("outlet", (860, 3584))], 5: [("switch", (337, 2129)), ("outlet", (307, 3665))],
        10: [("switch", (1710, 1935)), ("outlet", (1710, 3174))], 11: [("switch", (2242, 2242))],
    }
    for ds, case, vm in (("home_1012", "home_avg", None), ("home_n4", "home_n4", "home_1012_n4"), ("home_n6", "home_n6", "home_1012_n6"), ("home_n8", "home_n8", "home_1012_n8")):
        c = np.load(here / "ext" / "scale_bench" / "cases" / f"{case}.npz")
        viewmap = {int(k): v for k, v in json.loads((here / "out" / vm / "viewmap.json").read_text(encoding="utf-8")).items()} if vm else {v: v for v in range(13)}
        for v, items in guesses.items():
            if v not in viewmap:
                continue
            for kind, g in items:
                yield ds, f"写真 {v} の{kind}", photos[v], g, c[f"ma{viewmap[v]}"].astype(np.float64), c[f"mask{viewmap[v]}"], to_model
    # 10/21: AR の位置つきの復元
    job = here / "ext" / "home_1021_ar" / "100455"
    raw = load_raw(job / "raw.npz")
    grids = view_grids(raw)
    ids = raw["meta"]["ar_frames"]["ids"]
    for v, items in {5: [("switch", (52, 879)), ("outlet", (75, 1590))], 6: [("switch", (825, 840)), ("outlet", (810, 1545))]}.items():
        f = json.loads((job / "capture" / f"{ids[v]}.json").read_text(encoding="utf-8"))
        K, _ = _rotated(f)
        W, H = f["h"], f["w"]   # 縦向きに回した後
        s = min(1.0, MAX_SIDE / max(W, H))
        tw, th = round(W * s), round(H * s)
        sx, tx, sy, ty, _ = preprocess_map(tw, th)
        T = raw["cam2world"][v]
        mask, xyz = grids[v]
        xyz_cam = (xyz.astype(np.float64) - T[:3, 3]) @ T[:3, :3]
        yield "home_1021_ar", f"コマ {ids[v]}", job / "capture" / f"{ids[v]}.jpg", None, xyz_cam, mask, (sx * tw / W, tx, sy * th / H, ty), items


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in cases():
        if item[0] == "home_1021_ar":
            ds, label, path, _, xyz, mask, to_model, items = item
            todo = [(f"{label} の{k}", g) for k, g in items]
            rot = True
        else:
            ds, label, path, g, xyz, mask, to_model = item
            todo, rot = [(label, g)], False
        with Image.open(path) as im:
            im = im.convert("RGB").rotate(-90, expand=True) if rot else ImageOps.exif_transpose(im).convert("RGB")
            rgb = np.asarray(im)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        K = fit_K(xyz, mask)
        for name, g in todo:
            half = 70 if ds == "home_1021_ar" else 200
            for edge in ("outer", "inner"):
                corners = detect_plate(gray, g, half, edge)
                if corners is None:
                    print(f"{ds} {name} {edge}: 枠を見つけられず", flush=True)
                    break
                h, w, z, tilt, npl = plate_size(corners, to_model, xyz, mask, K)
                px_h = (np.linalg.norm(corners[0] - corners[3]) + np.linalg.norm(corners[1] - corners[2])) / 2
                row = {"ds": ds, "name": name, "edge": edge, "clean": name in CLEAN, "h_mm": h * 1000, "w_mm": w * 1000, "s_h": PLATE_H / h, "s_w": PLATE_W / w,
                       "z_m": z, "wall_tilt_deg": tilt, "px_h": px_h}
                rows.append(row)
                print(f"{ds} {name} {edge}{'' if row['clean'] else '（輪郭の拾い違い）'}: 縦 {h * 1000:.1f} 横 {w * 1000:.1f}mm（写真で縦 {px_h:.0f} 画素、奥行き {z:.2f}m、壁の傾き {tilt:.0f}°）"
                      f" ｜ 規格÷復元 縦 {PLATE_H / h:.3f} 横 {PLATE_W / w:.3f}", flush=True)
                x0, y0 = (np.array(g) - (half + 40)).astype(int)
                crop = Image.fromarray(rgb[max(y0, 0):y0 + 2 * half + 80, max(x0, 0):x0 + 2 * half + 80].copy())
                d = ImageDraw.Draw(crop)
                pts = [tuple(p - np.array([max(x0, 0), max(y0, 0)])) for p in corners]
                d.line(pts + [pts[0]], fill=(255, 0, 0), width=1)
                crop.save(OUT / f"{ds}_{name.replace(' ', '_')}_{edge}.png")
    for ds in TAPE:
        for edge in ("outer", "inner"):
            sub = [r for r in rows if r["ds"] == ds and r["edge"] == edge and r["clean"]]
            auto = [r for r in rows if r["ds"] == ds and r["edge"] == edge and r["wall_tilt_deg"] < 70 and abs(r["h_mm"] / r["w_mm"] / (PLATE_H / PLATE_W) - 1) < 0.12]
            if sub or auto:
                print(f"{ds} {edge}: 規格÷復元 縦 正しく拾えた {len(sub)} 個で {np.median([r['s_h'] for r in sub]) if sub else np.nan:.3f}"
                      f" ／ 自動の規則で残った {len(auto)} 個で {np.median([r['s_h'] for r in auto]) if auto else np.nan:.3f}")
    (OUT / "results.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
