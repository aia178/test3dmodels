"""実験（P3）: 写真の境目を、スイッチ・コンセントの上に通さない（保護領域）。

自宅の角（写真 13 枚）のドアの壁で、3 つの条件を比べる。
- 現行: 保護領域なし（正面から広く写した順に重ね、位置と明るさを合わせる）
- 1 枚固定: 保護領域を、その全体を写した 1 枚だけで塗る。境目は保護領域の縁にそのまま置く（位置合わせ・なじませなし）
- 保護＋つなぎ目: 同じ 1 枚で、保護領域の外の帯（3cm）まで塗り、周りに位置・明るさを合わせて帯の中で混ぜる

測るもの: 小物の中を通る写真の境目（画素数・写真の数）、被覆率、処理時間。見た目の幅と高さ（実物 約 7cm × 12cm、復元は約 4% 大きいので
7.3 × 12.5cm 前後が「正しい」）は、並べた画像の 1cm の目盛りで目で読む（板の縁は白い壁の上で弱く、自動で測ると内側の線を拾った）。並べた画像を experiments/out/seam_protect/ に書く。

使い方（server フォルダで）: uv run python ../experiments/seam_protect.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parent / "server"))
sys.path.insert(0, str(here))
from make_surfaces import load_views  # noqa: E402
from roomfit.textures import make_surfaces  # noqa: E402

OUT = here / "out" / "home_1012_avg"
DST = here / "out" / "seam_protect"
EDGE = 1   # ドアの壁
# 保護領域（辺の始点からの距離 左, 床からの高さ 上, 右, 下）[m]。現行の壁の画像を見て、開発者が指定した（少し広め）
OBJECTS = {"スイッチ": (0.27, 1.33, 0.35, 1.18), "コンセント": (0.215, 0.315, 0.31, 0.16)}
CONDITIONS = {"現行": None, "1枚固定": False, "保護+つなぎ目": True}
LABEL = {"現行": "A current", "1枚固定": "B single", "保護+つなぎ目": "C protect+seam"}   # 画像に書く名前（日本語の字体が無いため）


def to_px(box, L, hgt, shape):
    nv, nu = shape
    u0, h0, u1, h1 = box
    return int(u0 / L * nu), int((hgt - h0) / hgt * nv), int(np.ceil(u1 / L * nu)), int(np.ceil((hgt - h1) / hgt * nv))


def seams_inside(src, box):
    """箱の中で、隣の画素と塗った写真が違う所の数と、使われた写真の番号。"""
    x0, y0, x1, y1 = box
    s = src[y0:y1, x0:x1]
    n = int((s[:, 1:] != s[:, :-1]).sum() + (s[1:] != s[:-1]).sum())
    return n, sorted(set(np.unique(s).tolist()) - {-1})


def main():
    room, views = load_views(OUT, here / "data" / "home_1012" / "photos")
    O = np.asarray(room["outline"]); L = float(np.linalg.norm(O[(EDGE + 1) % len(O)] - O[EDGE])); hgt = room["ceiling_height"]
    DST.mkdir(parents=True, exist_ok=True)
    results, crops = {}, {}
    for name, band in CONDITIONS.items():
        protect = None if band is None else {EDGE: list(OBJECTS.values())}
        t = time.time()
        s = make_surfaces(room["outline"], room["edge_status"], hgt, views, protect=protect, seam_band=bool(band), details=True)
        dt = time.time() - t
        wall = next(w for w in s["walls"] if w["edge"] == EDGE)
        img, src, res = wall["image"], wall["src"], wall["res"]
        r = {"時間_秒": round(dt, 1), "被覆率": wall["coverage"], "塗った写真": wall.get("protected"), "小物": {}}
        for obj, box_m in OBJECTS.items():
            box = to_px(box_m, L, hgt, img.shape[:2])
            # 小物の中を通る境目: 箱（いまの画像での位置）と、保護したときは塗り直した領域（各写真での位置をすべて含む）の中で数える
            reg = next((pr["region"] for pr in wall.get("protected", []) if pr["box"] == list(box) and "region" in pr), None)
            look = box if reg is None else (min(box[0], reg[0]), min(box[1], reg[1]), max(box[2], reg[2]), max(box[3], reg[3]))
            n, used = seams_inside(src, look)
            r["小物"][obj] = {"境目の画素": n, "使った写真": used}
            crops.setdefault(obj, []).append((name, img, src, box))
        results[name] = r
        Image.fromarray(img).save(DST / f"wall_{EDGE}_{name}.jpg", quality=90)
        print(name, json.dumps(r, ensure_ascii=False))
    (DST / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    # 並べた画像: 条件ごとに、上が壁の画像、下が「塗った写真」の境目（赤）を重ねたもの
    for obj, items in crops.items():
        tiles = []
        for name, img, src, (x0, y0, x1, y1) in items:
            m = 45
            a, b, c, d = max(0, x0 - m), max(0, y0 - m), x1 + m, y1 + m
            tile = img[b:d, a:c].copy()
            s = src[b:d, a:c]
            edge = np.zeros(s.shape, bool)
            edge[:, 1:] |= s[:, 1:] != s[:, :-1]; edge[1:] |= s[1:] != s[:-1]
            over = tile.copy(); over[edge] = (255, 0, 0)
            pair = np.concatenate([tile, over], 0)
            pim = Image.fromarray(pair).resize((pair.shape[1] * 3, pair.shape[0] * 3), Image.NEAREST)
            # 1cm ごとの目盛り（上の画像の左端と上端）。見た目の幅・高さは、これで目で読む
            dr = ImageDraw.Draw(pim)
            step = 0.01 / res * 3
            for k in range(int(pim.width / step) + 1):
                dr.line([(k * step, 0), (k * step, 8 if k % 5 else 16)], fill=(0, 120, 255))
            for k in range(int(tile.shape[0] * 3 / step) + 1):
                dr.line([(0, k * step), (8 if k % 5 else 16, k * step)], fill=(0, 120, 255))
            dr.text((20, 4), LABEL[name], fill=(255, 0, 0))
            tiles.append(np.asarray(pim))
        hmax = max(t.shape[0] for t in tiles)
        row = np.concatenate([np.pad(t, ((0, hmax - t.shape[0]), (0, 12), (0, 0)), constant_values=255) for t in tiles], 1)
        Image.fromarray(row).save(DST / f"compare_{obj}.png")


if __name__ == "__main__":
    main()
