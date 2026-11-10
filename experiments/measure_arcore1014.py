"""評価: 2026-10-14 の ARCore の記録（クローゼット）から復元した結果で、扉 1 枚の幅を測る。

測る場所（写真の中の位置は、人が画像を見ておおよそ決めた。正確な端は明るさの境目から自動で探す）:
  - 左の扉: 左の枠の内側 〜 中央の合わせ目（コマ 36）
  - 右の扉: 中央の合わせ目 〜 右の端（コマ 37）
実測: クローゼットの幅は約 102cm（2026-10-12 にメジャーで測った値。扉 2 枚の合計にあたる）。扉 1 枚は約 50cm のはずだが、1 枚ずつは測っていない。
     なので「2 枚の合計が 102cm 前後か」を主に見る。±2cm 程度のあいまいさがある。

使い方（server フォルダで）: uv run python ../experiments/measure_arcore1014.py <出力フォルダ> [...]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
from roomfit.reconstruct import load_raw  # noqa: E402
from roomfit.worker import view_grids  # noqa: E402

# (元のコマ番号, 名前, 左端の x の割合, 右端の x の割合, 測る高さの範囲 y の割合, 内側はどちら向きか)
SPANS = [
    (36, "左の扉（枠の内側〜合わせ目）", 0.175, 0.615, (0.45, 0.75)),
    (37, "右の扉（合わせ目〜右端）", 0.345, 0.775, (0.40, 0.60)),
]


def find_edge(lum_row: np.ndarray, x0: int, reach: int) -> int:
    lo, hi = max(2, x0 - reach), min(len(lum_row) - 3, x0 + reach)
    g = np.abs(lum_row[lo + 2:hi + 2] - lum_row[lo - 2:hi - 2])
    return lo + int(np.argmax(g))


def measure(out: Path) -> None:
    raw = load_raw(out / "raw.npz")
    fmap = json.loads((out / "framemap.json").read_text())
    grids = view_grids(raw)
    total, parts = 0.0, []
    for old, name, fa, fb, (y0, y1) in SPANS:
        v = fmap[str(old)]
        mask, xyz = grids[v]
        h, w = mask.shape
        lum = raw["view_images"][v].astype(np.float64) @ [0.299, 0.587, 0.114]
        widths = []
        for y in range(int(y0 * h), int(y1 * h), 3):
            row = lum[y - 1:y + 2].mean(0)
            xa, xb = find_edge(row, int(fa * w), int(0.04 * w)), find_edge(row, int(fb * w), int(0.04 * w))
            pa = [xyz[y, x] for x in range(xa + 3, xa + 8) if mask[y, x]]          # 端の内側（扉の面）の点
            pb = [xyz[y, x] for x in range(xb - 7, xb - 2) if mask[y, x]]
            if len(pa) >= 3 and len(pb) >= 3:
                a, b = np.median(pa, axis=0), np.median(pb, axis=0)
                # 端までの数画素ぶんを、面に沿って外挿して足す
                per_px = np.linalg.norm(b - a) / max(1, (xb - 5) - (xa + 5))
                widths.append(np.linalg.norm(b - a) + per_px * 10)
        if not widths:
            print(f"  {name}: 測れず")
            return
        wcm = float(np.median(widths)) * 100
        parts.append(wcm)
        total += wcm
        cam = raw["cam2world"][v][:3, 3]
        dist = float(np.median(np.linalg.norm(xyz[int(y0 * h):int(y1 * h)][mask[int(y0 * h):int(y1 * h)]] - cam, axis=1)))
        print(f"  {name}: {wcm:.1f}cm（{len(widths)} 行の中央値、行ごとのばらつき {np.std(widths) * 100:.1f}cm）  カメラからの距離 {dist:.2f}m")
    print(f"  → 扉 2 枚の合計 {total:.1f}cm（実測 約 102cm、差 {(total / 102 - 1) * 100:+.1f}%）")


for a in sys.argv[1:]:
    print(Path(a).name)
    measure(Path(a))
