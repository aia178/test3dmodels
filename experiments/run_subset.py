"""実験: 写真の枚数を減らしたら復元はどうなるか。

同じ撮影（13 枚）から一部だけを選んで復元し、枚数ごとの結果を比べる。
「内見で何十枚も撮れない」という懸念に対して、何枚あれば足りるかを実データで確かめるためのもの。

使い方（server フォルダで）:
  uv run python ../experiments/run_subset.py <元の出力フォルダ(images/ がある)> <出力フォルダ> <使う写真の番号 例: 0,5,11,12>
"""
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
from roomfit.reconstruct import run_mapanything, save_raw  # noqa: E402
from roomfit.structure import StructureError  # noqa: E402
from roomfit.worker import structure_from_raw  # noqa: E402

src, out = Path(sys.argv[1]), Path(sys.argv[2])
idx = [int(x) for x in sys.argv[3].split(",")]
f35 = float(sys.argv[4]) if len(sys.argv) > 4 else None  # 35mm 換算の焦点距離を与える場合
(out / "images").mkdir(parents=True, exist_ok=True)
paths = []
for new, old in enumerate(idx):
    p = out / "images" / f"{new:03d}.jpg"
    shutil.copyfile(src / "images" / f"{old:03d}.jpg", p)
    paths.append(p)
(out / "viewmap.json").write_text(json.dumps({str(old): new for new, old in enumerate(idx)}), encoding="utf-8")
t0 = time.time()
raw = run_mapanything(paths, focal_35mm=f35)
raw["meta"]["sec_total"] = round(time.time() - t0, 1)
save_raw(raw, out / "raw.npz")
(out / "meta.json").write_text(json.dumps(raw["meta"], indent=1), encoding="utf-8")
try:
    room = structure_from_raw(raw, out)
    print(f"{len(idx)} photos {idx}: inference {raw['meta']['sec_inference']}s, ceiling {room['ceiling_height']}, edges {room['edge_status']}", flush=True)
except StructureError as e:
    print(f"{len(idx)} photos {idx}: inference {raw['meta']['sec_inference']}s, STRUCTURE FAILED: {e.code} {e.message}", flush=True)
