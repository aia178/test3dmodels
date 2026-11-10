"""実験: 撮影アプリ（ARCore）が保存した連続撮影の写真だけを使って復元する（姿勢・深度は使わない）。
標準カメラとは別の撮り方（動かしながら・横長 1080p・近距離）でも、同じ実測と合うかを見るためのもの。
使い方（server フォルダで）: uv run python ../experiments/run_arcore_frames.py <記録フォルダ> <出力フォルダ> <番号 例: 0,4,6>
"""
import json
import sys
import time
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
from roomfit.reconstruct import run_mapanything, save_raw  # noqa: E402
from roomfit.structure import StructureError  # noqa: E402
from roomfit.worker import structure_from_raw  # noqa: E402

src, out = Path(sys.argv[1]), Path(sys.argv[2])
idx = [int(x) for x in sys.argv[3].split(",")]
(out / "images").mkdir(parents=True, exist_ok=True)
paths = []
for new, old in enumerate(idx):
    with Image.open(src / f"{old:03d}.jpg") as im:
        im = im.convert("RGB").rotate(-90, expand=True)  # センサーの向き → 見たままの向き（縦持ち）
        im.thumbnail((1024, 1024), Image.LANCZOS)
        p = out / "images" / f"{new:03d}.jpg"
        im.save(p, "JPEG", quality=92)
        paths.append(p)
(out / "framemap.json").write_text(json.dumps({str(old): new for new, old in enumerate(idx)}), encoding="utf-8")
t0 = time.time()
raw = run_mapanything(paths)
raw["meta"]["sec_total"] = round(time.time() - t0, 1)
save_raw(raw, out / "raw.npz")
try:
    room = structure_from_raw(raw, out)
    print(f"{len(idx)} frames: inference {raw['meta']['sec_inference']}s, image_hw {raw['image_hw'][0].tolist()}, ceiling {room['ceiling_height']}, corner {room['diagnostics'].get('wall_corner_angle_deg')}, views {len(room['views'])}", flush=True)
except StructureError as e:
    print(f"{len(idx)} frames: STRUCTURE FAILED {e.code} {e.message}", flush=True)
