"""有効画素の情報を保存していなかった古い復元結果(raw.npz)から、写真の上で測るためのデータ(views/)を作る。
新しい復元ではサーバーが自動で作るので不要。使い方（server フォルダで）:
  uv run python ../experiments/export_views_legacy.py <raw.npz・room.json・images/ のあるフォルダ>
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gridtools import model_input_images, recover_grid  # noqa: E402
from roomfit.reconstruct import load_raw  # noqa: E402
from roomfit.worker import export_views  # noqa: E402

out = Path(sys.argv[1])
raw = load_raw(out / "raw.npz")
imgs = np.load(out / "model_images.npy") if (out / "model_images.npy").exists() else np.stack(model_input_images(sorted((out / "images").glob("*.jpg"))))
T = np.array(json.loads((out / "room.json").read_text(encoding="utf-8"))["T_room_from_world"])
grids = []
for v in range(len(raw["cam2world"])):
    g = recover_grid(raw, v, imgs[v])
    grids.append((g["valid"], g["xyz"].astype(np.float32)))
meta = export_views(grids, imgs, T, out)
print(f"{out.name}: {len(meta)} views written")
