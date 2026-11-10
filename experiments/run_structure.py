"""実験: raw.npz → 部屋形状(room.json) と、確認用の上面図 PNG。

使い方（server フォルダで）:
  uv run python ../experiments/run_structure.py <raw.npz のあるフォルダ>
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
from roomfit.reconstruct import load_raw  # noqa: E402
from roomfit.structure import StructureError  # noqa: E402
from roomfit.worker import structure_from_raw  # noqa: E402

d = Path(sys.argv[1])
raw = load_raw(d / "raw.npz")
print("points", raw["points"].shape, "views", len(raw["cam2world"]), "conf range", float(raw["conf"].min()), float(raw["conf"].max()))
try:
    room = structure_from_raw(raw, d)
except StructureError as e:
    print("STRUCTURE FAILED:", e.code, e.message)
    sys.exit(2)

print(json.dumps({k: room[k] for k in ("outline", "edge_status", "edge_support", "ceiling_height", "ceiling_observed", "obstacles", "diagnostics")}, ensure_ascii=False, indent=1))

n = room["cloud_points"]
buf = (d / "cloud.bin").read_bytes()
pts = np.frombuffer(buf[: n * 12], dtype="<f4").reshape(n, 3)
ol = np.array(room["outline"])
lo = np.minimum(pts[:, [0, 2]].min(0), ol.min(0)) - 0.2
hi = np.maximum(pts[:, [0, 2]].max(0), ol.max(0)) + 0.2
S = 900 / float((hi - lo).max())
W, H = (np.ceil((hi - lo) * S)).astype(int)
im = Image.new("RGB", (int(W), int(H)), "white")
px = im.load()
band = (pts[:, 1] > 0.3) & (pts[:, 1] < 1.5)
for x, z in ((pts[band][:, [0, 2]] - lo) * S).astype(int):
    if 0 <= x < W and 0 <= z < H:
        px[int(x), int(z)] = (150, 160, 155)
dr = ImageDraw.Draw(im)
for i in range(len(ol)):
    a, b = (ol[i] - lo) * S, (ol[(i + 1) % len(ol)] - lo) * S
    dr.line([tuple(a), tuple(b)], fill=(30, 110, 90) if room["edge_status"][i] == "observed" else (200, 80, 20), width=3)
for c in room["cameras"]:
    x, z = (np.array([c[0], c[2]]) - lo) * S
    dr.ellipse([x - 5, z - 5, x + 5, z + 5], fill=(40, 60, 200))
for ob in room["obstacles"]:
    x0, z0, x1, z1 = ob["rect"]
    dr.rectangle([tuple((np.array([x0, z0]) - lo) * S), tuple((np.array([x1, z1]) - lo) * S)], outline=(160, 0, 160), width=2)
dr.line([(10, H - 12), (10 + S, H - 12)], fill=(0, 0, 0), width=3)
dr.text((12, H - 28), "1m", fill=(0, 0, 0))
im.save(d / "topview.png")
print("saved", d / "topview.png")
