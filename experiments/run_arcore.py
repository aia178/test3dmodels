"""実験: 撮影アプリ（ARCore）の記録フォルダ → 部屋形状(room.json) と上面図 PNG。

使い方（server フォルダで）:
  uv run python ../experiments/run_arcore.py <記録フォルダ> <出力フォルダ>
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
from roomfit.arcore_import import ARCORE_UP, CaptureError, load_capture  # noqa: E402
from roomfit.structure import StructureError, StructureParams, build_room_structure  # noqa: E402

src, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
try:
    cap = load_capture(src)
except CaptureError as e:
    print("CAPTURE FAILED:", e.message)
    sys.exit(2)
print(json.dumps(cap["meta"], ensure_ascii=False, indent=1))
pts = cap["points"].astype(np.float64)
print("points:", len(pts), " height range (ARCore Y):", np.round(np.percentile(pts[:, 1], [1, 50, 99]), 2).tolist())
cams = cap["cam2world"][:, :3, 3]
print("camera height range:", round(float(cams[:, 1].min()), 2), round(float(cams[:, 1].max()), 2))

try:
    res = build_room_structure(pts, cap["cam2world"], cap["cam_index"], StructureParams(), up_hint=ARCORE_UP)
except StructureError as e:
    print("STRUCTURE FAILED:", e.code, e.message)
    res = None

T = res.T_room_from_world if res else np.eye(4)
rp = pts @ T[:3, :3].T + T[:3, 3]
xz = rp[:, [0, 2]] if res else pts[:, [0, 2]]
h = rp[:, 1] if res else pts[:, 1] - np.percentile(pts[:, 1], 1)
lo, hi = np.percentile(xz, 0.5, axis=0) - 0.3, np.percentile(xz, 99.5, axis=0) + 0.3
S = 900 / float((hi - lo).max())
W, H = np.ceil((hi - lo) * S).astype(int)
im = Image.new("RGB", (int(W), int(H)), "white")
band = (h > 0.3) & (h < 1.8)
ij = ((xz[band] - lo) * S).astype(int)
ok = (ij[:, 0] >= 0) & (ij[:, 0] < W) & (ij[:, 1] >= 0) & (ij[:, 1] < H)
arr = np.asarray(im).copy()
arr[ij[ok, 1], ij[ok, 0]] = (150, 160, 155)
im = Image.fromarray(arr)
dr = ImageDraw.Draw(im)
cr = cams @ T[:3, :3].T + T[:3, 3]
for c in cr:
    x, z = (np.array([c[0], c[2]]) - lo) * S
    dr.ellipse([x - 4, z - 4, x + 4, z + 4], fill=(40, 60, 200))
if res:
    ol = np.array(res.outline)
    for i in range(len(ol)):
        a, b = (ol[i] - lo) * S, (ol[(i + 1) % len(ol)] - lo) * S
        dr.line([tuple(a), tuple(b)], fill=(30, 110, 90) if res.edge_status[i] == "observed" else (200, 80, 20), width=3)
    lens = [round(float(np.linalg.norm(ol[(i + 1) % len(ol)] - ol[i])), 2) for i in range(len(ol))]
    print("outline edges (m):", lens)
    print("edge status:", res.edge_status)
    print("ceiling:", res.ceiling_height, res.ceiling_observed, " floor tilt vs ARCore up (deg):", round(res.diagnostics["floor_tilt_from_camera_up_deg"], 2))
dr.line([(10, H - 12), (10 + S, H - 12)], fill=(0, 0, 0), width=3)
dr.text((12, H - 28), "1m", fill=(0, 0, 0))
im.save(out / "topview.png")
print("saved", out / "topview.png")
