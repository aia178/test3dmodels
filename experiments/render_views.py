"""復元した 3D の点を、撮影した位置とは別の視点から描いた画像を作る（3D になっていることの確認用）。

写真をそのまま貼ったものではなく、点の 3D 位置を別の視点へ投影して描く。視点を変えても形が保たれていれば、
奥行きが復元できている。穴（白い所）は、どの写真にも写っていない場所。

使い方（server フォルダで）: uv run python ../experiments/render_views.py <raw.npz と room.json のあるフォルダ> [縮尺]
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
from roomfit.reconstruct import load_raw  # noqa: E402

out = Path(sys.argv[1])
scale = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
raw = load_raw(out / "raw.npz")
room = json.loads((out / "room.json").read_text(encoding="utf-8"))
T = np.array(room["T_room_from_world"])
keep = raw["conf"] >= np.percentile(raw["conf"], 15)
P = (raw["points"][keep].astype(np.float64) @ T[:3, :3].T + T[:3, 3]) * scale
C = raw["colors"][keep]
cams = np.array(room["cameras"]) * scale


def look_at(eye, target, up=np.array([0.0, 1.0, 0.0])):
    f = target - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return np.stack([r, -u, f])  # 行: カメラの右・下・前（世界座標での向き）


def render(eye, target, w=1280, h=960, fov_deg=60.0, splat=2):
    R = look_at(np.asarray(eye, float), np.asarray(target, float))
    pc = (P - eye) @ R.T
    z = pc[:, 2]
    ok = z > 0.2
    f = 0.5 * w / np.tan(np.radians(fov_deg) / 2)
    u = (f * pc[ok, 0] / z[ok] + w / 2).astype(np.int32)
    v = (f * pc[ok, 1] / z[ok] + h / 2).astype(np.int32)
    zz, cc = z[ok], C[ok]
    order = np.argsort(-zz)  # 遠い点から描き、近い点で上書きする
    u, v, cc = u[order], v[order], cc[order]
    img = np.full((h, w, 3), 255, np.uint8)
    for dy in range(splat):
        for dx in range(splat):
            uu, vv2 = u + dx, v + dy
            m = (uu >= 0) & (uu < w) & (vv2 >= 0) & (vv2 < h)
            img[vv2[m], uu[m]] = cc[m]
    return Image.fromarray(img)


# 視点を決める: 壁の点の重心を見る。撮影位置の平均から、後ろ・上へ離れた位置と、横へ回り込んだ位置
band = (P[:, 1] > 0.3) & (P[:, 1] < 2.0)
target = np.array([np.median(P[band, 0]), 1.1, np.median(P[band, 2])])
cam_mean = cams.mean(axis=0)
back = cam_mean - target
back[1] = 0
dist = np.linalg.norm(back)
back /= dist


def rot_y(vec, deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([vec[0] * c + vec[2] * s, vec[1], -vec[0] * s + vec[2] * c])


views = {
    "上から見下ろす": (target + back * (dist + 1.6) + np.array([0, 2.6, 0]), target + np.array([0, -0.4, 0])),
    "右へ回り込む": (target + rot_y(back, 40) * (dist + 1.0) + np.array([0, 0.9, 0]), target),
    "左へ回り込む": (target + rot_y(back, -40) * (dist + 1.0) + np.array([0, 0.9, 0]), target),
    "真上": (target + np.array([0.001, 6.0, 0.0]), target),
}
print("points:", len(P), " capture camera mean:", cam_mean.round(2).tolist(), " target:", target.round(2).tolist())
for i, (name, (eye, tgt)) in enumerate(views.items()):
    im = render(eye, tgt)
    d = ImageDraw.Draw(im)
    moved = float(np.linalg.norm(eye - cam_mean))
    d.rectangle([0, 0, 520, 26], fill=(255, 255, 255))
    d.text((8, 6), f"view {i + 1}: {moved:.1f} m away from where the photos were taken", fill=(40, 40, 40))
    im.save(out / f"render_{i + 1}.jpg", quality=88)
    print(f"render_{i + 1}.jpg  {name}: eye {np.round(eye, 2).tolist()}  ({moved:.1f} m from capture position)")
