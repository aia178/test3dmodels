"""復元の出力フォルダ（room.json, raw.npz, views/, images/）と元の写真から、壁と床の「写真を貼った画像」を作る。

アプリ本体の処理（roomfit.textures.make_surfaces）を、既にある復元結果に対して実行するための台本。
出力: <出力フォルダ>/surfaces/wall_<辺>.jpg, floor.jpg, surfaces.json（room.json の "surfaces" と同じ形）

使い方（server フォルダで）: uv run python ../experiments/make_surfaces.py <復元の出力フォルダ> <元の写真のフォルダ>
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
from roomfit.reconstruct import load_raw  # noqa: E402
from roomfit.textures import ViewData, make_surfaces, photo_mapping  # noqa: E402
from roomfit.worker import XYZ_INVALID, save_surfaces  # noqa: E402


def load_views(out: Path, photo_dir: Path) -> tuple[dict, list[ViewData]]:
    """復元の出力フォルダと元の写真から、room.json の中身と写真ごとの ViewData を読む。"""
    room = json.loads((out / "room.json").read_text(encoding="utf-8"))
    raw = load_raw(out / "raw.npz")
    T = np.array(room["T_room_from_world"])
    meta = json.loads((out / "views" / "views.json").read_text(encoding="utf-8"))
    photos = sorted(p for p in photo_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    pre = sorted((out / "images").glob("*.jpg"))
    views = []
    for m in meta:
        n, w, h = m["n"], m["w"], m["h"]
        C = T @ raw["cam2world"][n]; K = raw["intrinsics"][n]
        # 各画素が見ている 3D の点（部屋座標、mm）。古い形式の raw.npz では、色の一致で画素との対応を作り直したもの（export_views_legacy.py）
        q = np.frombuffer((out / "views" / f"{n:03d}.xyz").read_bytes(), "<i2").reshape(h, w, 3).astype(np.float64)
        q[(q == XYZ_INVALID).any(-1)] = np.nan
        q /= 1000.0
        photo = np.asarray(ImageOps.exif_transpose(Image.open(photos[n])).convert("RGB"))
        pw = Image.open(pre[n]).size
        views.append(ViewData(cam2room=C, K=K, grid_room=q, photo=photo,
                              to_photo=photo_mapping((h, w), pw, (photo.shape[1], photo.shape[0]))))
    return room, views


if __name__ == "__main__":
    out, photo_dir = Path(sys.argv[1]), Path(sys.argv[2])
    room, views = load_views(out, photo_dir)
    surf = make_surfaces(room["outline"], room["edge_status"], room["ceiling_height"], views)
    info = save_surfaces(surf, out)
    (out / "surfaces" / "surfaces.json").write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False))
