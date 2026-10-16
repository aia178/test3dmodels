"""復元結果（room.json と cloud.bin）を、アプリで読み込める「部屋ファイル」にまとめる。

アプリのホーム →「部屋ファイルを読み込む」で開ける。読み込み時の処理は、サーバーの復元が終わったときと同じ
（web/src/core/room.ts の applyJobResult）。元の写真は含めない。

使い方（server フォルダで）:
  uv run python ../experiments/make_room_file.py <room.json と cloud.bin のあるフォルダ> <部屋の名前> <出力ファイル> [測る写真の数=6] [測定の JSON]
  測定の JSON を渡すと、検証用の測定（実測と、図での長さの比較）を最初から入れた状態で読み込まれる。
"""
import base64
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

src, name, dst = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
room = json.loads((src / "room.json").read_text(encoding="utf-8"))
if (src / "surfaces" / "surfaces.json").exists():
    room["surfaces"] = json.loads((src / "surfaces" / "surfaces.json").read_text(encoding="utf-8"))
cloud = (src / "cloud.bin").read_bytes()
assert len(cloud) == room["cloud_points"] * 15, "cloud.bin の大きさが room.json と合いません"
views = []
vdir = src / "views"
if (vdir / "views.json").exists():
    meta = json.loads((vdir / "views.json").read_text(encoding="utf-8"))
    max_views = int(sys.argv[4]) if len(sys.argv) > 4 else 6  # ファイルが大きくなりすぎないよう、等間隔に間引く
    pick = meta if len(meta) <= max_views else [meta[round(i * (len(meta) - 1) / (max_views - 1))] for i in range(max_views)]
    for v in pick:
        views.append({**{k: v[k] for k in ("n", "w", "h")},
                      "jpgB64": base64.b64encode((vdir / f"{v['n']:03d}.jpg").read_bytes()).decode("ascii"),
                      "xyzB64": base64.b64encode((vdir / f"{v['n']:03d}.xyz").read_bytes()).decode("ascii")})
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(json.dumps({
    "app": "roomfit", "kind": "recon", "version": 1,
    "exportedAt": datetime.now(timezone.utc).isoformat(),
    "name": name,
    "note": "写真からの復元結果。元の写真は含みません。縮尺はモデルの推定のまま（実測での補正なし）。",
    "jobResult": {**room, "views": [{k: v[k] for k in ("n", "w", "h")} for v in views]},
    "views": views,
    "cloudB64": base64.b64encode(cloud).decode("ascii"),
    # 壁・床に写真を貼った画像（make_surfaces.py の出力があれば）
    "surfacesB64": {f.name: base64.b64encode(f.read_bytes()).decode("ascii") for f in sorted((src / "surfaces").glob("*.jpg"))} if (src / "surfaces").exists() else {},
    **({"measurements": json.loads(Path(sys.argv[5]).read_text(encoding="utf-8"))} if len(sys.argv) > 5 else {}),
}, ensure_ascii=False), encoding="utf-8")
print(f"{dst}  ({dst.stat().st_size / 1e6:.1f} MB, {room['cloud_points']} points, {room['meta']['n_views']} photos, {len(views)} views for measuring)")
