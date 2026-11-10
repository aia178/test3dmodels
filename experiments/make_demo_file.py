"""デモ用の部屋ファイル（企画説明で見せる用）を作る。

元: demo/home_corner_cal1.roomfit.json（自宅の角、クローゼットの幅 1 か所で縮尺合わせ、検証用の実測 3 か所）
足すもの:
  - ドア側の壁の写真を、スイッチの上に写真の境目を通さない版（§18、experiments/out/seam_protect/）に差し替える
  - 設備: 部屋のドア（引き戸・入口）、クローゼットの扉（引き戸）、コンセント（位置は壁の写真から読んだおおよその値）
  - 家具 2 つと配置案 3 つ（置ける例・通路が狭くなる要確認の例・問題が出る例）
出力: demo/home_corner_demo.roomfit.json（元のファイルは変えない）

使い方（server フォルダで）: uv run python ../experiments/make_demo_file.py
"""
import base64
import json
import math
from pathlib import Path

here = Path(__file__).resolve().parent
SRC = here.parent / "demo" / "home_corner_cal1.roomfit.json"
DST = here.parent / "demo" / "home_corner_demo.roomfit.json"
WALL1 = here / "out" / "seam_protect" / "wall_1_保護+つなぎ目.jpg"
AT = "2026-10-19T12:00:00+09:00"


def main() -> None:
    d = json.loads(SRC.read_text(encoding="utf-8"))
    d["name"] = "自宅の角（デモ用・1か所だけ測った）"
    d["note"] = "企画説明用。設備・家具・配置案を最初から入れてある。寸法の検証用の実測 3 か所つき。"
    if WALL1.exists():
        d["surfacesB64"]["wall_1.jpg"] = base64.b64encode(WALL1.read_bytes()).decode()
    O = d["jobResult"]["outline"]
    # 縮尺合わせの倍率（アプリと同じ計算: 実測 ÷ 測定の 2 点の長さ）
    cal = next(m for m in d["measurements"] if m["role"] == "calibration")
    est = math.hypot(cal["b"]["x"] - cal["a"]["x"], cal["b"]["z"] - cal["a"]["z"])
    s = cal["measuredM"] / est
    # 設備（復元のままの縮尺で書く。縮尺合わせで位置が一緒に伸び縮みする）。辺 0 = クローゼットの壁、辺 1 = ドアの壁
    e1_start_z = O[1][1]
    door_mid_z = (-0.9495 + -0.0518) / 2          # 検証用のドアの幅の測定の 2 点の中点
    d["fixtures"] = [
        {"id": "fx-door", "type": "door", "edgeIndex": 1, "offset": round(door_mid_z - e1_start_z, 3), "width": 0.85, "height": 2.19, "bottom": 0.0,
         "source": "manual", "confirmed": True, "door": {"hinge": "start", "opensInto": "sliding", "maxAngleDeg": 90, "isEntrance": True},
         "note": "部屋のドア（引き戸、枠込み）"},
        {"id": "fx-closet", "type": "door", "edgeIndex": 0, "offset": round((0.0104 + 1.0695) / 2 - O[0][0], 3), "width": 1.02, "height": 2.26, "bottom": 0.0,
         "source": "manual", "confirmed": True, "door": {"hinge": "start", "opensInto": "sliding", "maxAngleDeg": 90, "isEntrance": False},
         "note": "クローゼットの扉（引き戸）"},
        {"id": "fx-outlet", "type": "outlet", "edgeIndex": 1, "offset": 0.26, "width": 0.07, "height": 0.12, "bottom": 0.18,
         "source": "manual", "confirmed": True, "note": "ドア左のコンセント（壁の写真から読んだおおよその位置）"},
    ]
    d["furniture"] = [
        {"id": "demo-chest", "name": "チェスト（引き出し 4 段）", "category": "チェスト・収納", "w": 0.80, "d": 0.45, "h": 0.90,
         "dimSource": "product_page", "dimConfirmed": True, "dimNote": "商品ページの寸法（例）", "shape": "box", "legClearance": None,
         "ownership": "planned", "useSpace": {"kind": "drawer", "depth": 0.40}, "photoIds": [], "color": "#c98f5a", "createdAt": AT},
        {"id": "demo-shelf", "name": "本棚", "category": "棚・本棚", "w": 0.60, "d": 0.30, "h": 1.80,
         "dimSource": "measured", "dimConfirmed": True, "shape": "box", "legClearance": None,
         "ownership": "owned", "useSpace": {"kind": "access", "depth": 0.50}, "photoIds": [], "color": "#6f8fae", "createdAt": AT},
    ]
    # 配置案（縮尺合わせ後の部屋の座標）。部屋の範囲: x は左の壁〜ドアの壁、z はクローゼットの壁〜手前
    x0, x1 = O[0][0] * s, O[1][0] * s
    z0, z1 = O[0][1] * s, O[2][1] * s
    gap = 0.02
    door_z = (door_mid_z) * s
    shelf_ok = {"id": "p-shelf", "furnitureId": "demo-shelf", "x": round(x1 - 0.15 - gap, 3), "z": round(door_z + 0.425 * s + 0.30 + 0.20, 3),
                "rotY": -math.pi / 2, "elevation": 0}
    chest_ok = {"id": "p-chest", "furnitureId": "demo-chest", "x": round(x0 + 0.225 + gap, 3), "z": round(z0 + 0.85, 3), "rotY": math.pi / 2, "elevation": 0}
    chest_bad = {"id": "p-chest", "furnitureId": "demo-chest", "x": round((x0 + x1) / 2 - 0.05, 3), "z": round(z0 + 0.225 + gap, 3), "rotY": 0.0, "elevation": 0}
    shelf_bad = {"id": "p-shelf", "furnitureId": "demo-shelf", "x": round(x1 - 0.15 - gap, 3), "z": round(door_z, 3), "rotY": -math.pi / 2, "elevation": 0}
    d["layouts"] = [   # 置ける → 要確認 → 問題あり の順に見せる
        {"id": "demo-a", "roomId": "", "name": "案A：本棚だけ置く（置ける例）", "items": [shelf_ok], "updatedAt": AT},
        {"id": "demo-b", "roomId": "", "name": "案B：チェストも置く（通路が狭くなる例）", "items": [shelf_ok, chest_ok], "updatedAt": AT},
        {"id": "demo-c", "roomId": "", "name": "案C：クローゼットとドアの前に置く（問題が出る例）", "items": [chest_bad, shelf_bad], "updatedAt": AT},
    ]
    DST.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    print(f"保存: {DST}（縮尺合わせの倍率 {s:.4f}、部屋 x {x0:.2f}〜{x1:.2f}m・z {z0:.2f}〜{z1:.2f}m）")


if __name__ == "__main__":
    main()
