"""印刷した AprilTag（縮尺マーカー）から、部屋の大きさを合わせる倍率を出す（メジャーを使わない縮尺合わせ。任意の機能）。

マーカーは docs/markers/ の A4 の紙（黒い四角の一辺 MARKER_SIZE_M、AprilTag 36h11）。利用者は印刷の大きさを
カード（ISO の ID-1、85.60×53.98mm）を重ねて確かめてから、平らな壁に貼って撮る。

測り方:
1. 元の写真（数千画素）でマーカーの 4 つの角を見つける（OpenCV の aruco、角は小数の画素まで詰める）。
2. 角の位置を、モデルの画素（392×518 前後）へ移す。モデルの 1 画素ごとの 3D の点は数 mm〜1cm 揺れるので、角の 4 点だけを拾わず、
   マーカーの周りの 3D の点に「画素 → 3D」の一次式（平らな壁の上なら、狭い範囲ではほぼ一次式）を当てはめて、その式で角の 3D を出す。
3. 4 辺の長さの平均と MARKER_SIZE_M の比が、その写真・そのマーカーでの倍率。4 辺の長さがそろわない（曲がって貼られた、
   手前に物がある、当てはめが崩れた）もの、モデルの画像で小さすぎるものは捨てる。
4. すべての写真・マーカーの倍率の中央値を使う。ばらつき（最大と最小の差 ÷ 中央値）も返す。

精度は実物でまだ確かめていない（docs/検証記録.md に結果を書く）。スイッチの板（規格の寸法と仮定）で同じことをしたときは、
見積もりが 0.97〜1.04 とばらついた（実測からの倍率は 0.962）。マーカーは大きさが確実で、角がはっきりしているので、それより良いと見込んでいる。
"""
from __future__ import annotations

import numpy as np

MARKER_SIZE_M = 0.150          # 印刷した黒い四角の一辺
MARKER_DICT = "DICT_APRILTAG_36h11"
MIN_MODEL_PX = 12.0            # モデルの画像でマーカーの一辺がこれ未満なら、3D が粗すぎるので使わない
MAX_SIDE_RATIO = 1.15          # 4 辺の長さの最大 ÷ 最小がこれを超えたら使わない
FIT_MARGIN = 0.6               # 当てはめに使う範囲: マーカーの外側へ、一辺のこの割合だけ広げる


def detect(photo: np.ndarray) -> list[tuple[int, np.ndarray]]:
    """写真（H×W×3 または H×W）からマーカーを探す。戻り値: [(番号, 角 4×2（元の写真の画素、左上から時計回り）)]。"""
    import cv2

    g = photo if photo.ndim == 2 else cv2.cvtColor(np.ascontiguousarray(photo), cv2.COLOR_RGB2GRAY)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, MARKER_DICT)), params)
    corners, ids, _ = det.detectMarkers(g)
    if ids is None:
        return []
    return [(int(i), c.reshape(4, 2).astype(np.float64)) for i, c in zip(ids.ravel(), corners)]


def _side_lengths(P: np.ndarray) -> np.ndarray:
    return np.linalg.norm(P - np.roll(P, -1, axis=0), axis=1)


def measure_one(grid_room: np.ndarray, corners_model: np.ndarray) -> dict | None:
    """モデルの画素での角（4×2、画素 i の中心が i+0.5）から、マーカーの一辺の 3D の長さを出す。使えなければ None（理由は "reject"）。"""
    hm, wm = grid_room.shape[:2]
    side_px = float(np.mean(_side_lengths(corners_model)))
    if side_px < MIN_MODEL_PX:
        return {"reject": "small", "side_px": round(side_px, 1)}
    lo = np.floor(corners_model.min(0) - FIT_MARGIN * side_px).astype(int)
    hi = np.ceil(corners_model.max(0) + FIT_MARGIN * side_px).astype(int)
    x0, y0 = max(0, lo[0]), max(0, lo[1])
    x1, y1 = min(wm, hi[0] + 1), min(hm, hi[1] + 1)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return {"reject": "edge"}
    G = grid_room[y0:y1, x0:x1].reshape(-1, 3)
    vv, uu = np.mgrid[y0:y1, x0:x1]
    A = np.c_[uu.ravel() + 0.5, vv.ravel() + 0.5, np.ones(uu.size)]
    ok = np.isfinite(G).all(1)
    if ok.sum() < 30:
        return {"reject": "no_points"}
    # 一次式（画素 → 3D）を当てはめ、外れた点（手前の物・別の面）を 1 回除いてやり直す
    coef, *_ = np.linalg.lstsq(A[ok], G[ok], rcond=None)
    res = np.linalg.norm(A[ok] @ coef - G[ok], axis=1)
    keep = res < max(3 * np.median(res), 0.005)
    coef, *_ = np.linalg.lstsq(A[ok][keep], G[ok][keep], rcond=None)
    P = np.c_[corners_model, np.ones(4)] @ coef
    sides = _side_lengths(P)
    if sides.max() / max(sides.min(), 1e-9) > MAX_SIDE_RATIO:
        return {"reject": "skewed", "sides_m": [round(float(s), 4) for s in sides]}
    return {"side_m": float(np.mean(sides)), "side_px": round(side_px, 1),
            "fit_residual_mm": round(float(np.median(res[keep])) * 1000, 1)}


def estimate_scale(views: list, size_m: float = MARKER_SIZE_M) -> dict | None:
    """views: textures.ViewData の並び（grid_room・photo・to_photo を使う）。マーカーが 1 つも使えなければ None。
    戻り値: {"factor": 部屋に掛ける倍率, "n": 使えた数, "spread_pct": ばらつき, "size_m": マーカーの一辺, "detections": [...]}"""
    rows = []
    for vi, vd in enumerate(views):
        s, ox, oy, k = vd.to_photo
        for mid, c in detect(vd.photo):
            cm = np.c_[c[:, 0] / k * s - ox, c[:, 1] / k * s - oy]   # 元の写真の画素 → モデルの画素
            r = measure_one(vd.grid_room, cm)
            if r is None:
                continue
            r.update(view=vi, id=mid)
            if "side_m" in r:
                r["factor"] = round(size_m / r["side_m"], 4)
                r["side_m"] = round(r["side_m"], 4)
            rows.append(r)
    f = np.array([r["factor"] for r in rows if "factor" in r])
    if len(f) == 0:
        return {"factor": None, "n": 0, "size_m": size_m, "detections": rows} if rows else None
    med = float(np.median(f))
    return {"factor": round(med, 4), "n": int(len(f)), "spread_pct": round(float((f.max() - f.min()) / med * 100), 1),
            "size_m": size_m, "detections": rows}
