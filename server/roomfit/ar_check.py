"""撮影アプリ（ARCore）の位置を、写真どうしの見え方で検算し、区間ごとに「AR の位置を使う・使わない」を決める。

ARCore は「追跡中」と表示したまま位置が飛んだり、移動の向きを取り違えたりする（OnePlus 13 で多い、docs/検証記録.md §16〜§17）。
隣り合うコマ（1 つ先・2 つ先）の写真から特徴点を対応させ、本質行列（2 枚の写真の対応から向きの変化と動いた方向を求める計算）で
出した「向きの変化」「動いた方向」を、AR の値と比べる。動いた量（m）は写真からは分からないので比べない。

区間の決め方:
1. 組ごとに、向きの差が ROT_BAD_DEG を超えるか、動いた方向の差が DIR_BAD_DEG を超えたら「食い違い」。
2. コマ k と k+1 の間をまたぐ組（k-1→k+1, k→k+1, k→k+2）で、食い違いが一致以上なら、そこで区間を切る。
   隣り合うコマの間で 1 秒あたり MAX_SPEED_MPS を超えて動いた（位置が飛んだ）所も切る。
3. 区間の中で方向を判定できた組が MIN_DIR_PAIRS 以上・食い違いが MAX_BAD_SHARE 以下・向きの差の中央値が ROT_OK_DEG 以下なら「使える」。
4. 使える区間のうち、いちばん長い 1 つだけで AR の位置を使う。区間どうしは間で位置が食い違っているので、混ぜない。
   写真が無い・対応が取れないなどで 1 組も判定できなければ「未確認」。どのコマも使わない側に倒す（OnePlus で位置が 10 回飛んだのに
   1 組も判定できなかった記録があった）。写真だけで復元できない処理（AR の深度を重ねる load_capture）は、呼ぶ側で扱いを決める。
5. 使う区間の中でカメラが動いた範囲（extent_m）も返す。動きが小さいと、AR の位置が実寸でも大きさの手がかりにならない（§16）。
   MIN_EXTENT_M 未満なら scale_cue = "weak"（形はそろうが、大きさは要確認）。10:35 の記録では、17cm しか動いていない 4 コマに
   位置を渡すと、写真どうしの食い違いは 35.6mm → 6.0mm に減ったが、扉の幅は −8.5% になった（§19）。
   なお、写真で確かめられるのは動いた「向き」だけで、動いた「量」が正しいかは分からない。

しきい値は保存した 24 本の記録（vivo 8 本・OnePlus 16 本）で決めた仮の値（§19）。ほかの端末・部屋での妥当性は未確認。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SCALE = 0.5              # 特徴点を探す前に写真を縮める倍率（1920×1080 → 960×540）
MIN_INLIERS = 40
MIN_BASE_M = 0.05        # AR の移動がこれ未満の組は、動いた方向を判定しない
MIN_PARALLAX_PX = 8.0    # 回転を除いた画素の動きがこれ未満の組も、動いた方向を判定しない
ROT_BAD_DEG = 5.0
DIR_BAD_DEG = 30.0
ROT_OK_DEG = 3.0
MIN_DIR_PAIRS = 2
MAX_BAD_SHARE = 0.10
MIN_FRAMES = 3
MAX_SPEED_MPS = 1.0
MIN_EXTENT_M = 0.5       # 使う区間でカメラがこれ未満しか動いていなければ、大きさの手がかりは弱い（§16・§19、仮の値）


def _rot_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def compare_pair(c2w_a: np.ndarray, c2w_b: np.ndarray, K: np.ndarray, pts_a: np.ndarray, pts_b: np.ndarray) -> dict:
    """写真 a, b の対応点（元の画素、N×2）と AR の姿勢（カメラ→世界、OpenCV の向き）を比べる。"""
    import cv2

    if len(pts_a) < MIN_INLIERS:
        return {"status": "few_matches", "matches": len(pts_a)}
    n1 = cv2.undistortPoints(pts_a.reshape(-1, 1, 2).astype(np.float64), K, None).reshape(-1, 2)
    n2 = cv2.undistortPoints(pts_b.reshape(-1, 1, 2).astype(np.float64), K, None).reshape(-1, 2)
    E, inl = cv2.findEssentialMat(n1, n2, np.eye(3), cv2.RANSAC, 0.999, 1.0 / K[0, 0])
    if E is None or E.shape != (3, 3):
        return {"status": "no_E", "matches": len(pts_a)}
    nin, R, t, _ = cv2.recoverPose(E, n1, n2, np.eye(3), mask=inl.copy())
    if nin < MIN_INLIERS:
        return {"status": "few_inliers", "matches": len(pts_a), "inliers": int(nin)}
    # AR の相対姿勢（カメラ a の座標 → カメラ b の座標）
    Ra, ca, Rb, cb = c2w_a[:3, :3], c2w_a[:3, 3], c2w_b[:3, :3], c2w_b[:3, 3]
    R_ar = Rb.T @ Ra
    t_ar = Rb.T @ (ca - cb)
    base = float(np.linalg.norm(ca - cb))
    # 回転を除いたときの画素の動き（視差）。小さいと動いた方向は写真から決まらない
    h1 = np.c_[n1, np.ones(len(n1))] @ R_ar.T
    parallax = float(np.median(np.linalg.norm(h1[:, :2] / h1[:, 2:3] - n2, axis=1)) * K[0, 0])
    out = {"status": "ok", "matches": len(pts_a), "inliers": int(nin), "rot_err_deg": round(_rot_deg(R.T @ R_ar), 2),
           "base_m": round(base, 3), "parallax_px": round(parallax, 1)}
    if base > MIN_BASE_M and parallax > MIN_PARALLAX_PX:
        cosang = float(np.dot(t.ravel(), t_ar / np.linalg.norm(t_ar)))
        out["dir_err_deg"] = round(float(np.degrees(np.arccos(np.clip(cosang, -1, 1)))), 1)
    return out


def measure_pairs(frames: list[dict]) -> list[dict]:
    """frames: [{"c2w": 4×4（OpenCV の向き）, "K": 3×3, "jpg": Path | None, "ts": 秒}]。隣り合うコマ（1 つ先・2 つ先）の組を比べる。"""
    import cv2

    sift = cv2.SIFT_create(4000)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    feats = []
    for f in frames:
        im = cv2.imread(str(f["jpg"]), cv2.IMREAD_GRAYSCALE) if f.get("jpg") else None
        if im is None:                                   # 写真が無い・壊れている（転送で 0 バイトになったものがあった）
            feats.append(None)
            continue
        im = cv2.resize(im, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
        kp, des = sift.detectAndCompute(im, None)
        feats.append((np.float32([k.pt for k in kp]) / SCALE, des) if des is not None and len(des) >= 50 else None)
    rows = []
    for i in range(len(frames)):
        for j in (i + 1, i + 2):
            if j >= len(frames):
                continue
            a, b = frames[i], frames[j]
            r: dict = {"a": i, "b": j, "dt": round(b["ts"] - a["ts"], 3),
                       "move_m": round(float(np.linalg.norm(b["c2w"][:3, 3] - a["c2w"][:3, 3])), 3)}
            if feats[i] is None or feats[j] is None:
                r["status"] = "no_features"
            else:
                m = matcher.knnMatch(feats[i][1], feats[j][1], k=2)
                good = [x[0] for x in m if len(x) == 2 and x[0].distance < 0.75 * x[1].distance]
                r.update(compare_pair(a["c2w"], b["c2w"], a["K"], feats[i][0][[g.queryIdx for g in good]], feats[j][0][[g.trainIdx for g in good]]))
            rows.append(r)
    return rows


def _bad(r: dict) -> bool | None:
    """食い違いなら True、合っていれば False、判定できなければ None。"""
    if r.get("status") != "ok":
        return None
    if r["rot_err_deg"] > ROT_BAD_DEG or r.get("dir_err_deg", 0.0) > DIR_BAD_DEG:
        return True
    return False if "dir_err_deg" in r else None


def decide_segments(pairs: list[dict], n: int, positions: np.ndarray | None = None) -> dict:
    """組ごとの比較から、AR の位置を使うコマを決める。戻り値の "use_pose" はコマごとの真偽。
    positions: コマごとの AR のカメラ位置（n×3、m）。渡すと、使う区間の中でカメラが動いた範囲 extent_m を返す。"""
    judged = [r for r in pairs if _bad(r) is not None]
    if not judged:
        return {"status": "unchecked", "use_pose": [False] * n, "segments": [], "pairs_judged": 0, "pairs_bad": 0, "dir_err_med": None,
                "extent_m": None, "scale_cue": None, "note": "写真どうしの対応が取れず、AR の位置を検算できなかった"}
    cut = []
    for k in range(n - 1):
        span = [_bad(r) for r in pairs if r["a"] <= k < r["b"]]
        nb, ng = sum(x is True for x in span), sum(x is False for x in span)
        # 位置の飛びは、隣り合うコマ（k→k+1）の速さで見る（2 つ先の組で見ると、飛びの 1 つ手前でも切ってしまう）
        jump = any(r["a"] == k and r["b"] == k + 1 and 0 < r["dt"] < 2.0 and r["move_m"] / r["dt"] > MAX_SPEED_MPS for r in pairs)
        if jump or (nb > 0 and nb >= ng):
            cut.append(k)
    segs, start = [], 0
    for k in cut + [n - 1]:
        segs.append((start, k))
        start = k + 1
    out_segs = []
    for s, e in segs:
        inside = [r for r in pairs if s <= r["a"] and r["b"] <= e]
        d = [r for r in inside if _bad(r) is not None]
        rot = [r["rot_err_deg"] for r in inside if r.get("status") == "ok"]
        nbad = sum(_bad(r) for r in d)
        ok = (e - s + 1 >= MIN_FRAMES and len(d) >= MIN_DIR_PAIRS and nbad <= MAX_BAD_SHARE * len(d)
              and bool(rot) and float(np.median(rot)) <= ROT_OK_DEG)
        out_segs.append({"first": s, "last": e, "frames": e - s + 1, "dir_pairs": len(d), "bad_pairs": nbad,
                         "rot_err_med": round(float(np.median(rot)), 2) if rot else None, "usable": ok})
    good = [g for g in out_segs if g["usable"]]
    use = [False] * n
    if good:
        best = max(good, key=lambda g: (g["frames"], g["dir_pairs"]))
        best["chosen"] = True
        for i in range(best["first"], best["last"] + 1):
            use[i] = True
    nbad_all = sum(_bad(r) for r in judged)
    status = "rejected" if not good else ("ok" if all(use) else "partial")
    dirs = [r["dir_err_deg"] for r in judged if "dir_err_deg" in r]
    extent = None
    if positions is not None and any(use):
        P = np.asarray(positions, float)[np.array(use)]
        extent = round(float(np.linalg.norm(P[:, None] - P[None], axis=-1).max()), 3)
    return {"status": status, "use_pose": use, "segments": out_segs, "pairs_judged": len(judged), "pairs_bad": nbad_all,
            "dir_err_med": round(float(np.median(dirs)), 1) if dirs else None, "extent_m": extent,
            "scale_cue": None if extent is None else ("ok" if extent >= MIN_EXTENT_M else "weak")}


def check_capture(folder: Path) -> dict:
    """撮影アプリの記録フォルダ（NNN.json / NNN.jpg）を検算する。use_pose は記録のコマの順（NNN の順）。"""
    from .arcore_import import GL_FROM_CV, quat_to_rot

    frames = []
    for p in sorted(folder.glob("[0-9][0-9][0-9].json")):
        j = json.loads(p.read_text(encoding="utf-8"))
        c2w = np.eye(4)
        c2w[:3, :3] = quat_to_rot(j["q"]) @ GL_FROM_CV
        c2w[:3, 3] = j["t"]
        K = np.array([[j["fx"], 0, j["cx"]], [0, j["fy"], j["cy"]], [0, 0, 1.0]])
        jpg = p.with_suffix(".jpg")
        frames.append({"id": j["id"], "c2w": c2w, "K": K, "jpg": jpg if jpg.exists() else None, "ts": j["timestampNs"] / 1e9})
    pairs = measure_pairs(frames)
    res = decide_segments(pairs, len(frames), np.array([f["c2w"][:3, 3] for f in frames]) if frames else None)
    res["frame_ids"] = [f["id"] for f in frames]
    res["pairs"] = pairs
    return res
