"""実験: ARCore の位置・向きが「写真どうしの見え方」と食い違っていないかを、写真だけから確かめる（astra 案 P1）。

狙い: OnePlus 13 は ARCore が「追跡中」と言いながら位置が飛ぶ。TRACKING の表示だけでは見分けられないので、
     隣り合う保存コマの写真から、特徴点の対応で「向きの変化」と「動いた方向」を求め、AR の値と比べる。
     食い違いが大きい組を「AR を信用しない区間」の候補にする。

座標: 記録の姿勢は OpenGL 規約（+X 右, +Y 上, -Z 前）のカメラ → 世界。OpenCV 規約へは R_cv = R_gl @ diag(1,-1,-1)。
     写真はセンサーの向きのまま使う（回さない）。内部パラメータも保存された JPEG のもの。

使い方（server フォルダで）:
  uv run python ../experiments/ar_consistency.py <記録フォルダ...>  → 標準出力に表、experiments/out/ar_consistency.json に詳細
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
from roomfit.arcore_import import GL_FROM_CV, quat_to_rot  # noqa: E402

SCALE = 0.5            # 特徴点を探す前に写真を縮める倍率（1920×1080 → 960×540）
MIN_INLIERS = 40


def load(d: Path):
    frames = []
    for f in sorted(d.glob("[0-9][0-9][0-9].json")):
        j = json.loads(f.read_text(encoding="utf-8"))
        c2w = np.eye(4)
        c2w[:3, :3] = quat_to_rot(j["q"]) @ GL_FROM_CV
        c2w[:3, 3] = j["t"]
        K = np.array([[j["fx"], 0, j["cx"]], [0, j["fy"], j["cy"]], [0, 0, 1.0]])
        frames.append({"id": j["id"], "ts": j["timestampNs"] / 1e9, "c2w": c2w, "K": K, "jpg": f.with_suffix(".jpg"),
                       "tracking": j.get("tracking")})
    return frames


def rot_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def feats(sift, fr):
    im = cv2.imread(str(fr["jpg"]), cv2.IMREAD_GRAYSCALE)
    if im is None:                                   # 転送で壊れた写真（0 バイト）
        return np.zeros((0, 2), np.float32), None
    im = cv2.resize(im, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
    kp, des = sift.detectAndCompute(im, None)
    return np.float32([k.pt for k in kp]) / SCALE, des


def compare(a, b, fa, fb, matcher):
    pa, da = fa
    pb, db = fb
    if da is None or db is None or len(da) < 50 or len(db) < 50:
        return {"status": "few_features"}
    m = matcher.knnMatch(da, db, k=2)
    good = [x[0] for x in m if len(x) == 2 and x[0].distance < 0.75 * x[1].distance]
    if len(good) < MIN_INLIERS:
        return {"status": "few_matches", "matches": len(good)}
    p1 = pa[[g.queryIdx for g in good]]
    p2 = pb[[g.trainIdx for g in good]]
    # 2 枚の内部パラメータは同じとみなせる（同じ記録）。正規化座標で本質行列を求める
    n1 = cv2.undistortPoints(p1.reshape(-1, 1, 2), a["K"], None).reshape(-1, 2)
    n2 = cv2.undistortPoints(p2.reshape(-1, 1, 2), b["K"], None).reshape(-1, 2)
    E, inl = cv2.findEssentialMat(n1, n2, np.eye(3), cv2.RANSAC, 0.999, 1.0 / a["K"][0, 0])
    if E is None or E.shape != (3, 3):
        return {"status": "no_E", "matches": len(good)}
    nin, R, t, _ = cv2.recoverPose(E, n1, n2, np.eye(3), mask=inl.copy())
    # AR の相対姿勢（カメラ a の座標 → カメラ b の座標）
    Ra, ca, Rb, cb = a["c2w"][:3, :3], a["c2w"][:3, 3], b["c2w"][:3, :3], b["c2w"][:3, 3]
    R_ar = Rb.T @ Ra
    t_ar = Rb.T @ (ca - cb)
    base = float(np.linalg.norm(ca - cb))
    # 回転を除いたときの画素の動き（視差）。小さいと動いた方向は写真から決まらない
    h1 = cv2.convertPointsToHomogeneous(n1).reshape(-1, 3) @ R_ar.T
    h1 = h1[:, :2] / h1[:, 2:3]
    parallax_px = float(np.median(np.linalg.norm(h1 - n2, axis=1)) * a["K"][0, 0])
    out = {"status": "ok", "matches": len(good), "inliers": int(nin), "rot_ar_deg": round(rot_deg(R_ar), 2),
           "rot_err_deg": round(rot_deg(R.T @ R_ar), 2), "base_m": round(base, 3), "parallax_px": round(parallax_px, 1)}
    if base > 0.05 and parallax_px > 8:
        cosang = float(np.dot(t.ravel(), t_ar / np.linalg.norm(t_ar)))
        out["dir_err_deg"] = round(float(np.degrees(np.arccos(np.clip(cosang, -1, 1)))), 1)
    return out


def main() -> None:
    sift = cv2.SIFT_create(4000)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    report = {}
    for d in map(Path, sys.argv[1:]):
        sess = json.loads((d / "session.json").read_text(encoding="utf-8"))
        frames = load(d)
        if len(frames) < 3:
            continue
        fe = [feats(sift, f) for f in frames]
        rows = []
        for i in range(len(frames) - 1):
            for gap in (1, 2):
                j = i + gap
                if j >= len(frames):
                    continue
                r = compare(frames[i], frames[j], fe[i], fe[j], matcher)
                r.update({"a": frames[i]["id"], "b": frames[j]["id"], "dt": round(frames[j]["ts"] - frames[i]["ts"], 2)})
                rows.append(r)
        ok = [r for r in rows if r["status"] == "ok" and r["inliers"] >= MIN_INLIERS]
        rot = np.array([r["rot_err_deg"] for r in ok]) if ok else np.zeros(0)
        dirs = np.array([r["dir_err_deg"] for r in ok if "dir_err_deg" in r])
        summ = {"device": sess.get("device"), "poseJumps": sess.get("poseJumps"), "frames": len(frames), "pairs": len(rows), "checked": len(ok),
                "rot_err_med": round(float(np.median(rot)), 2) if len(rot) else None,
                "rot_err_max": round(float(rot.max()), 2) if len(rot) else None,
                "rot_bad_gt3deg": int((rot > 3).sum()), "dir_checked": len(dirs),
                "dir_err_med": round(float(np.median(dirs)), 1) if len(dirs) else None,
                "dir_bad_gt30deg": int((dirs > 30).sum())}
        report[f"{d.parent.name}/{d.name}"] = {"summary": summ, "pairs": rows}
        print(f"{d.parent.name}/{d.name}", json.dumps(summ, ensure_ascii=False), flush=True)
    out = Path(__file__).resolve().parent / "out" / "ar_consistency.json"
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
