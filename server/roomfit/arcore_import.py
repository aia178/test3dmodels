"""RoomFit 撮影アプリ（ARCore）の記録を、点群とカメラ姿勢に変換する。

記録の形式は android/…/MainActivity.java の先頭コメントと session.json を参照。

座標の変換（ここが一番間違えやすい）:
- 記録の姿勢は「カメラ → ARCore 世界」。世界は右手系・+Y 上・単位 m。
- カメラ座標は OpenGL 規約（+X 右, +Y 上, -Z 前）。画像の画素 (u 右, v 下) と深度 z（光軸方向の距離）から
  OpenCV 規約の点 (x, y, z) = ((u-cx)/fx*z, (v-cy)/fy*z, z) を作り、(x, -y, -z) として OpenGL 規約へ移す。
- structure.py は OpenCV 規約の cam2world を受け取るので、R_cv = R_gl @ diag(1, -1, -1) を渡す。

使う深度:
- 生深度(rawdepth16)＋信頼度(conf8)＋その時刻の姿勢(rawT/rawQ)がそろっていて、姿勢の時刻差が小さいものを優先する。
- それが無い記録では、平滑化された深度(depth16)と写真の姿勢を使う（写真と同じ時刻）。平滑化された深度は
  見えていない所も埋めるため、壁の位置の根拠としては弱い。どちらを使ったかを meta に残す。

AR の位置の検算（roomfit.ar_check）:
- ARCore は「追跡中」のまま位置が飛んだり、移動の向きを取り違えたりする。食い違う位置で深度を重ねると壁が二重になるので、
  写真どうしの見え方と合う一番長い区間のコマだけを使う。どこも合わなければ取り込まない。検算できなければ全部使い、meta に残す。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

MIN_DEPTH_M, MAX_DEPTH_M = 0.3, 5.0
RAW_MIN_CONF = 128            # 信頼度の低い画素を捨てるしきい値。誤差の大きさへは換算しない。
RAW_MAX_POSE_DIFF_MS = 20.0
MAX_SPEED_MPS = 1.0           # これより速く動いた直後の場面は、位置が飛んでいる可能性があるので使わない


class CaptureError(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


def quat_to_rot(q) -> np.ndarray:
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


GL_FROM_CV = np.diag([1.0, -1.0, -1.0])


def unproject(depth_m: np.ndarray, fx: float, fy: float, cx: float, cy: float, img_w: int, img_h: int) -> tuple[np.ndarray, np.ndarray]:
    """深度画像（写真と同じ向き・低解像度）を OpenCV 規約のカメラ座標の点へ。戻り値 (点 Nx3, 使った画素のマスク)。"""
    dh, dw = depth_m.shape
    sx, sy = dw / img_w, dh / img_h
    u, v = np.meshgrid(np.arange(dw) + 0.5, np.arange(dh) + 0.5)
    ok = (depth_m > MIN_DEPTH_M) & (depth_m < MAX_DEPTH_M)
    z = depth_m[ok]
    x = (u[ok] - cx * sx) / (fx * sx) * z
    y = (v[ok] - cy * sy) / (fy * sy) * z
    return np.stack([x, y, z], axis=1), ok


def load_capture(folder: Path, check_poses: bool = True) -> dict:
    frames = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(folder.glob("[0-9][0-9][0-9].json"))]
    if len(frames) < 3:
        raise CaptureError("記録された場面が少なすぎます。3 枚以上になるように、ゆっくり向きを変えながら記録してください。")
    keep, pose_check = None, None
    if check_poses:
        from .ar_check import check_capture

        chk = check_capture(folder)
        pose_check = {k: v for k, v in chk.items() if k not in ("pairs", "use_pose", "frame_ids")}
        if chk["status"] == "rejected":
            raise CaptureError("スマホの AR の位置が、写真の見え方と合いませんでした（端末が移動の向きを取り違えている可能性があります）。"
                               "家具や模様の多い所を画面に入れて、ゆっくり記録し直してください。")
        if chk["status"] != "unchecked":
            keep = {fid for fid, u in zip(chk["frame_ids"], chk["use_pose"]) if u}
    pts, cols, conf, cam_idx, poses = [], [], [], [], []
    used_raw = used_smooth = skipped_fast = skipped_nodepth = skipped_check = 0
    prev_t, prev_ts = None, None
    for f in frames:
        if keep is not None and f["id"] not in keep:
            skipped_check += 1
            continue
        t = np.array(f["t"], float)
        ts = f["timestampNs"] / 1e9
        fast = prev_t is not None and ts > prev_ts and np.linalg.norm(t - prev_t) / (ts - prev_ts) > MAX_SPEED_MPS and (ts - prev_ts) < 2.0
        prev_t, prev_ts = t, ts
        if fast:
            skipped_fast += 1
            continue
        fid = f["id"]
        raw_p, conf_p, dep_p = folder / f"{fid}.rawdepth16", folder / f"{fid}.conf8", folder / f"{fid}.depth16"
        depth = cmask = kind = None
        pose_t, pose_q = t, f["q"]
        if raw_p.exists() and conf_p.exists() and "rawT" in f and f.get("rawPoseDiffMs", 1e9) <= RAW_MAX_POSE_DIFF_MS:
            depth = np.fromfile(raw_p, dtype="<u2").reshape(f["rawH"], f["rawW"]).astype(np.float64) / 1000.0
            cmask = np.fromfile(conf_p, dtype=np.uint8).reshape(f["rawH"], f["rawW"])
            depth = np.where(cmask >= RAW_MIN_CONF, depth, 0.0)
            pose_t, pose_q = np.array(f["rawT"], float), f["rawQ"]
            kind = "raw"
        elif dep_p.exists() and "depthW" in f:
            depth = np.fromfile(dep_p, dtype="<u2").reshape(f["depthH"], f["depthW"]).astype(np.float64) / 1000.0
            kind = "smooth"
        if depth is None or not (depth > 0).any():
            skipped_nodepth += 1
            continue
        p_cv, ok = unproject(depth, f["fx"], f["fy"], f["cx"], f["cy"], f["w"], f["h"])
        if len(p_cv) == 0:
            skipped_nodepth += 1
            continue
        # 実際に点として使えた場面だけを数える
        used_raw += kind == "raw"
        used_smooth += kind == "smooth"
        R_cv = quat_to_rot(pose_q) @ GL_FROM_CV
        pw = p_cv @ R_cv.T + pose_t
        jpg = folder / f"{fid}.jpg"
        if jpg.exists():
            with Image.open(jpg) as im:
                small = np.asarray(im.convert("RGB").resize((depth.shape[1], depth.shape[0]), Image.BILINEAR))
            c = small[ok]
        else:
            c = np.full((len(pw), 3), 150, np.uint8)
        P = np.eye(4)
        P[:3, :3], P[:3, 3] = R_cv, pose_t
        pts.append(pw.astype(np.float32))
        cols.append(c.astype(np.uint8))
        conf.append((cmask[ok] if cmask is not None else np.full(len(pw), 255, np.uint8)).astype(np.float32))
        cam_idx.append(np.full(len(pw), len(poses), np.int32))
        poses.append(P)
    if len(poses) < 3:
        raise CaptureError("使える場面が足りません（深度が取れていない、または動きが速すぎました）。ゆっくり動かして記録し直してください。")
    return {
        "points": np.concatenate(pts), "colors": np.concatenate(cols), "conf": np.concatenate(conf),
        "cam_index": np.concatenate(cam_idx), "cam2world": np.stack(poses),
        "meta": {
            "model": "arcore-depth", "device": "phone", "n_views": len(poses), "n_frames_recorded": len(frames),
            "frames_raw_depth": used_raw, "frames_smoothed_depth": used_smooth,
            "frames_skipped_too_fast": skipped_fast, "frames_skipped_no_depth": skipped_nodepth,
            "frames_skipped_pose_check": skipped_check, "pose_check": pose_check,
            "scale_source": "ARCore の姿勢（端末の動きから推定した実寸）。実測での確認は別に必要。",
        },
    }


ARCORE_UP = np.array([0.0, 1.0, 0.0])
