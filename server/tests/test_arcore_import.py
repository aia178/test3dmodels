"""撮影アプリの記録 → 世界座標の点、の座標変換のテスト（計算で作った記録を使う。精度の証明ではない）。"""
import json

import numpy as np
import pytest

from roomfit.arcore_import import CaptureError, load_capture, quat_to_rot

W, H, DW, DH = 1920, 1080, 160, 90
FX = FY = 1280.0
CX, CY = 960.0, 540.0


def _write_frame(folder, idx, t, q, depth_m, ts, raw=False, raw_pose_diff_ms=0.0, conf=255):
    fid = f"{idx:03d}"
    j = {"id": fid, "timestampNs": int(ts * 1e9), "t": list(t), "q": list(q), "fx": FX, "fy": FY, "cx": CX, "cy": CY, "w": W, "h": H}
    d = (np.asarray(depth_m) * 1000).astype("<u2")
    if raw:
        d.tofile(folder / f"{fid}.rawdepth16")
        np.full((DH, DW), conf, np.uint8).tofile(folder / f"{fid}.conf8")
        j.update({"rawW": DW, "rawH": DH, "rawT": list(t), "rawQ": list(q), "rawPoseDiffMs": raw_pose_diff_ms})
    else:
        d.tofile(folder / f"{fid}.depth16")
        j.update({"depthW": DW, "depthH": DH})
    (folder / f"{fid}.json").write_text(json.dumps(j), encoding="utf-8")


def test_quaternion_identity_and_yaw():
    assert np.allclose(quat_to_rot([0, 0, 0, 1]), np.eye(3))
    s = np.sin(np.pi / 4)
    R = quat_to_rot([0, s, 0, s])  # Y 軸まわり +90°
    assert np.allclose(R @ np.array([0, 0, -1.0]), [-1, 0, 0], atol=1e-9)  # 前(-Z)が -X を向く


def test_points_land_where_the_camera_looks(tmp_path):
    depth = np.full((DH, DW), 2.0)
    # 向きを変えない（OpenGL 規約: 前は -Z、上は +Y、右は +X）カメラを 3 か所に置く
    for i, x in enumerate([0.0, 0.1, 0.2]):
        _write_frame(tmp_path, i, (x, 1.5, 0.0), (0, 0, 0, 1), depth, ts=i * 1.0)
    cap = load_capture(tmp_path)
    p = cap["points"][cap["cam_index"] == 0]
    assert np.allclose(p[:, 2], -2.0, atol=1e-5), "深度 2m の面は、カメラの前方（世界の -Z）2m にある"
    assert abs(p[:, 0].mean()) < 0.02 and abs(p[:, 1].mean() - 1.5) < 0.02
    # 画像の右端は世界の +X、画像の下端は世界の低い側（-Y）。左右・上下が反転していないこと
    grid = p.reshape(DH, DW, 3)
    assert grid[DH // 2, -1, 0] > grid[DH // 2, 0, 0]
    assert grid[-1, DW // 2, 1] < grid[0, DW // 2, 1]
    # 見えている幅: 2m 先で 1920/1280*2 = 3.0m
    assert abs((grid[DH // 2, -1, 0] - grid[DH // 2, 0, 0]) - 3.0 * (DW - 1) / DW) < 0.01
    # structure.py へ渡す姿勢は OpenCV 規約: カメラの +Z（前）が世界の -Z、+Y（下）が世界の -Y
    R = cap["cam2world"][0][:3, :3]
    assert np.allclose(R @ [0, 0, 1], [0, 0, -1]) and np.allclose(R @ [0, 1, 0], [0, -1, 0])
    assert np.isclose(np.linalg.det(R), 1.0)


def test_rotated_camera(tmp_path):
    s = np.sin(np.pi / 4)
    for i in range(3):
        _write_frame(tmp_path, i, (0.0, 1.0, 0.1 * i), (0, s, 0, s), np.full((DH, DW), 1.5), ts=i * 1.0)
    cap = load_capture(tmp_path)
    p = cap["points"][cap["cam_index"] == 0]
    assert np.allclose(p[:, 0], -1.5, atol=1e-5)  # -X 方向 1.5m


def test_stale_raw_depth_is_not_used_and_fast_frames_are_skipped(tmp_path):
    good = np.full((DH, DW), 2.0)
    _write_frame(tmp_path, 0, (0, 1.5, 0), (0, 0, 0, 1), good, ts=0.0, raw=True, raw_pose_diff_ms=5)
    _write_frame(tmp_path, 1, (0.1, 1.5, 0), (0, 0, 0, 1), good, ts=1.0, raw=True, raw_pose_diff_ms=900)  # 姿勢と時刻が合わない
    _write_frame(tmp_path, 2, (0.2, 1.5, 0), (0, 0, 0, 1), good, ts=2.0, raw=True, raw_pose_diff_ms=5, conf=10)  # 信頼度が低い
    _write_frame(tmp_path, 3, (3.0, 1.5, 0), (0, 0, 0, 1), good, ts=2.2)  # 0.2 秒で 2.8m 動いた = 位置が飛んでいる
    _write_frame(tmp_path, 4, (3.05, 1.5, 0), (0, 0, 0, 1), good, ts=3.2)
    _write_frame(tmp_path, 5, (3.1, 1.5, 0), (0, 0, 0, 1), good, ts=4.2)
    cap = load_capture(tmp_path)
    m = cap["meta"]
    assert m["frames_raw_depth"] == 1 and m["frames_skipped_too_fast"] == 1
    assert m["frames_skipped_no_depth"] == 2 and m["n_views"] == 3


def test_too_few_frames_gives_user_message(tmp_path):
    _write_frame(tmp_path, 0, (0, 1.5, 0), (0, 0, 0, 1), np.full((DH, DW), 2.0), ts=0.0)
    with pytest.raises(CaptureError) as e:
        load_capture(tmp_path)
    assert "記録" in e.value.message
