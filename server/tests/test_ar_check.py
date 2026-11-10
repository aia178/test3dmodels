"""AR の位置の検算（roomfit.ar_check）のテスト。作り物のカメラと対応点、作り物の組の比較結果を使う（精度の証明ではない）。"""
import json

import numpy as np
import pytest

from roomfit import ar_check
from roomfit.ar_check import compare_pair, decide_segments
from roomfit.arcore_import import CaptureError, load_capture

K = np.array([[800.0, 0, 480], [0, 800.0, 270], [0, 0, 1]])


def _cam(pos, yaw_deg=0.0):
    """部屋の pos から +z 向き（OpenCV の向き）を見るカメラ。y 軸まわりに yaw_deg 回す。"""
    a = np.radians(yaw_deg)
    c2w = np.eye(4)
    c2w[:3, :3] = [[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]
    c2w[:3, 3] = pos
    return c2w


def _project(c2w, P):
    pc = (P - c2w[:3, 3]) @ c2w[:3, :3]
    uv = pc @ K.T
    return uv[:, :2] / uv[:, 2:3]


def _pair(a, b, dir_err=None, rot=0.5, status="ok", dt=1.0, move=0.15):
    r = {"a": a, "b": b, "status": status, "dt": dt, "move_m": move}
    if status == "ok":
        r["rot_err_deg"] = rot
        if dir_err is not None:
            r["dir_err_deg"] = dir_err
    return r


def _pairs(n, bad=(), dir_bad=160.0):
    """1 つ先・2 つ先の組。bad に入ったコマに触れる組は、動いた方向が食い違う。"""
    return [_pair(i, j, dir_bad if (i in bad or j in bad) else 3.0) for i in range(n) for j in (i + 1, i + 2) if j < n]


def test_compare_pair_agrees_with_correct_ar_and_flags_reversed_motion():
    rng = np.random.default_rng(0)
    P = np.c_[rng.uniform(-1.5, 1.5, 300), rng.uniform(-1, 1, 300), rng.uniform(2.5, 4.0, 300)]
    a, b = _cam([0, 0, 0]), _cam([0.3, 0, 0.05], yaw_deg=4)
    pa, pb = _project(a, P), _project(b, P)
    ok = compare_pair(a, b, K, pa, pb)
    assert ok["status"] == "ok" and ok["rot_err_deg"] < 0.5 and ok["dir_err_deg"] < 3
    # AR が移動の向きを逆に報告した（写真は同じ）
    b_rev = b.copy(); b_rev[:3, 3] = 2 * a[:3, 3] - b[:3, 3]
    rev = compare_pair(a, b_rev, K, pa, pb)
    assert rev["dir_err_deg"] > 150


def test_all_consistent_uses_every_frame():
    d = decide_segments(_pairs(10), 10)
    assert d["status"] == "ok" and all(d["use_pose"])


def test_reversed_section_is_cut_and_longest_good_section_is_used():
    d = decide_segments(_pairs(14, bad={4, 5}), 14)
    assert d["status"] == "partial"
    assert d["use_pose"] == [False] * 6 + [True] * 8          # 6〜13 の 8 コマ（0〜3 より長い）
    assert sum(g["usable"] for g in d["segments"]) == 2       # 0〜3 も使えるが、区間どうしは混ぜない


def test_position_jump_cuts_even_without_photo_evidence():
    pairs = _pairs(10)
    for r in pairs:
        if r["a"] <= 6 < r["b"]:
            r.update(status="few_matches", move_m=3.0)
            r.pop("rot_err_deg", None); r.pop("dir_err_deg", None)
    d = decide_segments(pairs, 10, positions=np.c_[np.arange(10) * 0.1, np.zeros(10), np.zeros(10)])
    assert d["use_pose"] == [True] * 7 + [False] * 3
    assert d["extent_m"] == pytest.approx(0.6)
    assert d["scale_cue"] == "ok"
    small = decide_segments(_pairs(10), 10, positions=np.c_[np.arange(10) * 0.02, np.zeros(10), np.zeros(10)])
    assert small["scale_cue"] == "weak"                         # 18cm しか動いていない: 大きさの手がかりは弱い


def test_everywhere_inconsistent_is_rejected_and_unjudged_is_unchecked():
    assert decide_segments(_pairs(8, bad=set(range(8))), 8)["status"] == "rejected"
    d = decide_segments([_pair(i, i + 1, status="few_matches") for i in range(7)], 8)
    assert d["status"] == "unchecked" and not any(d["use_pose"])


def _write_capture(folder, n=5):
    for i in range(n):
        j = {"id": f"{i:03d}", "timestampNs": int(i * 1e9), "t": [0.1 * i, 1.5, 0.0], "q": [0, 0, 0, 1],
             "fx": 1280.0, "fy": 1280.0, "cx": 960.0, "cy": 540.0, "w": 1920, "h": 1080, "depthW": 16, "depthH": 9}
        (np.full((9, 16), 2000, "<u2")).tofile(folder / f"{i:03d}.depth16")
        (folder / f"{i:03d}.json").write_text(json.dumps(j), encoding="utf-8")


def test_load_capture_keeps_only_checked_frames(tmp_path, monkeypatch):
    _write_capture(tmp_path)
    monkeypatch.setattr(ar_check, "check_capture", lambda f: {"status": "partial", "use_pose": [False, True, True, True, False],
                                                               "frame_ids": ["000", "001", "002", "003", "004"], "segments": [], "pairs": []})
    cap = load_capture(tmp_path)
    assert cap["meta"]["n_views"] == 3 and cap["meta"]["frames_skipped_pose_check"] == 2
    assert cap["meta"]["pose_check"]["status"] == "partial"


def test_load_capture_refuses_when_ar_disagrees_everywhere(tmp_path, monkeypatch):
    _write_capture(tmp_path)
    monkeypatch.setattr(ar_check, "check_capture", lambda f: {"status": "rejected", "use_pose": [False] * 5,
                                                               "frame_ids": [f"{i:03d}" for i in range(5)], "segments": [], "pairs": []})
    with pytest.raises(CaptureError):
        load_capture(tmp_path)
