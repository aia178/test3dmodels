"""撮影アプリの記録を MapAnything の入力にする処理（roomfit.ar_views）と、その受付・ジョブの流れのテスト。作り物の記録を使う（精度の証明ではない）。"""
import io
import json
import zipfile

import numpy as np
import pytest
from PIL import Image

from roomfit import ar_views, worker
from roomfit.ar_views import extract_capture, plan_views
from roomfit.arcore_import import GL_FROM_CV, CaptureError, quat_to_rot

W, H = 640, 480
FX, FY, CX, CY = 480.0, 470.0, 322.0, 236.0


def _frame(i: int, x: float) -> dict:
    return {"id": f"{i:03d}", "timestampNs": int(i * 5e8), "tracking": "TRACKING", "t": [x, 1.4, 0.0], "q": [0.0, 0.0, 0.0, 1.0],
            "fx": FX, "fy": FY, "cx": CX, "cy": CY, "w": W, "h": H}


def _capture(folder, n=6, step=0.2):
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (folder / f"{i:03d}.json").write_text(json.dumps(_frame(i, i * step)), encoding="utf-8")
        Image.new("RGB", (W, H), (20 * i, 100, 150)).save(folder / f"{i:03d}.jpg")
    return folder


def _check(trusted):
    def check(folder):
        ids = sorted(p.stem for p in folder.glob("[0-9][0-9][0-9].json"))
        return {"status": "ok", "frame_ids": ids, "use_pose": [i in trusted for i in ids], "extent_m": 1.0, "scale_cue": "ok", "pairs": []}
    return check


def _zip(files: dict) -> bytes:
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)
    return b.getvalue()


def test_extract_keeps_only_capture_files_flat(tmp_path):
    zp = tmp_path / "c.zip"
    zp.write_bytes(_zip({"rec/001.json": "{}", "rec/001.jpg": b"x", "rec/session.json": "{}", "../evil.json": "{}", "rec/notes.txt": "x"}))
    out = extract_capture(zp, tmp_path / "capture")
    assert sorted(p.name for p in out.iterdir()) == ["001.jpg", "001.json", "session.json"]
    assert not (tmp_path / "evil.json").exists()


def test_rotated_pose_and_intrinsics_project_like_the_original_photo(tmp_path):
    """縦向きに回した写真の画素 (u', v') は、元の写真の (H - v, u) になる。"""
    f = _frame(0, 0.0)
    f["q"] = [0.1, 0.2, -0.05, 0.97]
    f["q"] = list(np.array(f["q"]) / np.linalg.norm(f["q"]))
    K_rot, c2w_rot = ar_views._rotated(f)
    c2w = np.eye(4)
    c2w[:3, :3] = quat_to_rot(f["q"]) @ GL_FROM_CV
    c2w[:3, 3] = f["t"]
    P = c2w[:3, :3] @ np.array([0.3, -0.2, 2.0]) + c2w[:3, 3]

    def project(T, K):
        pc = (P - T[:3, 3]) @ T[:3, :3]
        uv = K @ pc
        return uv[:2] / uv[2]
    u, v = project(c2w, np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]]))
    u2, v2 = project(c2w_rot, K_rot)
    assert np.allclose([u2, v2], [H - v, u], atol=1e-6)


def test_positions_go_only_to_frames_that_passed_the_check(tmp_path):
    folder = _capture(tmp_path / "cap")
    plan = plan_views(folder, check=_check({"002", "003", "004"}))
    assert len(plan["ids"]) >= 3
    assert plan["posed"] == [fid in {"002", "003", "004"} for fid in plan["ids"]]
    assert "pairs" not in plan["pose_check"] and plan["pose_check"]["status"] == "ok"


@pytest.mark.parametrize("status", ["unchecked", "rejected"])
def test_no_positions_when_the_check_cannot_vouch(tmp_path, status):
    folder = _capture(tmp_path / "cap")
    chk = _check({"000", "001", "002", "003", "004", "005"})
    plan = plan_views(folder, check=lambda f: {**chk(f), "status": status})
    assert not any(plan["posed"])


def test_too_few_photos_is_a_capture_error(tmp_path):
    folder = _capture(tmp_path / "cap", n=2)
    with pytest.raises(CaptureError):
        plan_views(folder, check=_check(set()))


def test_job_from_capture_passes_positions_and_marks_scale_source(tmp_path, monkeypatch):
    job = tmp_path / "job"
    (job / "upload").mkdir(parents=True)
    src = _capture(tmp_path / "cap")
    (job / "upload" / "000.zip").write_bytes(_zip({p.name: p.read_bytes() for p in src.iterdir()}))
    monkeypatch.setattr("roomfit.ar_check.check_capture", _check({"001", "002", "003", "004"}))
    seen = {}

    def fake_run(paths, on_stage=lambda s: None, views=None, **kw):
        seen["n"], seen["posed"] = len(paths), ["camera_poses" in v for v in views]
        return {"meta": {"model": "fake"}}
    monkeypatch.setattr("roomfit.reconstruct.run_mapanything", fake_run)
    raw = worker.run_ar_capture(job / "upload" / "000.zip", job)
    assert seen["n"] == len(raw["meta"]["ar_frames"]["ids"]) and seen["posed"] == raw["meta"]["ar_frames"]["posed"]
    assert any(seen["posed"]) and not all(seen["posed"])
    assert worker.scale_source(raw["meta"]) == "ar_pose"
    assert worker.scale_source({**raw["meta"], "pose_check": {"scale_cue": "weak"}}) == "ar_pose_weak"
    assert worker.scale_source({"model": "x"}) == "model"
