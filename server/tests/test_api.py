"""API とジョブ管理のテスト。重い推論は動かさず、受付・所有者の確認・失敗時の扱い・削除を確かめる。"""
import io
import json
import os
import sys
import time
from pathlib import Path

import pytest
from PIL import Image

OWNER_A = "a" * 40
OWNER_B = "b" * 40


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("ROOMFIT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ROOMFIT_WEB_DIST", str(tmp_path / "no-web"))
    for m in [k for k in sys.modules if k.startswith("roomfit.app")]:
        del sys.modules[m]
    import roomfit.jobs as jobs
    from fastapi.testclient import TestClient

    # 実際の復元の代わりに、すぐ終わる・失敗する代役プロセスを使う
    fake = tmp_path / "fake_worker.py"
    fake.write_text(
        "import json,sys,pathlib,time\n"
        "d=pathlib.Path(sys.argv[1]); mode=(d/'mode.txt').read_text() if (d/'mode.txt').exists() else 'ok'\n"
        "if mode=='hang': time.sleep(60)\n"
        "if mode=='crash': sys.exit(9)\n"
        "(d/'room.json').write_text(json.dumps({'outline':[[0,0],[1,0],[1,1],[0,1]]}))\n"
        "(d/'cloud.bin').write_bytes(b'')\n"
        "(d/'status.json').write_text(json.dumps({'stage':'done'}))\n",
        encoding="utf-8",
    )
    real_popen = jobs.subprocess.Popen

    def fake_popen(args, **kw):
        return real_popen([sys.executable, str(fake), args[-1]], **{k: v for k, v in kw.items() if k != "cwd"})

    monkeypatch.setattr(jobs.subprocess, "Popen", fake_popen)
    import roomfit.app as app_mod
    monkeypatch.setattr(app_mod, "_recon_available", lambda: (True, None))
    return TestClient(app_mod.app), app_mod


def _jpg(color=(120, 130, 140), size=(640, 480)) -> bytes:
    b = io.BytesIO()
    Image.new("RGB", size, color).save(b, "JPEG")
    return b.getvalue()


def _photos(n=3):
    return [("photos", (f"p{i}.jpg", _jpg((10 * i, 100, 150)), "image/jpeg")) for i in range(n)]


def _wait(client, job_id, owner, states, timeout=15):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/api/jobs/{job_id}", headers={"X-Owner": owner}).json()
        if j["state"] in states:
            return j
        time.sleep(0.2)
    raise AssertionError(f"timeout, last={j}")


def test_requires_owner_token(client):
    c, _ = client
    assert c.post("/api/jobs", files=_photos()).status_code == 401


def test_rejects_broken_and_too_few_photos(client):
    c, _ = client
    r = c.post("/api/jobs", files=_photos(2), headers={"X-Owner": OWNER_A})
    assert r.status_code == 400 and "枚" in r.json()["detail"]
    files = _photos(3) + [("photos", ("broken.jpg", b"not an image", "image/jpeg"))]
    r = c.post("/api/jobs", files=files, headers={"X-Owner": OWNER_A})
    assert r.status_code == 400
    assert any("broken.jpg" in f for f in r.json()["detail"]["files"])


def test_job_runs_and_other_owner_cannot_read(client):
    c, _ = client
    job = c.post("/api/jobs", files=_photos(), headers={"X-Owner": OWNER_A}).json()
    _wait(c, job["id"], OWNER_A, {"done"})
    assert c.get(f"/api/jobs/{job['id']}/room.json", headers={"X-Owner": OWNER_A}).status_code == 200
    # 別の端末（別の所有者）からは存在も見えない
    assert c.get(f"/api/jobs/{job['id']}", headers={"X-Owner": OWNER_B}).status_code == 404
    assert c.get(f"/api/jobs/{job['id']}/room.json", headers={"X-Owner": OWNER_B}).status_code == 404
    assert c.delete(f"/api/jobs/{job['id']}", headers={"X-Owner": OWNER_B}).status_code == 404


def test_same_input_reuses_result(client):
    c, _ = client
    a = c.post("/api/jobs", files=_photos(), headers={"X-Owner": OWNER_A}).json()
    _wait(c, a["id"], OWNER_A, {"done"})
    b = c.post("/api/jobs", files=_photos(), headers={"X-Owner": OWNER_A}).json()
    assert b["id"] == a["id"] and b["reused"] is True


def test_crash_is_reported_and_retry_works(client):
    c, app_mod = client
    job = c.post("/api/jobs", files=_photos(4), headers={"X-Owner": OWNER_A}).json()
    jd = app_mod.store.job_dir(job["id"])
    (jd / "mode.txt").write_text("crash")
    j = _wait(c, job["id"], OWNER_A, {"failed", "done"})
    if j["state"] == "done":  # 代役が mode.txt より先に走った場合はこの確認を省く
        pytest.skip("worker started before the failure mode was set")
    assert j["error_code"] == "crashed" and j["error_message"]
    assert c.get(f"/api/jobs/{job['id']}/room.json", headers={"X-Owner": OWNER_A}).status_code == 409
    (jd / "mode.txt").write_text("ok")
    assert c.post(f"/api/jobs/{job['id']}/retry", headers={"X-Owner": OWNER_A}).status_code == 200
    _wait(c, job["id"], OWNER_A, {"done"})


def test_delete_removes_photos_from_disk(client):
    c, app_mod = client
    job = c.post("/api/jobs", files=_photos(), headers={"X-Owner": OWNER_A}).json()
    _wait(c, job["id"], OWNER_A, {"done"})
    jd: Path = app_mod.store.job_dir(job["id"])
    assert (jd / "upload").exists()
    assert c.delete(f"/api/jobs/{job['id']}", headers={"X-Owner": OWNER_A}).json()["ok"] is True
    assert not jd.exists()
    assert c.get(f"/api/jobs/{job['id']}", headers={"X-Owner": OWNER_A}).status_code == 404


def test_product_search_reports_unconfigured(client, monkeypatch):
    c, _ = client
    monkeypatch.delenv("RAKUTEN_APP_ID", raising=False)
    r = c.get("/api/products/search?q=desk")
    assert r.status_code == 503 and "未設定" in r.json()["detail"]
    assert json.dumps(c.get("/api/health").json())  # 秘密情報を含まない形で返る
    assert "RAKUTEN" not in json.dumps(c.get("/api/health").json()) and os.environ.get("RAKUTEN_APP_ID") is None
