"""復元ジョブの記録と実行。状態は SQLite とジョブごとのフォルダに残り、サーバーを再起動しても消えない。"""
from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

PIPELINE_VERSION = "mapanything-apache/structure-v2"   # v2: 17 枚以上は範囲に分けて復元してつなぐ（写真の順番に意味がある）
JOB_TIMEOUT_SEC = 3 * 60 * 60                           # 40 枚（4 範囲）で約 35 分。余裕を見て 3 時間
ACTIVE = ("queued", "running")


def owner_hash(owner: str) -> str:
    return hashlib.sha256(owner.encode()).hexdigest()


class JobStore:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.jobs_dir = data_dir / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = data_dir / "roomfit.sqlite3"
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._proc_job: str | None = None
        with self._db() as db:
            db.execute(
                """CREATE TABLE IF NOT EXISTS jobs(
                    id TEXT PRIMARY KEY, owner TEXT NOT NULL, state TEXT NOT NULL, input_hash TEXT NOT NULL,
                    pipeline TEXT NOT NULL, n_photos INTEGER NOT NULL, created_at REAL NOT NULL,
                    started_at REAL, finished_at REAL, cancel_requested INTEGER NOT NULL DEFAULT 0)"""
            )
            # 前回の起動中に実行中だったものは、中断として記録する（勝手に成功扱いにしない）
            db.execute("UPDATE jobs SET state='interrupted', finished_at=? WHERE state='running'", (time.time(),))
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _db(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.db_path, isolation_level=None)
        db.row_factory = sqlite3.Row
        return db

    def job_dir(self, job_id: str) -> Path:
        return self.jobs_dir / job_id

    # ------------------------------------------------------------ 操作
    def create(self, owner: str, files: list[tuple[str, bytes]]) -> tuple[str, bool]:
        """戻り値 (job_id, 既存の結果を再利用したか)。"""
        h = hashlib.sha256()
        for digest in (hashlib.sha256(b).hexdigest() for _, b in files):   # 順番も入力の一部（範囲の分け方が変わる）
            h.update(digest.encode())
        h.update(PIPELINE_VERSION.encode())
        input_hash = h.hexdigest()
        oh = owner_hash(owner)
        with self._lock, self._db() as db:
            row = db.execute(
                "SELECT id FROM jobs WHERE owner=? AND input_hash=? AND state IN ('done','queued','running') ORDER BY created_at DESC",
                (oh, input_hash),
            ).fetchone()
            if row and self.job_dir(row["id"]).exists():
                return row["id"], True
            job_id = uuid.uuid4().hex
            up = self.job_dir(job_id) / "upload"
            up.mkdir(parents=True)
            for i, (name, data) in enumerate(files):
                ext = Path(name).suffix.lower() if Path(name).suffix.lower() in (".jpg", ".jpeg", ".png", ".webp") else ".jpg"
                (up / f"{i:03d}{ext}").write_bytes(data)
            db.execute(
                "INSERT INTO jobs(id, owner, state, input_hash, pipeline, n_photos, created_at) VALUES(?,?,?,?,?,?,?)",
                (job_id, oh, "queued", input_hash, PIPELINE_VERSION, len(files), time.time()),
            )
            return job_id, False

    def get(self, owner: str, job_id: str) -> dict | None:
        with self._db() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=? AND owner=?", (job_id, owner_hash(owner))).fetchone()
        if not row:
            return None
        out = {k: row[k] for k in ("id", "state", "pipeline", "n_photos", "created_at", "started_at", "finished_at")}
        st = self._read_status(job_id)
        out["stage"] = st.get("stage") if row["state"] in ("running", "done", "failed") else None
        if row["state"] == "running" and st.get("chunks"):   # 範囲に分けて復元しているとき: 何範囲中の何番目か
            out["chunk"], out["chunks"] = st.get("chunk"), st.get("chunks")
        if row["state"] == "failed":
            out["error_code"] = st.get("error_code", "internal")
            out["error_message"] = st.get("error_message", "処理に失敗しました。")
            out["error_detail"] = st.get("error_detail")
        if row["state"] == "queued":
            with self._db() as db:
                out["queue_ahead"] = db.execute(
                    "SELECT COUNT(*) c FROM jobs WHERE state IN ('queued','running') AND created_at<?", (row["created_at"],)
                ).fetchone()["c"]
        return out

    def cancel(self, owner: str, job_id: str) -> bool:
        with self._lock, self._db() as db:
            row = db.execute("SELECT state FROM jobs WHERE id=? AND owner=?", (job_id, owner_hash(owner))).fetchone()
            if not row or row["state"] not in ACTIVE:
                return False
            if row["state"] == "queued":
                db.execute("UPDATE jobs SET state='cancelled', finished_at=? WHERE id=?", (time.time(), job_id))
            else:
                db.execute("UPDATE jobs SET cancel_requested=1 WHERE id=?", (job_id,))
            return True

    def retry(self, owner: str, job_id: str) -> bool:
        with self._lock, self._db() as db:
            row = db.execute("SELECT state FROM jobs WHERE id=? AND owner=?", (job_id, owner_hash(owner))).fetchone()
            if not row or row["state"] in ACTIVE or row["state"] == "done":
                return False
            if not (self.job_dir(job_id) / "upload").exists() and not (self.job_dir(job_id) / "raw.npz").exists():
                return False
            st = self.job_dir(job_id) / "status.json"
            if st.exists():
                st.unlink()
            db.execute(
                "UPDATE jobs SET state='queued', cancel_requested=0, started_at=NULL, finished_at=NULL, created_at=? WHERE id=?",
                (time.time(), job_id),
            )
            return True

    def delete(self, owner: str, job_id: str) -> bool:
        """写真・復元結果・記録をすべて消す。"""
        with self._lock, self._db() as db:
            row = db.execute("SELECT state FROM jobs WHERE id=? AND owner=?", (job_id, owner_hash(owner))).fetchone()
            if not row:
                return False
            if row["state"] == "running":
                db.execute("UPDATE jobs SET cancel_requested=1 WHERE id=?", (job_id,))
        if row["state"] == "running":
            for _ in range(50):
                time.sleep(0.2)
                with self._db() as db:
                    if db.execute("SELECT state FROM jobs WHERE id=?", (job_id,)).fetchone()["state"] != "running":
                        break
        shutil.rmtree(self.job_dir(job_id), ignore_errors=True)
        with self._db() as db:
            db.execute("DELETE FROM jobs WHERE id=?", (job_id,))
        return True

    def delete_all(self, owner: str) -> int:
        with self._db() as db:
            ids = [r["id"] for r in db.execute("SELECT id FROM jobs WHERE owner=?", (owner_hash(owner),))]
        return sum(1 for i in ids if self.delete(owner, i))

    # ------------------------------------------------------------ 実行
    def _read_status(self, job_id: str) -> dict:
        p = self.job_dir(job_id) / "status.json"
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _loop(self) -> None:
        while True:
            try:
                self._tick()
            except Exception:
                pass
            time.sleep(0.5)

    def _tick(self) -> None:
        with self._db() as db:
            row = db.execute("SELECT id FROM jobs WHERE state='queued' ORDER BY created_at LIMIT 1").fetchone()
        if not row:
            return
        job_id = row["id"]
        jd = self.job_dir(job_id)
        with self._db() as db:
            db.execute("UPDATE jobs SET state='running', started_at=? WHERE id=? AND state='queued'", (time.time(), job_id))
        log = open(jd / "worker.log", "ab")
        proc = subprocess.Popen(
            [sys.executable, "-m", "roomfit.worker", str(jd)],
            cwd=str(Path(__file__).resolve().parents[1]), stdout=log, stderr=subprocess.STDOUT,
        )
        start = time.time()
        final = None
        while proc.poll() is None:
            time.sleep(0.5)
            with self._db() as db:
                r = db.execute("SELECT cancel_requested FROM jobs WHERE id=?", (job_id,)).fetchone()
            if r is None or r["cancel_requested"]:
                proc.kill()
                final = "cancelled"
                break
            if time.time() - start > JOB_TIMEOUT_SEC:
                proc.kill()
                final = "failed"
                (jd / "status.json").write_text(json.dumps({
                    "stage": "failed", "error_code": "timeout",
                    "error_message": "処理に時間がかかりすぎたため中止しました。写真の枚数を減らしてお試しください。",
                }, ensure_ascii=False), encoding="utf-8")
                break
        proc.wait()
        log.close()
        if final is None:
            st = self._read_status(job_id)
            if proc.returncode == 0 and st.get("stage") == "done":
                final = "done"
            else:
                final = "failed"
                if st.get("stage") != "failed":
                    # 例外も出さずに落ちた（メモリ不足で OS に止められた等）
                    (jd / "status.json").write_text(json.dumps({
                        "stage": "failed", "error_code": "crashed",
                        "error_message": "処理が途中で止まりました。メモリ不足の可能性があります。写真の枚数を減らしてお試しください。",
                    }, ensure_ascii=False), encoding="utf-8")
        with self._db() as db:
            db.execute("UPDATE jobs SET state=?, finished_at=? WHERE id=?", (final, time.time(), job_id))
