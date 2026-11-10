"""RoomFit サーバー。復元ジョブの受付・状態・結果の配信と、Web 版の配信。

- 室内写真は私的な情報なので、所有者トークン（端末ごとに発行）が一致する要求にだけ応える。
- 秘密情報（楽天アプリIDなど）は環境変数から読み、クライアントへは渡さない。
- ログへ画像・トークンは書かない。
"""
from __future__ import annotations

import io
import json
import os
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from .jobs import PIPELINE_VERSION, JobStore

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("ROOMFIT_DATA_DIR", ROOT / "data"))
WEB_DIST = Path(os.environ.get("ROOMFIT_WEB_DIST", ROOT.parent / "web" / "dist"))
MIN_PHOTOS, MAX_PHOTOS = 3, int(os.environ.get("ROOMFIT_MAX_PHOTOS", "40"))
MAX_PHOTO_BYTES = 25 * 1024 * 1024
MAX_CAPTURE_BYTES = 400 * 1024 * 1024

app = FastAPI(title="RoomFit", docs_url="/api/docs", openapi_url="/api/openapi.json")
store = JobStore(DATA_DIR)


def _owner(x_owner: str | None) -> str:
    if not x_owner or not (16 <= len(x_owner) <= 80):
        raise HTTPException(401, "この端末の識別情報がありません。ページを読み込み直してください。")
    return x_owner


def _recon_available() -> tuple[bool, str | None]:
    try:
        import importlib.util
        if importlib.util.find_spec("torch") is None or importlib.util.find_spec("mapanything") is None:
            return False, "復元用のライブラリが入っていません（セットアップ手順の uv sync を実行してください）。"
        return True, None
    except Exception as e:  # pragma: no cover
        return False, str(e)


@app.get("/api/health")
def health():
    ok, why = _recon_available()
    return {
        "ok": True,
        "recon_available": ok,
        "recon_unavailable_reason": why,
        "pipeline": PIPELINE_VERSION,
        "min_photos": MIN_PHOTOS,
        "max_photos": MAX_PHOTOS,
        "products": {"rakuten": bool(os.environ.get("RAKUTEN_APP_ID"))},
        # 写真の送信先の説明に使う。外部の AI サービスへは送らない構成。
        "photo_destination": "このサーバー（RoomFit を動かしているPC）内だけで処理・保存します。外部のサービスへは送信しません。",
    }


@app.post("/api/jobs")
async def create_job(photos: list[UploadFile] = File(...), x_owner: str | None = Header(default=None)):
    owner = _owner(x_owner)
    ok, why = _recon_available()
    if not ok:
        raise HTTPException(503, why)
    if not (MIN_PHOTOS <= len(photos) <= MAX_PHOTOS):
        raise HTTPException(400, f"写真は {MIN_PHOTOS}〜{MAX_PHOTOS} 枚で送ってください（現在 {len(photos)} 枚）。")
    files, broken = [], []
    for up in photos:
        data = await up.read()
        name = up.filename or "photo.jpg"
        if len(data) > MAX_PHOTO_BYTES:
            broken.append(f"{name}（大きすぎます）")
            continue
        try:
            with Image.open(io.BytesIO(data)) as im:
                im.verify()
            with Image.open(io.BytesIO(data)) as im:
                im.load()
                if min(im.size) < 200:
                    broken.append(f"{name}（小さすぎます）")
                    continue
        except Exception:
            broken.append(f"{name}（画像として開けません）")
            continue
        files.append((name, data))
    if broken:
        raise HTTPException(400, {"message": "使えない写真があります。外してからもう一度お試しください。", "files": broken})
    job_id, reused = store.create(owner, files)
    return {"id": job_id, "reused": reused}


@app.post("/api/jobs/capture")
async def create_capture_job(capture: UploadFile = File(...), x_owner: str | None = Header(default=None)):
    """撮影アプリ（ARCore）の記録フォルダを zip にしたもの（NNN.jpg / NNN.json）を受け付ける。AR の位置を入力に使う（roomfit.ar_views）。"""
    owner = _owner(x_owner)
    ok, why = _recon_available()
    if not ok:
        raise HTTPException(503, why)
    data = await capture.read()
    if len(data) > MAX_CAPTURE_BYTES:
        raise HTTPException(400, "記録が大きすぎます。1 回の記録だけを送ってください。")
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            names = {Path(n).name for n in z.namelist()}
    except zipfile.BadZipFile:
        raise HTTPException(400, "記録を zip として開けません。")
    n_frames = sum(1 for n in names if n.endswith(".json") and n[:3].isdigit() and n.replace(".json", ".jpg") in names)
    if n_frames < MIN_PHOTOS:
        raise HTTPException(400, f"写真のある場面が {MIN_PHOTOS} 枚以上必要です（現在 {n_frames} 枚）。")
    job_id, reused = store.create(owner, [("capture.zip", data)])
    return {"id": job_id, "reused": reused, "frames": n_frames}


def _job_or_404(owner: str, job_id: str) -> dict:
    job = store.get(owner, job_id)
    if not job:
        raise HTTPException(404, "この復元処理は見つかりません。削除されたか、別の端末で作られたものです。")
    return job


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str, x_owner: str | None = Header(default=None)):
    return _job_or_404(_owner(x_owner), job_id)


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str, x_owner: str | None = Header(default=None)):
    owner = _owner(x_owner)
    _job_or_404(owner, job_id)
    return {"ok": store.cancel(owner, job_id)}


@app.post("/api/jobs/{job_id}/retry")
def retry_job(job_id: str, x_owner: str | None = Header(default=None)):
    owner = _owner(x_owner)
    _job_or_404(owner, job_id)
    if not store.retry(owner, job_id):
        raise HTTPException(409, "この処理はやり直せない状態です。")
    return {"ok": True}


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str, x_owner: str | None = Header(default=None)):
    owner = _owner(x_owner)
    _job_or_404(owner, job_id)
    return {"ok": store.delete(owner, job_id)}


@app.delete("/api/jobs")
def delete_all_jobs(x_owner: str | None = Header(default=None)):
    return {"deleted": store.delete_all(_owner(x_owner))}


def _result_file(owner: str, job_id: str, name: str) -> Path:
    job = _job_or_404(owner, job_id)
    if job["state"] != "done":
        raise HTTPException(409, "復元がまだ終わっていません。")
    p = store.job_dir(job_id) / name
    if not p.exists():
        raise HTTPException(404, "結果のファイルがありません。")
    return p


@app.get("/api/jobs/{job_id}/room.json")
def job_room(job_id: str, x_owner: str | None = Header(default=None)):
    return JSONResponse(json.loads(_result_file(_owner(x_owner), job_id, "room.json").read_text(encoding="utf-8")))


@app.get("/api/jobs/{job_id}/cloud.bin")
def job_cloud(job_id: str, x_owner: str | None = Header(default=None)):
    return FileResponse(_result_file(_owner(x_owner), job_id, "cloud.bin"), media_type="application/octet-stream")


@app.get("/api/jobs/{job_id}/views/{name}")
def job_view(job_id: str, name: str, x_owner: str | None = Header(default=None)):
    """写真の上で測るためのデータ（NNN.jpg / NNN.xyz）。"""
    import re
    if not re.fullmatch(r"\d{3}\.(jpg|xyz)", name):
        raise HTTPException(404, "ファイルがありません。")
    p = _result_file(_owner(x_owner), job_id, f"views/{name}")
    return FileResponse(p, media_type="image/jpeg" if name.endswith(".jpg") else "application/octet-stream")


@app.get("/api/jobs/{job_id}/surfaces/{name}")
def job_surface(job_id: str, name: str, x_owner: str | None = Header(default=None)):
    """壁・床に写真を貼った画像（wall_<辺>.jpg / floor.jpg）。"""
    import re
    if not re.fullmatch(r"(wall_\d{1,3}|floor)\.jpg", name):
        raise HTTPException(404, "ファイルがありません。")
    return FileResponse(_result_file(_owner(x_owner), job_id, f"surfaces/{name}"), media_type="image/jpeg")


@app.get("/api/products/search")
def product_search(q: str, page: int = 1):
    """楽天市場 商品検索API。アプリIDが未設定なら使えないことを正直に返す。"""
    app_id = os.environ.get("RAKUTEN_APP_ID")
    if not app_id:
        raise HTTPException(503, "商品検索は未設定です（サーバーに楽天のアプリIDが登録されていません）。URL と寸法を自分で入力して登録できます。")
    params = {"applicationId": app_id, "keyword": q, "format": "json", "formatVersion": 2, "hits": 20, "page": page, "imageFlag": 1}
    if os.environ.get("RAKUTEN_AFFILIATE_ID"):
        params["affiliateId"] = os.environ["RAKUTEN_AFFILIATE_ID"]
    url = "https://app.rakuten.co.jp/services/api/IchibaItem/Search/20220601?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            data = json.loads(r.read().decode("utf-8"))
    except Exception:
        raise HTTPException(502, "楽天市場から商品情報を取得できませんでした。時間をおいてお試しください。")
    now = datetime.now(timezone.utc).isoformat()
    items = []
    for it in data.get("Items", []):
        items.append({
            "title": it.get("itemName"),
            "price": it.get("itemPrice"),
            "shop": it.get("shopName"),
            "url": it.get("affiliateUrl") or it.get("itemUrl"),
            "affiliate": bool(it.get("affiliateUrl")),
            "image": (it.get("mediumImageUrls") or [None])[0],
            "caption": (it.get("itemCaption") or "")[:1500],
            "fetchedAt": now,
        })
    return {"items": items, "count": data.get("count", 0), "source": "楽天市場 商品検索API"}


if WEB_DIST.exists():
    app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")
