"""1 件の復元ジョブを実行する別プロセス。python -m roomfit.worker <job_dir>

段階は status.json に書く。取得できない進捗率は作らない（段階だけ）。
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from .structure import StructureError, StructureParams, build_room_structure

CLOUD_MAX_POINTS = 600_000
CONF_DROP_PERCENTILE = 15  # 信頼度の低い点を捨てる割合。誤差の大きさへは換算しない。


def write_status(job_dir: Path, **kw) -> None:
    p = job_dir / "status.json"
    cur = {}
    if p.exists():
        try:
            cur = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            cur = {}
    cur.update(kw)
    cur["updated_at"] = time.time()
    tmp = job_dir / "status.json.tmp"
    tmp.write_text(json.dumps(cur, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, p)


def equalize_view_colors(points: np.ndarray, colors: np.ndarray, cam_index: np.ndarray, cell: float = 0.01) -> tuple[np.ndarray, dict]:
    """写真ごとの明るさ・色味の違いを合わせる。

    スマホは写す向きで明るさを自動で変えるため、同じ扉が写真によって明るさの違う色で写る。そのまま重ねると、
    明るい写真の点が手前に来た所が白っぽいまだらに見える（2026-10-16、元の写真にはむらが無いことを確認）。
    同じ小さな立方体に入った点を写真ごとに比べ、写真ごとの倍率（色の 3 成分それぞれ）を、ずれの中央値から決める。
    """
    key = np.floor(points / cell).astype(np.int64)
    _, vox = np.unique(key, axis=0, return_inverse=True)
    vox = vox.ravel()
    n_view = int(cam_index.max()) + 1
    logc = np.log(np.maximum(colors.astype(np.float64), 4.0))
    gains = np.zeros((n_view, 3))
    # 立方体×写真ごとの平均色
    pair = vox * n_view + cam_index
    up, pinv = np.unique(pair, return_inverse=True)
    pinv = pinv.ravel()
    cnt = np.bincount(pinv)
    mean = np.stack([np.bincount(pinv, logc[:, k]) / cnt for k in range(3)], 1)
    pv, pview = up // n_view, up % n_view
    multi = np.bincount(pv)[pv] >= 2                 # 2 枚以上の写真に写っている立方体だけ使う
    for _ in range(3):
        adj = mean - gains[pview]
        vm = np.stack([np.bincount(pv, adj[:, k]) / np.bincount(pv) for k in range(3)], 1)   # 立方体ごとの、合わせた後の平均
        r = mean - vm[pv]
        for v in range(n_view):
            m = multi & (pview == v)
            if m.sum() >= 200:
                gains[v] = np.median(r[m], axis=0)
        gains -= gains.mean(0)                           # 全体の明るさは変えない
    out = np.clip(np.exp(logc - gains[cam_index]), 0, 255).astype(np.uint8)
    return out, {"view_gain": np.round(np.exp(gains), 3).tolist()}


def voxel_downsample(points: np.ndarray, colors: np.ndarray, target: int) -> tuple[np.ndarray, np.ndarray]:
    """小さな立方体ごとに点を 1 つにまとめる。位置と色は、その立方体に入った点の平均にする。
    1 つを選ぶだけだと、写真ごとの明るさの違いや数 mm の位置のずれで、同じ面がまだら模様に見える（2026-10-16）。"""
    if len(points) <= target:
        return points, colors
    size = 0.006
    while True:
        key = np.floor(points / size).astype(np.int64)
        _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
        if len(cnt) <= target or size > 0.2:
            inv = inv.ravel()
            p = np.zeros((len(cnt), 3)); c = np.zeros((len(cnt), 3))
            np.add.at(p, inv, points.astype(np.float64))
            np.add.at(c, inv, colors.astype(np.float64))
            pts, col = (p / cnt[:, None]).astype(np.float32), c / cnt[:, None]
            return pts, majority_color(pts, col)
        size *= 1.3


def majority_color(points: np.ndarray, colors: np.ndarray, cell: float = 0.02) -> np.ndarray:
    """2cm の立方体ごとに、色の中央値（多数派の色）をとる。
    写真ごとの位置のずれで、白い壁の点が手前の扉の上に少しはみ出して重なると、扉に白い筋が出る。少数の色を目立たなくする。"""
    key = np.floor(points / cell).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    order = np.argsort(inv, kind="stable")
    bounds = np.flatnonzero(np.diff(inv[order])) + 1
    out = np.empty_like(colors)
    for grp in np.split(order, bounds):
        out[grp] = np.median(colors[grp], axis=0)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


# 質の目安のしきい値。同じ角の 7 通りの復元（docs/検証記録.md §11）で、実測とよく合ったものと合わなかったものを分けた値。
# 1 つの部屋から決めた値なので、ほかの部屋での妥当性は未確認。
Q_DISAGREE_MM = 11.0     # 写真どうしの食い違い（中央値）がこれを超えたら要注意
Q_OVERLAP = 0.85         # ほかの写真と重なる点の割合がこれ未満なら要注意
Q_FOCAL_RATIO = 1.15     # 写真ごとに推定された画角のばらつき（最大÷最小）がこれを超えたら要注意


def capture_quality(raw: dict) -> dict:
    """実測なしで、復元がどれだけ信頼できそうかの目安を出す。

    同じ場所を 2 枚以上の写真が見ているとき、それぞれの写真から出した 3D の位置は一致するはず。
    その食い違いが大きい復元は、実測との差も大きかった。これは誤差そのものではなく、撮り直しを勧めるかどうかの目安。

    スマホの AR の位置を使った復元では、raw["meta"]["pose_check"]（roomfit.ar_check の判定）も見る。位置を使った区間で
    カメラがあまり動いていないと、形はそろっても大きさの手がかりが弱い（10:35 の記録で −8.5%、docs/検証記録.md §19）ので "ar_scale" を付ける。
    """
    from scipy.spatial import cKDTree

    P, ci = raw["points"].astype(np.float64), raw["cam_index"]
    n = len(raw["cam2world"])
    rng = np.random.default_rng(0)
    med, share = [], []
    for v in range(n):
        a, o = P[ci == v], P[ci != v]
        if len(a) < 1000 or len(o) < 1000:
            continue
        a = a[rng.choice(len(a), min(len(a), 15000), replace=False)]
        o = o[rng.choice(len(o), min(len(o), 150000), replace=False)]
        d, _ = cKDTree(o).query(a)
        near = d < 0.10
        share.append(float(near.mean()))
        if near.sum() > 300:
            med.append(float(np.median(d[near])))
    fx = raw["intrinsics"][:, 0, 0] if "intrinsics" in raw else None   # AR の深度を重ねた取り込み（load_capture）には無い
    q = {
        "disagreement_mm": round(float(np.mean(med)) * 1000, 1) if med else None,
        "overlap": round(float(np.mean(share)), 3) if share else 0.0,
        "focal_ratio": round(float(fx.max() / fx.min()), 3) if fx is not None else None,
        "n_photos": int(n),
    }
    reasons = []
    if q["disagreement_mm"] is None or q["disagreement_mm"] > Q_DISAGREE_MM:
        reasons.append("disagreement")
    if q["overlap"] < Q_OVERLAP:
        reasons.append("overlap")
    if q["focal_ratio"] is not None and q["focal_ratio"] > Q_FOCAL_RATIO:
        reasons.append("focal")
    pc = raw.get("meta", {}).get("pose_check")
    if pc:
        q["ar"] = {k: pc.get(k) for k in ("status", "extent_m", "scale_cue", "pairs_judged", "pairs_bad")}
        if pc.get("scale_cue") == "weak":
            reasons.append("ar_scale")
    ch = raw.get("meta", {}).get("chunks")
    if ch:  # 範囲に分けて復元してつないだ場合: つなぎ目の合い方も見る
        from .stitch import LINK_SIZE_GAP_WARN_PCT, LINK_WARN_MM

        q["chunks"] = ch["count"]
        q["link_residual_mm"] = max((l["residual_median_mm"] for l in ch["links"]), default=None)
        # 共通の写真で見た、隣の範囲との大きさの比（1 からのずれの最大）。範囲どうしで大きさの見積もりが食い違っている目安
        q["link_size_gap_pct"] = round(max((abs(l.get("size_ratio_seen", 1.0) - 1) for l in ch["links"]), default=0.0) * 100, 1)
        if (q["link_residual_mm"] is not None and q["link_residual_mm"] > LINK_WARN_MM) or q["link_size_gap_pct"] > LINK_SIZE_GAP_WARN_PCT:
            reasons.append("stitch")
    q["level"] = "poor" if reasons else "good"
    q["reasons"] = reasons
    return q


def view_grids(raw: dict) -> list[tuple[np.ndarray, np.ndarray]]:
    """view ごとの (有効画素のマスク H×W, 世界座標 H×W×3)。raw が有効画素の情報を持つ場合だけ使える。"""
    out, bit0, pt0 = [], 0, 0
    bits = np.unpackbits(raw["valid_bits"])
    for h, w in raw["image_hw"]:
        n = int(h) * int(w)
        nbytes = (n + 7) // 8
        mask = bits[bit0:bit0 + n].astype(bool).reshape(int(h), int(w))
        bit0 += nbytes * 8
        k = int(mask.sum())
        xyz = np.full((int(h), int(w), 3), np.nan, np.float32)
        xyz[mask] = raw["points"][pt0:pt0 + k]
        pt0 += k
        out.append((mask, xyz))
    return out


XYZ_INVALID = -32768


def encode_xyz_mm(xyz_room: np.ndarray) -> bytes:
    """部屋座標 [m] の H×W×3 を、mm 単位の int16（リトルエンディアン）にする。値が無い画素は -32768。"""
    mm = np.round(xyz_room * 1000.0)
    bad = ~np.isfinite(mm).all(axis=-1) | (np.abs(mm) > 32000).any(axis=-1)
    q = np.where(bad[..., None], XYZ_INVALID, mm).astype("<i2")
    return q.tobytes()


def export_views(grids: list[tuple[np.ndarray, np.ndarray]], images: np.ndarray, T: np.ndarray, job_dir: Path) -> list[dict]:
    """写真の上で 2 点を選んで測るためのデータ。views/NNN.jpg（モデルが見た画像）と views/NNN.xyz（各画素の部屋座標）。"""
    from PIL import Image

    vdir = job_dir / "views"
    vdir.mkdir(exist_ok=True)
    meta = []
    for i, (mask, xyz) in enumerate(grids):
        room = xyz.astype(np.float64) @ T[:3, :3].T + T[:3, 3]
        (vdir / f"{i:03d}.xyz").write_bytes(encode_xyz_mm(room))
        Image.fromarray(images[i]).save(vdir / f"{i:03d}.jpg", "JPEG", quality=90)
        meta.append({"n": i, "w": int(mask.shape[1]), "h": int(mask.shape[0]), "valid": round(float(mask.mean()), 4)})
    (vdir / "views.json").write_text(json.dumps(meta), encoding="utf-8")
    return meta


def structure_from_raw(raw: dict, job_dir: Path) -> dict:
    """生の復元結果から room.json と cloud.bin を作る。写真が無くても再実行できる。"""
    pts, conf = raw["points"].astype(np.float64), raw["conf"]
    keep = conf >= np.percentile(conf, CONF_DROP_PERCENTILE)
    pts_k, cam_k, col_k = pts[keep], raw["cam_index"][keep], raw["colors"][keep]
    res = build_room_structure(pts_k, raw["cam2world"], cam_k, StructureParams())
    T = res.T_room_from_world
    room_pts = (pts_k @ T[:3, :3].T + T[:3, 3]).astype(np.float32)
    col_eq, color_diag = equalize_view_colors(room_pts, col_k, cam_k)
    cp, cc = voxel_downsample(room_pts, col_eq, CLOUD_MAX_POINTS)
    with open(job_dir / "cloud.bin", "wb") as f:
        f.write(cp.astype("<f4").tobytes())
        f.write(cc.astype(np.uint8).tobytes())
    cams = raw["cam2world"][:, :3, 3] @ T[:3, :3].T + T[:3, 3]
    room = {
        "format": 1,
        "unit": "m",
        "coord": "right-handed, Y-up, floor y=0, outline in (x,z) counter-clockwise",
        "outline": res.outline,
        "edge_status": res.edge_status,
        "edge_support": res.edge_support,
        "ceiling_height": res.ceiling_height,
        "ceiling_observed": res.ceiling_observed,
        "obstacles": res.obstacles,
        "cameras": cams.round(4).tolist(),
        "T_room_from_world": T.round(6).tolist(),
        "cloud_points": int(len(cp)),
        "views": export_views(view_grids(raw), raw["view_images"], T, job_dir) if "valid_bits" in raw and "view_images" in raw else [],
        "scale_status": "model_estimate",
        "capture_quality": capture_quality(raw),
        "diagnostics": {**res.diagnostics, **color_diag},
        "meta": raw["meta"],
    }
    (job_dir / "room.json").write_text(json.dumps(room, ensure_ascii=False), encoding="utf-8")
    return room


def run_chunked(imgs: list[Path], chunks: list[list[int]], job_dir: Path) -> dict:
    """写真が多いとき: 範囲に分けて順に復元し、共通の写真でつなぐ（roomfit.stitch）。
    進み具合は「何範囲中の何番目か」だけを status に書く（実際に数えられるものだけ）。途中の結果は chunks/ に残し、再実行ではやり直さない。"""
    from .reconstruct import load_model, load_raw, run_mapanything, save_raw
    from .stitch import merge_raws

    cdir = job_dir / "chunks"
    cdir.mkdir(exist_ok=True)
    model, raws = None, []
    for k, idx in enumerate(chunks):
        p = cdir / f"{k:02d}.npz"
        if p.exists():
            raws.append(load_raw(p))
            continue
        if model is None:
            model = load_model()
        write_status(job_dir, stage="inferring", chunk=k + 1, chunks=len(chunks))
        raw = run_mapanything([imgs[i] for i in idx], model=model)
        save_raw(raw, p)
        raws.append(raw)
    del model
    write_status(job_dir, stage="stitching", chunk=None, chunks=len(chunks))
    return merge_raws(raws)


def save_surfaces(surf: dict, out_dir: Path) -> dict:
    """make_surfaces の結果を surfaces/ に JPEG で保存し、room.json に書く情報を返す。"""
    from PIL import Image

    sdir = out_dir / "surfaces"
    sdir.mkdir(exist_ok=True)
    info: dict = {"walls": [], "floor": None}
    for w in surf["walls"]:
        name = f"wall_{w['edge']}.jpg"
        Image.fromarray(w["image"]).save(sdir / name, "JPEG", quality=88)
        info["walls"].append({"edge": w["edge"], "file": name, "coverage": w["coverage"]})
    if surf["floor"] is not None:
        Image.fromarray(surf["floor"]["image"]).save(sdir / "floor.jpg", "JPEG", quality=88)
        info["floor"] = {"file": "floor.jpg", "coverage": surf["floor"]["coverage"], "bbox": surf["floor"]["bbox"]}
    return info


def job_views(raw: dict, room: dict, job_dir: Path) -> list | None:
    """元の写真と、写真ごとに各画素が見ている 3D の点（部屋座標）をそろえる（壁の写真貼りとマーカーの縮尺合わせで使う）。
    元の写真が無い・数が合わないなどで作れなければ None。"""
    from PIL import Image, ImageOps

    from .textures import ViewData, photo_mapping

    if "valid_bits" not in raw:
        return None
    ups = sorted((job_dir / "upload").iterdir()) if (job_dir / "upload").exists() else []
    pres = sorted((job_dir / "images").glob("*.jpg"))
    if not ups or len(ups) != len(raw["cam2world"]) or len(pres) != len(ups):
        return None
    T = np.array(room["T_room_from_world"])
    views = []
    for v, (mask, xyz) in enumerate(view_grids(raw)):
        grid = xyz.astype(np.float64) @ T[:3, :3].T + T[:3, 3]
        with Image.open(ups[v]) as im:
            photo = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
        with Image.open(pres[v]) as im:
            pw = im.size
        views.append(ViewData(cam2room=T @ raw["cam2world"][v], K=raw["intrinsics"][v], grid_room=grid, photo=photo,
                              to_photo=photo_mapping(mask.shape, pw, (photo.shape[1], photo.shape[0]))))
    return views


def add_surfaces(raw: dict, room: dict, job_dir: Path, views: list | None = None) -> None:
    """壁と床に元の写真を貼った画像を作り、room.json に "surfaces" として書き足す（roomfit.textures）。"""
    from .textures import make_surfaces

    views = views if views is not None else job_views(raw, room, job_dir)
    if views is None:
        return
    info = save_surfaces(make_surfaces(room["outline"], room["edge_status"], room["ceiling_height"], views), job_dir)
    room["surfaces"] = info
    (job_dir / "room.json").write_text(json.dumps(room, ensure_ascii=False), encoding="utf-8")


def add_marker_scale(room: dict, job_dir: Path, views: list | None) -> None:
    """写真に印刷マーカーが写っていれば、部屋の大きさを合わせる倍率を room.json に "marker_scale" として書き足す
    （roomfit.marker_scale）。部屋の形そのものは変えない（倍率を当てるかは画面側で決め、記録に残す）。"""
    from .marker_scale import estimate_scale

    if views is None:
        return
    ms = estimate_scale(views)
    if ms is None:
        return
    room["marker_scale"] = ms
    (job_dir / "room.json").write_text(json.dumps(room, ensure_ascii=False), encoding="utf-8")


def main(job_dir: Path) -> int:
    from .reconstruct import load_raw, preprocess_images, run_mapanything, save_raw
    from .stitch import plan_chunks

    t0 = time.time()
    try:
        raw_path = job_dir / "raw.npz"
        if raw_path.exists():
            raw = load_raw(raw_path)  # 同じ入力での再実行では重い推論をやり直さない
        else:
            write_status(job_dir, stage="preprocessing")
            src = sorted((job_dir / "upload").iterdir())
            imgs = preprocess_images(src, job_dir / "images")
            write_status(job_dir, stage="loading_model")  # ライブラリの読み込みだけでも時間がかかる
            chunks = plan_chunks(len(imgs))
            if len(chunks) == 1:
                raw = run_mapanything(imgs, on_stage=lambda s: write_status(job_dir, stage=s))
            else:
                raw = run_chunked(imgs, chunks, job_dir)
            save_raw(raw, raw_path)
        write_status(job_dir, stage="structuring")
        room = structure_from_raw(raw, job_dir)
        views = None
        try:
            views = job_views(raw, room, job_dir)
        except Exception:
            (job_dir / "surfaces_error.log").write_text(traceback.format_exc(), encoding="utf-8")
        try:   # 印刷マーカーでの縮尺合わせ（任意）。失敗しても復元そのものは成功として扱う
            add_marker_scale(room, job_dir, views)
        except Exception:
            (job_dir / "marker_error.log").write_text(traceback.format_exc(), encoding="utf-8")
        try:   # 見た目のための画像。失敗しても復元そのものは成功として扱う（点の表示は使える）
            add_surfaces(raw, room, job_dir, views)
        except Exception:
            (job_dir / "surfaces_error.log").write_text(traceback.format_exc(), encoding="utf-8")
        write_status(job_dir, stage="done", seconds=round(time.time() - t0, 1))
        return 0
    except StructureError as e:
        write_status(job_dir, stage="failed", error_code=e.code, error_message=e.message)
        return 2
    except MemoryError:
        write_status(job_dir, stage="failed", error_code="out_of_memory",
                     error_message="メモリが足りず処理できませんでした。写真の枚数を減らして再度お試しください。")
        return 3
    except Exception as e:  # 詳細は開発者向けログへ。利用者には次の行動が分かる文だけ返す。
        (job_dir / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        write_status(job_dir, stage="failed", error_code="internal",
                     error_message="復元の途中でエラーが起きました。もう一度試しても直らない場合は、写真を変えてお試しください。",
                     error_detail=f"{type(e).__name__}: {e}"[:300])
        return 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
