"""撮影アプリ（ARCore）の記録を、MapAnything の入力（写真＋内部パラメータ＋検算に通った区間の実寸の位置）にする。

写真だけの復元は、大きさを部屋によって ±5% 外す（docs/検証記録.md §22）。AR の位置は端末の動きから測った実寸なので、
入力に渡すと大きさと写真どうしのつながりが良くなる（§15）。ただし端末によっては「追跡中」のまま位置が飛ぶ・向きを取り違えるので、
roomfit.ar_check で写真と合うと確かめた 1 区間のコマにだけ位置を渡し、ほかのコマは写真と内部パラメータだけにする（§19）。
位置を渡した区間でカメラが 0.5m 未満しか動いていなければ、ar_check が scale_cue = "weak" を返し、質の判定が「大きさは要確認」を付ける。

向き: 記録の写真はセンサーの向き（横長）。縦持ちで撮る前提で時計回りに 90 度回して渡す（experiments/run_arcore_posed.py と同じ）。
回した後の画素 (u', v') = (H - v, u) なので、カメラ座標は (x', y', z) = (-y, x, z)、内部パラメータは fx' = fy, fy' = fx, cx' = H - cy, cy' = cx。
"""
from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from .arcore_import import GL_FROM_CV, CaptureError, quat_to_rot
from .reconstruct import MAX_SIDE

ROT_CW = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
MAX_VIEWS = 16           # 1 回の復元に渡す枚数の上限（16GB の PC で 1 回に収まる枚数。§2）
MIN_VIEWS = 8            # 重なりが十分でも、これだけは渡す（1 つの角で 6〜8 枚が良かった。§12）
CAPTURE_FILE = re.compile(r"(\d{3}\.(json|jpg)|session\.json)")
MAX_CAPTURE_FILES = 2000


def extract_capture(zip_path: Path, dest: Path) -> Path:
    """記録の zip を展開する。中のフォルダ構成は問わず、NNN.json / NNN.jpg と session.json だけを平らに取り出す。"""
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if not n.endswith("/")]
        if len(names) > MAX_CAPTURE_FILES:
            raise CaptureError("記録のファイルが多すぎます。1 回の記録だけを送ってください。")
        for n in names:
            base = Path(n).name
            if CAPTURE_FILE.fullmatch(base):
                (dest / base).write_bytes(z.read(n))
    return dest


def _frames(folder: Path) -> list[dict]:
    frames = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(folder.glob("[0-9][0-9][0-9].json"))]
    return [f for f in frames if (folder / f"{f['id']}.jpg").exists()]


def _rotated(f: dict) -> tuple[np.ndarray, np.ndarray]:
    """縦向きに回した写真での内部パラメータ（元の写真の画素）と、OpenCV 規約の cam2world。"""
    K = np.array([[f["fy"], 0, f["h"] - f["cy"]], [0, f["fx"], f["cx"]], [0, 0, 1.0]])
    c2w = np.eye(4)
    c2w[:3, :3] = quat_to_rot(f["q"]) @ GL_FROM_CV @ ROT_CW.T
    c2w[:3, 3] = f["t"]
    return K, c2w


def plan_views(folder: Path, check=None) -> dict:
    """どのコマを使い、どのコマに位置を渡すかを決める（重い推論の前の、写真と記録だけでできる部分）。
    戻り値: {"ids": 使うコマ, "posed": 位置を渡すか, "K": 縦向きの内部パラメータ, "c2w": 縦向きの姿勢, "pose_check": 検算の要約}"""
    from .ar_check import check_capture
    from .frame_select import select_frames

    frames = _frames(folder)
    if len(frames) < 3:
        raise CaptureError("写真のある場面が少なすぎます。3 枚以上になるように、ゆっくり向きを変えながら記録してください。")
    chk = (check or check_capture)(folder)
    trust = {fid: bool(u) for fid, u in zip(chk["frame_ids"], chk["use_pose"])} if chk.get("status") not in ("unchecked", "rejected") else {}
    Ks, c2ws = zip(*(_rotated(f) for f in frames))
    # 重なりは主に向きで決まり、向きは AR が飛ぶ端末でも 1〜2° で合う（§16）ので、全コマの AR の姿勢で選ぶ
    sel = select_frames(np.stack(c2ws), np.stack(Ks), (frames[0]["h"], frames[0]["w"]), max_frames=MAX_VIEWS)["idx"]
    if len(sel) < min(len(frames), MIN_VIEWS):   # どのコマも重なっていると、重なりだけで選ぶと両端の 2 枚になる
        sel = sorted({int(round(x)) for x in np.linspace(0, len(frames) - 1, min(len(frames), MIN_VIEWS))})
    ids = [frames[i]["id"] for i in sel]
    return {"ids": ids, "posed": [trust.get(fid, False) for fid in ids], "K": [Ks[i] for i in sel], "c2w": [c2ws[i] for i in sel],
            "pose_check": {k: v for k, v in chk.items() if k not in ("pairs", "use_pose", "frame_ids", "segments")}}


def build_views(folder: Path, out_images: Path, plan: dict) -> tuple[list[Path], list]:
    """plan_views の結果から、縦向きにした写真（out_images）と MapAnything の入力を作る。"""
    import torch
    from mapanything.utils.image import preprocess_inputs

    out_images.mkdir(parents=True, exist_ok=True)
    paths, views = [], []
    for i, (fid, posed, K, c2w) in enumerate(zip(plan["ids"], plan["posed"], plan["K"], plan["c2w"])):
        with Image.open(folder / f"{fid}.jpg") as im:
            im = im.convert("RGB").rotate(-90, expand=True)
            full_w = im.size[0]
            im.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
            dp = out_images / f"{i:03d}.jpg"
            im.save(dp, "JPEG", quality=92)
            arr = np.asarray(im)
        s = arr.shape[1] / full_w
        view = {"img": torch.from_numpy(arr.copy()),
                "intrinsics": torch.from_numpy((K * np.array([[s], [s], [1.0]])).astype(np.float32))}
        if posed:
            view["camera_poses"] = torch.from_numpy(c2w.astype(np.float32))
            view["is_metric_scale"] = torch.tensor([True])
        paths.append(dp)
        views.append(view)
    return paths, preprocess_inputs(views)
