"""実験: 撮影アプリ（ARCore）の記録を、写真に加えて「カメラ情報」と「カメラの位置・向き」もモデルへ渡して復元する。

狙い: MapAnything の論文では、写真だけだと大きさの誤差が大きく、カメラ情報と実寸の位置を入力に足すと小さくなる（docs/検証記録.md に経緯）。
     スマホの AR 機能が出す位置は実寸なので、それを入力にすれば、何も測らずに大きさが合う可能性がある。
     これまでは「復元した後で ARCore の位置と突き合わせる」ことしか試しておらず、入力として渡すのは初めて。

座標の変換（間違えやすい）:
  - 記録の姿勢は OpenGL 規約（+X 右, +Y 上, -Z 前）のカメラ → 世界。OpenCV 規約へは R_cv = R_gl @ diag(1,-1,-1)。
  - 保存された写真はセンサーの向き（横長）。縦持ちで見たままの向きにするため、時計回りに 90 度回す。
    回した後の画素 (u', v') = (H - v, u) なので、カメラ座標は (x', y', z) = (-y, x, z)。
    内部パラメータは fx' = fy, fy' = fx, cx' = H - cy, cy' = cx。

使い方（server フォルダで）:
  uv run python ../experiments/run_arcore_posed.py <記録フォルダ> <出力フォルダ> <番号 例: 26,28,30> [pose|auto|intr|none]
    pose = カメラ情報と位置を渡す（既定）, intr = カメラ情報だけ, none = 写真だけ（比較用）
    auto = カメラ情報は全部に、位置は「写真の見え方と合う区間」（roomfit.ar_check、記録全体で判定）のコマだけに渡す
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
from roomfit.ar_check import check_capture  # noqa: E402
from roomfit.arcore_import import GL_FROM_CV, quat_to_rot  # noqa: E402
from roomfit.reconstruct import load_model, save_raw  # noqa: E402
from roomfit.structure import StructureError  # noqa: E402
from roomfit.worker import capture_quality, structure_from_raw  # noqa: E402

ROT_CW = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])   # (x, y, z) → (-y, x, z)


def main() -> None:
    import torch
    from mapanything.utils.image import preprocess_inputs

    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    idx = [int(x) for x in sys.argv[3].split(",")]
    mode = sys.argv[4] if len(sys.argv) > 4 else "pose"
    (out / "images").mkdir(parents=True, exist_ok=True)
    trust, chk = None, None
    if mode == "auto":
        chk = check_capture(src)
        trust = {int(fid): u for fid, u in zip(chk["frame_ids"], chk["use_pose"])}
        print("AR の検算:", chk["status"], "使う区間:", [(g["first"], g["last"]) for g in chk["segments"] if g.get("chosen")],
              "位置を渡すコマ:", [o for o in idx if trust.get(o)], flush=True)
    views, given = [], []
    for new, old in enumerate(idx):
        f = json.loads((src / f"{old:03d}.json").read_text(encoding="utf-8"))
        with Image.open(src / f"{old:03d}.jpg") as im:
            im = im.convert("RGB").rotate(-90, expand=True)
            full_w, full_h = im.size                                   # 回した後（縦長）
            im.thumbnail((1024, 1024), Image.LANCZOS)
            im.save(out / "images" / f"{new:03d}.jpg", "JPEG", quality=92)
            arr = np.asarray(im)
        s = arr.shape[1] / full_w
        K = np.array([[f["fy"] * s, 0, (f["h"] - f["cy"]) * s], [0, f["fx"] * s, f["cx"] * s], [0, 0, 1]], np.float32)
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = quat_to_rot(f["q"]) @ GL_FROM_CV @ ROT_CW.T
        c2w[:3, 3] = f["t"]
        view = {"img": torch.from_numpy(arr.copy())}
        posed = mode == "pose" or (mode == "auto" and trust.get(old, False))
        if mode in ("pose", "intr", "auto"):
            view["intrinsics"] = torch.from_numpy(K)
        if posed:
            view["camera_poses"] = torch.from_numpy(c2w)
            view["is_metric_scale"] = torch.tensor([True])
        views.append(view)
        given.append({"old": old, "K": K.tolist(), "cam2world": c2w.tolist(), "posed": posed})
    (out / "framemap.json").write_text(json.dumps({str(old): new for new, old in enumerate(idx)}), encoding="utf-8")
    (out / "given.json").write_text(json.dumps({"mode": mode, "views": given, "ar_check": None if chk is None else
                                                {k: v for k, v in chk.items() if k != "pairs"}}, ensure_ascii=False), encoding="utf-8")
    model = load_model()
    t0 = time.time()
    with torch.inference_mode():
        preds = model.infer(preprocess_inputs(views), memory_efficient_inference=True, use_amp=False, apply_mask=True, mask_edges=True,
                            apply_confidence_mask=False)
    sec = round(time.time() - t0, 1)

    pts, cols, conf, cam_idx, poses, intr, sizes, masks, imgs = [], [], [], [], [], [], [], [], []
    for vi, pr in enumerate(preds):
        p3 = pr["pts3d"][0].float().cpu().numpy()
        img = pr["img_no_norm"][0].float().cpu().numpy()
        cf = pr["conf"][0].float().cpu().numpy()
        ok = pr["mask"][0].cpu().numpy().reshape(cf.shape).astype(bool) & np.isfinite(p3).all(-1)
        pts.append(p3[ok].astype(np.float32)); cols.append((np.clip(img[ok], 0, 1) * 255).astype(np.uint8)); conf.append(cf[ok].astype(np.float32))
        cam_idx.append(np.full(int(ok.sum()), vi, np.int32)); poses.append(pr["camera_poses"][0].float().cpu().numpy())
        intr.append(pr["intrinsics"][0].float().cpu().numpy()); sizes.append(list(cf.shape)); masks.append(np.packbits(ok.ravel()))
        imgs.append((np.clip(img, 0, 1) * 255).astype(np.uint8))
    msf = preds[0].get("metric_scaling_factor")
    raw = {"points": np.concatenate(pts), "colors": np.concatenate(cols), "conf": np.concatenate(conf), "cam_index": np.concatenate(cam_idx),
           "cam2world": np.stack(poses).astype(np.float64), "intrinsics": np.stack(intr).astype(np.float64), "image_hw": np.array(sizes),
           "valid_bits": np.concatenate(masks), "view_images": np.stack(imgs),
           "meta": {"model": "facebook/map-anything-apache", "n_views": len(idx), "sec_inference": sec, "input_mode": mode,
                    "metric_scaling_factor": None if msf is None else float(np.asarray(msf.float().cpu()).ravel()[0])}}
    save_raw(raw, out / "raw.npz")

    # 渡した位置と、出力の位置がどれだけ一致したか（モデルが入力に従ったかの確認）
    # （auto では、位置を渡したコマどうしだけで比べる）
    sel = np.array([g["posed"] or mode != "auto" for g in given])
    P_in = np.array([g["cam2world"] for g in given])[sel][:, :3, 3]
    P_out = raw["cam2world"][sel][:, :3, 3]
    base_in = np.linalg.norm(P_in[:, None] - P_in[None], axis=-1)
    base_out = np.linalg.norm(P_out[:, None] - P_out[None], axis=-1)
    far = base_in > 0.15
    ratio = float(np.median(base_out[far] / base_in[far])) if far.any() else float("nan")
    fx_in = float(np.mean([g["K"][0][0] for g in given])) * raw["image_hw"][0][1] / arr.shape[1]
    print(f"{mode}: {len(idx)} 枚 推論 {sec} 秒 / カメラどうしの距離（出力 ÷ ARCore）の中央値 {ratio:.3f} / 焦点距離 入力 {fx_in:.0f} 出力 {raw['intrinsics'][:, 0, 0].mean():.0f}", flush=True)
    print("質:", capture_quality(raw), flush=True)
    try:
        room = structure_from_raw(raw, out)
        print("天井高:", room["ceiling_height"], " 壁:", room["edge_status"], flush=True)
    except StructureError as e:
        print("間取りの抽出は失敗:", e.code, flush=True)


if __name__ == "__main__":
    main()
