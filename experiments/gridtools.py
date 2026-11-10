"""復元結果(raw.npz)を、モデルが使った画素格子（view ごとの H×W）へ戻す道具。

raw.npz は有効な画素の点だけを行優先の順に並べて保存している。どの画素が欠けたかは保存していなかったので、
(1) 行の折り返し（投影した u が大きく戻る所）で行を分け、(2) 行の中は投影した u と順序で列を決める。
(3) 決めた画素の色が、モデル入力画像の同じ画素の色と一致することで検算する。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

DINO_MEAN = np.array([0.485, 0.456, 0.406])
DINO_STD = np.array([0.229, 0.224, 0.225])


def model_input_images(image_paths: list[Path]) -> list[np.ndarray]:
    """モデルに入る大きさ・切り抜きの画像を、uint8 の H×W×3 で返す。"""
    from mapanything.utils.image import load_images

    out = []
    for v in load_images([str(p) for p in image_paths]):
        t = v["img"][0].permute(1, 2, 0).cpu().numpy()
        out.append((np.clip(t * DINO_STD + DINO_MEAN, 0, 1) * 255).round().astype(np.uint8))
    return out


def _align_row(colors: np.ndarray, image_row: np.ndarray, guess: np.ndarray) -> np.ndarray:
    """画素が欠けた行で、点 k を列 k+s (s は単調に増える) へ割り当てる。色の一致を主、投影位置を従の手がかりにする。"""
    m, W = len(colors), len(image_row)
    S = W - m + 1
    k = np.arange(m)[:, None]
    s = np.arange(S)[None, :]
    cost = np.abs(colors[:, None, :].astype(np.int32) - image_row[(k + s).clip(0, W - 1)].astype(np.int32)).sum(-1).astype(np.float64)
    cost += 0.05 * np.abs((k + s) - guess[:, None])
    dp = np.empty_like(cost)
    arg = np.empty((m, S), np.int32)
    dp[0] = cost[0]
    for i in range(1, m):
        pm = np.minimum.accumulate(dp[i - 1])
        idx = np.zeros(S, np.int32)  # 前の行で、s 以下の範囲の最小の位置
        best = 0
        for j in range(1, S):
            if dp[i - 1, j] < dp[i - 1, best]:
                best = j
            idx[j] = best
        arg[i] = idx
        dp[i] = cost[i] + pm
    out = np.empty(m, np.int64)
    j = int(np.argmin(dp[-1]))
    for i in range(m - 1, -1, -1):
        out[i] = i + j
        if i:
            j = int(arg[i, j])
    return out


def recover_grid(raw: dict, view: int, image: np.ndarray | None = None) -> dict:
    """image（モデル入力画像）を渡すと、画素が欠けた行を色で突き合わせて正確に戻す。"""
    H, W = (int(x) for x in raw["image_hw"][view])
    sel = raw["cam_index"] == view
    P = raw["points"][sel].astype(np.float64)
    C = raw["colors"][sel]
    conf = raw["conf"][sel]
    T, K = raw["cam2world"][view], raw["intrinsics"][view]
    pc = (P - T[:3, 3]) @ T[:3, :3]
    u = K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2]
    vv = K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2]
    breaks = np.flatnonzero(np.diff(u) < -W / 2) + 1
    starts = np.concatenate([[0], breaks, [len(P)]])
    n_rows = len(starts) - 1
    rows = np.empty(len(P), np.int64)
    cols = np.empty(len(P), np.int64)
    if n_rows == H:
        row_of_seg = np.arange(H)
    else:  # 丸ごと欠けた行がある場合は、投影した v で行を決める
        med = np.array([np.median(vv[starts[i]:starts[i + 1]]) for i in range(n_rows)])
        row_of_seg = np.clip(np.round(med - 0.5).astype(int), 0, H - 1)
        if len(np.unique(row_of_seg)) != n_rows:
            raise RuntimeError(f"view {view}: 行を一意に決められません")
    for i in range(n_rows):
        a, b = starts[i], starts[i + 1]
        m = b - a
        rows[a:b] = row_of_seg[i]
        if m == W:
            cols[a:b] = np.arange(W)
            continue
        guess = np.round(u[a:b] - 0.5).astype(int)
        if image is not None:
            cols[a:b] = _align_row(C[a:b], image[row_of_seg[i]], guess)
            continue
        c = np.empty(m, np.int64)
        prev = -1
        for k in range(m):
            c[k] = min(max(prev + 1, guess[k]), W - (m - k))
            prev = c[k]
        cols[a:b] = c
    xyz = np.full((H, W, 3), np.nan)
    rgb = np.zeros((H, W, 3), np.uint8)
    cf = np.full((H, W), np.nan)
    xyz[rows, cols] = P
    rgb[rows, cols] = C
    cf[rows, cols] = conf
    return {"xyz": xyz, "rgb": rgb, "conf": cf, "valid": np.isfinite(xyz[..., 0]), "n_rows_found": n_rows, "H": H, "W": W}


def check_against_image(grid: dict, image: np.ndarray) -> dict:
    """格子に戻した色と、モデル入力画像の色の差。ずれていれば差が大きくなる。"""
    v = grid["valid"]
    d = np.abs(grid["rgb"].astype(int) - image.astype(int))[v]
    shifted = np.abs(grid["rgb"][:, 1:].astype(int) - image[:, :-1].astype(int))[v[:, 1:]]
    return {"mean_abs_diff": float(d.mean()), "p99_abs_diff": float(np.percentile(d, 99)), "mean_abs_diff_if_shifted_1px": float(shifted.mean())}
