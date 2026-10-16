"""範囲に分けて復元したものをつなぐ処理（roomfit.stitch）のテスト。作り物の部屋で、答えが分かっている変換を戻せるかを見る。"""
import numpy as np
import pytest

from roomfit.stitch import merge_raws, plan_chunks, robust_similarity
from roomfit.worker import view_grids

H, W = 60, 80


def _rot(axis, deg):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    t = np.radians(deg)
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


def _view_points(i: int) -> np.ndarray:
    """写真 i に写る点（H×W×3）。床 (y=0) と奥の壁 (z=3) を、写真ごとに x 方向へ 0.4m ずつずらして見る。凹凸をつけて向きが決まるようにする。"""
    u, v = np.meshgrid(np.linspace(0, 1, W), np.linspace(0, 1, H))
    x = i * 0.4 + u * 1.6
    wall = v < 0.5
    y = np.where(wall, (0.5 - v) * 4.8, 0.0)
    z = np.where(wall, 3.0 + 0.15 * np.sin(x * 3.0), 3.0 - (v - 0.5) * 5.0)
    y = y + np.where(wall, 0.0, 0.05 * np.sin(x * 2.0 + z))
    return np.stack([x, y, z], -1)


def _raw(indices, s=1.0, R=np.eye(3), t=np.zeros(3), drop_seed=0):
    """写真 indices を、世界座標を s·R·x + t で動かした座標系で「復元」した raw を作る。"""
    rng = np.random.default_rng(drop_seed)
    pts, cols, conf, cam, bits, c2w = [], [], [], [], [], []
    for k, i in enumerate(indices):
        p = _view_points(i).reshape(-1, 3) @ R.T * s + t
        ok = rng.random(H * W) > 0.1                      # 有効画素は範囲ごとに少し違う
        pts.append(p[ok].astype(np.float32)); cols.append(np.full((int(ok.sum()), 3), 100 + i, np.uint8))
        conf.append(np.ones(int(ok.sum()), np.float32)); cam.append(np.full(int(ok.sum()), k, np.int32))
        bits.append(np.packbits(ok))
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = s * R @ np.array([i * 0.4 + 0.8, 1.2, 0.0]) + t
        c2w.append(T)
    n = len(indices)
    return {"points": np.concatenate(pts), "colors": np.concatenate(cols), "conf": np.concatenate(conf), "cam_index": np.concatenate(cam),
            "cam2world": np.stack(c2w), "intrinsics": np.stack([np.eye(3)] * n), "image_hw": np.array([[H, W]] * n),
            "valid_bits": np.concatenate(bits), "view_images": np.full((n, H, W, 3), 128, np.uint8), "meta": {"n_views": n, "sec_inference": 1.0}}


def test_plan_chunks_covers_all_and_shares():
    assert plan_chunks(12) == [list(range(12))]
    assert plan_chunks(16) == [list(range(16))]
    for n in (17, 20, 23, 30, 40, 57):
        ch = plan_chunks(n)
        assert ch[0][0] == 0 and ch[-1][-1] == n - 1
        assert sorted(set(sum(ch, []))) == list(range(n))
        for a, b in zip(ch, ch[1:]):
            assert a[-3:] == b[:3]                      # 隣どうしは 3 枚を共有
        assert all(8 <= len(c) <= 13 for c in ch), (n, [len(c) for c in ch])


def test_robust_similarity_ignores_outliers():
    rng = np.random.default_rng(1)
    src = rng.normal(size=(4000, 3))
    R = _rot([1, 2, 3], 25)
    dst = src @ R.T * 1.07 + [0.3, -0.2, 0.5]
    dst[:400] += rng.normal(scale=0.5, size=(400, 3))     # 1 割ははずれ
    s, r, t, res = robust_similarity(src, dst)
    assert s == pytest.approx(1.07, abs=2e-3)
    assert np.allclose(r, R, atol=2e-3)


def test_merge_restores_known_transform():
    chunks = plan_chunks(20)
    R1, R2 = _rot([0, 1, 0], 20), _rot([1, 1, 0], -12)
    raws = [_raw(chunks[0]),
            _raw(chunks[1], 0.93, R1, np.array([0.5, 0.1, -0.3]), 1)]
    if len(chunks) > 2:
        raws.append(_raw(chunks[2], 1.08, R2, np.array([-0.2, 0.3, 0.4]), 2))
    m = merge_raws(raws, match_scale=True)
    assert len(m["cam2world"]) == 20 and m["meta"]["n_views"] == 20
    assert m["meta"]["chunks"]["count"] == len(chunks)
    assert m["meta"]["chunks"]["links"][0]["scale_to_prev"] == pytest.approx(1 / 0.93, abs=2e-3)
    grids = view_grids(m)
    assert len(grids) == 20
    for i, (mask, xyz) in enumerate(grids):               # どの写真の点も、元の世界座標（最初の範囲の座標）へ戻っている
        want = _view_points(i)
        assert np.abs(xyz[mask] - want[mask]).max() < 5e-3, i
        assert np.allclose(m["cam2world"][i][:3, 3], [i * 0.4 + 0.8, 1.2, 0.0], atol=5e-3)
        assert np.allclose(m["cam2world"][i][:3, :3], np.eye(3), atol=2e-3)
    assert (np.bincount(m["cam_index"]) > 0).all() and m["cam_index"].max() == 19
    assert len(m["points"]) == len(m["colors"]) == len(m["conf"]) == len(m["cam_index"])
    assert m["colors"][m["cam_index"] == 15][0, 0] == 115     # 色も写真と対応したまま


def test_merge_default_keeps_each_chunk_size():
    """既定では大きさを変えない: 範囲ごとに回転・移動だけ違う場合は元へ戻り、大きさが違う範囲は、その大きさのまま置かれる。"""
    chunks = plan_chunks(30)
    assert len(chunks) == 3
    R1, R2 = _rot([0, 1, 0], 20), _rot([1, 1, 0], -12)
    m = merge_raws([_raw(chunks[0]), _raw(chunks[1], 1.0, R1, np.array([0.5, 0.1, -0.3]), 1), _raw(chunks[2], 1.0, R2, np.array([-0.2, 0.3, 0.4]), 2)])
    assert m["meta"]["chunks"]["match_scale"] is False
    for i, (mask, xyz) in enumerate(view_grids(m)):
        assert np.abs(xyz[mask] - _view_points(i)[mask]).max() < 1e-2, i
    big = merge_raws([_raw(chunks[0]), _raw(chunks[1], 1.05, R1, np.array([0.5, 0.1, -0.3]), 1)])
    link = big["meta"]["chunks"]["links"][0]
    assert link["scale_to_prev"] == 1.0 and link["size_ratio_seen"] == pytest.approx(1 / 1.05, abs=3e-3)
    g = view_grids(big)
    a, b = g[len(chunks[0])][1], _view_points(len(chunks[0]))          # 2 つ目の範囲の最初の新しい写真: 大きさは 5% 大きいまま
    assert np.nanmax(a[..., 0]) - np.nanmin(a[..., 0]) == pytest.approx(1.05 * (b[..., 0].max() - b[..., 0].min()), rel=0.02)


def test_merge_single_is_passthrough():
    r = _raw(list(range(5)))
    assert merge_raws([r]) is r
