"""形状抽出の処理テスト。

ここで使うのは計算で作った合成の点群。処理の正しさ（座標変換・向き・凹凸の保持・未確認の扱い）を
確かめるためのもので、実際の部屋での計測精度の証明には使わない。
"""
import numpy as np
import pytest

from roomfit.structure import StructureError, build_room_structure, rotation_up_to_y, simplify_rectilinear

RNG = np.random.default_rng(1)


def _plane(p0, u, v, nu, nv, step=0.02):
    a = np.arange(0, nu, step)
    b = np.arange(0, nv, step)
    A, B = np.meshgrid(a, b)
    return p0 + A.reshape(-1, 1) * u + B.reshape(-1, 1) * v


def make_room(skip_wall: str | None = None, noise=0.004):
    """L 字ではなく、4.0m × 3.0m・高さ 2.4m の部屋。x=4 側の壁の途中に 30cm×40cm の柱が出ている。
    room 座標 (x, y上, z)。柱は (3.7..4.0, z 1.0..1.4)。左右非対称なので鏡映の検出に使える。"""
    X, Y, Z = np.eye(3)
    parts = {
        "floor": _plane(np.zeros(3), X, Z, 4.0, 3.0),
        "ceil": _plane(np.array([0, 2.4, 0]), X, Z, 4.0, 3.0),
        "z0": _plane(np.zeros(3), X, Y, 4.0, 2.4),
        "z3": _plane(np.array([0, 0, 3.0]), X, Y, 4.0, 2.4),
        "x0": _plane(np.zeros(3), Z, Y, 3.0, 2.4),
        "x4a": _plane(np.array([4.0, 0, 0]), Z, Y, 1.0, 2.4),
        "x4b": _plane(np.array([4.0, 0, 1.4]), Z, Y, 1.6, 2.4),
        "pillar_front": _plane(np.array([3.7, 0, 1.0]), Z, Y, 0.4, 2.4),
        "pillar_s1": _plane(np.array([3.7, 0, 1.0]), X, Y, 0.3, 2.4),
        "pillar_s2": _plane(np.array([3.7, 0, 1.4]), X, Y, 0.3, 2.4),
    }
    if skip_wall:
        parts.pop(skip_wall)
    pts = np.concatenate(list(parts.values()))
    pts = pts + RNG.normal(0, noise, pts.shape)
    cams_room = np.array([[1.5, 1.4, 1.5], [2.5, 1.4, 1.2], [2.0, 1.4, 2.0]])
    return pts, cams_room


def to_world(pts_room, cams_room, yaw_deg=27.0, tilt_deg=6.0):
    """room → 任意に回した world。カメラは OpenCV 規約（画像の上 = カメラの -Y）。"""
    yaw, tilt = np.radians(yaw_deg), np.radians(tilt_deg)
    Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(tilt), -np.sin(tilt)], [0, np.sin(tilt), np.cos(tilt)]])
    # room の +Y(上) を world の -Y 付近へ（OpenCV の世界は下が +Y になりがち）
    flip = np.diag([1.0, -1.0, -1.0])
    R = Rx @ Ry @ flip
    assert np.linalg.det(R) > 0.999
    t = np.array([0.7, -0.2, 1.3])
    pw = pts_room @ R.T + t
    cw = cams_room @ R.T + t
    poses = []
    up_w = R @ np.array([0, 1.0, 0])
    for c in cw:
        P = np.eye(4)
        y = -up_w                      # カメラの +Y は下
        z = R @ np.array([1.0, 0, 0])  # 適当な水平方向を向く
        x = np.cross(y, z)
        P[:3, 0], P[:3, 1], P[:3, 2], P[:3, 3] = x, y, z, c
        poses.append(P)
    return pw, np.stack(poses)


def _cam_index(pw, poses):
    d = np.linalg.norm(pw[:, None, :] - poses[None, :, :3, 3], axis=2)
    return d.argmin(axis=1).astype(np.int32)


def _canon(outline):
    """輪郭を、回転（90°単位）と平行移動を除いて比べられる形にする。鏡映は除かない。"""
    v = np.array(outline)
    outs = []
    for k in range(4):
        c, s = np.cos(k * np.pi / 2), np.sin(k * np.pi / 2)
        r = v @ np.array([[c, -s], [s, c]]).T
        r = r - r.min(axis=0)
        outs.append(r)
    return outs


def _area(v):
    v = np.array(v)
    return 0.5 * float(np.sum(v[:, 0] * np.roll(v[:, 1], -1) - np.roll(v[:, 0], -1) * v[:, 1]))


def _has_vertex(cands, target, tol=0.05):
    return any(np.min(np.linalg.norm(c - np.array(target), axis=1)) < tol for c in cands)


def test_rotation_is_proper_and_maps_up():
    up = np.array([0.1, -0.97, 0.2])
    up /= np.linalg.norm(up)
    R = rotation_up_to_y(up)
    assert np.allclose(R @ up, [0, 1, 0], atol=1e-9)
    assert np.isclose(np.linalg.det(R), 1.0)


def test_room_with_pillar_is_recovered_and_not_mirrored():
    pts, cams = make_room()
    pw, poses = to_world(pts, cams)
    res = build_room_structure(pw, poses, _cam_index(pw, poses))
    assert np.isclose(np.linalg.det(res.T_room_from_world[:3, :3]), 1.0, atol=1e-6)
    assert _area(res.outline) > 0, "輪郭は (x,z) で反時計回り"
    # 面積: 4×3 から柱 0.3×0.4 を引いた 11.88 ㎡
    assert abs(_area(res.outline) - 11.88) < 0.25
    assert res.ceiling_observed and abs(res.ceiling_height - 2.4) < 0.03
    # 柱の凹み（深さ 30cm・幅 40cm）が輪郭に残っている
    v = np.array(res.outline)
    steps = np.abs(np.roll(v, -1, axis=0) - v).sum(axis=1)
    assert np.sum(np.abs(steps - 0.3) < 0.04) >= 2 and np.sum(np.abs(steps - 0.4) < 0.04) >= 1
    assert len(res.outline) == 8
    # 鏡映していない: 正しい形（回転のみ許す）には一致し、鏡に映した形には一致しない
    expected = np.array([[0, 0], [4, 0], [4, 1.0], [3.7, 1.0], [3.7, 1.4], [4, 1.4], [4, 3], [0, 3]], float)
    got = _canon(res.outline)

    def matches(target):
        t = target - target.min(axis=0)
        return any(all(_has_vertex([g], p, 0.06) for p in t) for g in got)

    mirrored = expected * np.array([-1, 1])
    assert matches(expected)
    assert not matches(mirrored)
    assert all(s == "observed" for s in res.edge_status)


def test_unphotographed_wall_is_unobserved_not_missing():
    pts, cams = make_room(skip_wall="z3")
    pw, poses = to_world(pts, cams)
    res = build_room_structure(pw, poses, _cam_index(pw, poses))
    assert "unobserved" in res.edge_status
    assert res.edge_status.count("observed") >= 5


def test_corner_shot_from_further_back_than_the_nearest_floor_seen():
    """1 つの角を 2m ほど離れて撮ると、足もとの床は写らず、カメラは復元できた点の範囲の外に立つ（2026-10-21 の自宅の記録）。"""
    X, Y, Z = np.eye(3)
    pts = np.concatenate([_plane(np.zeros(3), X, Z, 1.2, 1.2), _plane(np.zeros(3), X, Y, 1.5, 2.4), _plane(np.zeros(3), Z, Y, 1.5, 2.4)])
    pts = pts + RNG.normal(0, 0.004, pts.shape)
    pw, poses = to_world(pts, np.array([[2.0, 1.4, 2.0], [2.3, 1.4, 1.6], [1.6, 1.4, 2.3]]))
    res = build_room_structure(pw, poses, _cam_index(pw, poses))
    assert res.edge_status.count("observed") >= 2


def test_small_step_below_tolerance_is_merged_but_pillar_kept():
    v = np.array([[0, 0], [2, 0], [2, 0.02], [4, 0.02], [4, 1], [3.7, 1], [3.7, 1.4], [4, 1.4], [4, 3], [0, 3]], float)
    out = simplify_rectilinear(v, tol=0.04, min_edge=0.05)
    assert len(out) == 8  # 2cm の段差は消え、30cm の柱は残る


def test_failure_has_user_message():
    with pytest.raises(StructureError) as e:
        build_room_structure(np.zeros((100, 3)), np.stack([np.eye(4)]), np.zeros(100, dtype=np.int32))
    assert e.value.code == "too_few_points" and "写真" in e.value.message


def test_ceiling_height_is_measured_from_floor_even_far_from_origin():
    """世界座標の原点が部屋から遠く、床と天井の法線がわずかに違っても、天井高は床からの高さで出ること。
    （実データで、2 つの平面の係数の差を高さとして使い 6cm 低く出た不具合の再発防止）"""
    pts, cams = make_room()
    # 天井だけ 0.6° 傾ける（復元のゆがみを模す）。部屋の中央での高さは 2.4m のまま
    top = np.abs(pts[:, 1] - 2.4) < 0.02
    pts[top, 1] += np.tan(np.radians(0.6)) * (pts[top, 0] - 2.0)
    pw, poses = to_world(pts, cams)
    shift = np.array([25.0, -3.0, 40.0])
    poses[:, :3, 3] += shift
    res = build_room_structure(pw + shift, poses, _cam_index(pw + shift, poses))
    assert res.ceiling_observed and abs(res.ceiling_height - 2.4) < 0.015


def test_non_right_corner_is_rectified_keeping_wall_lengths():
    """復元で壁の角が 90° からずれた場合（ここでは 86°）、直角に直され、壁に沿った長さは保たれること。"""
    pts, cams = make_room()
    sh = np.tan(np.radians(4.0))
    S = np.array([[1, 0, sh], [0, 1, 0], [0, 0, 1.0]])   # z に比例して x がずれる = x=一定の壁が 4° 傾く
    S[:, 2] /= np.linalg.norm(S[[0, 2], 2])              # 傾いた壁に沿った長さは変えない
    pw, poses = to_world(pts @ S.T, cams @ S.T)
    res = build_room_structure(pw, poses, _cam_index(pw, poses))
    d = res.diagnostics
    assert d["rectified"] and abs(d["wall_corner_angle_deg"] - 86.0) < 0.6
    assert abs(_area(res.outline) - 11.88) < 0.3
    v = np.array(res.outline)
    ext = np.sort(v.max(axis=0) - v.min(axis=0))
    assert abs(ext[0] - 3.0) < 0.04 and abs(ext[1] - 4.0) < 0.04
    assert np.linalg.det(res.T_room_from_world[:3, :3]) > 0.9


def test_right_corner_is_left_untouched():
    pts, cams = make_room()
    pw, poses = to_world(pts, cams)
    res = build_room_structure(pw, poses, _cam_index(pw, poses))
    assert not res.diagnostics["rectified"] and abs(res.diagnostics["wall_corner_angle_deg"] - 90) < 0.3


def test_view_grids_and_xyz_encoding_roundtrip(tmp_path):
    """有効画素のビット列から画素格子へ戻せること、mm の int16 で書いた座標が 1mm 以内で戻ること。"""
    from roomfit.worker import XYZ_INVALID, encode_xyz_mm, export_views, view_grids

    rng = np.random.default_rng(3)
    masks = [rng.random((5, 7)) > 0.2, rng.random((5, 7)) > 0.5]
    grids_true = [rng.normal(0, 2, (5, 7, 3)).astype(np.float32) for _ in masks]
    raw = {
        "image_hw": np.array([[5, 7], [5, 7]]),
        "valid_bits": np.concatenate([np.packbits(m.ravel()) for m in masks]),
        "points": np.concatenate([g[m] for g, m in zip(grids_true, masks)]),
    }
    got = view_grids(raw)
    for (m, xyz), mt, gt in zip(got, masks, grids_true):
        assert (m == mt).all() and np.allclose(xyz[m], gt[m]) and np.isnan(xyz[~m]).all()
    T = np.eye(4)
    T[:3, 3] = [0.5, 0.0, -1.0]
    meta = export_views(got, np.zeros((2, 5, 7, 3), np.uint8), T, tmp_path)
    assert [v["n"] for v in meta] == [0, 1] and (tmp_path / "views" / "001.jpg").exists()
    q = np.frombuffer((tmp_path / "views" / "000.xyz").read_bytes(), dtype="<i2").reshape(5, 7, 3)
    assert (q[~masks[0]] == XYZ_INVALID).all()
    assert np.abs(q[masks[0]] / 1000.0 - (grids_true[0][masks[0]] + T[:3, 3])).max() <= 0.0006
    assert len(encode_xyz_mm(np.full((2, 2, 3), 100.0))) == 24  # 範囲外(100m)も落ちずに「値なし」になる


def test_capture_quality_flags_disagreeing_photos():
    """同じ壁を見た 3 枚の写真の 3D 位置が一致していれば good、1 枚だけ 4cm ずれていれば poor になること。"""
    from roomfit.worker import capture_quality

    rng = np.random.default_rng(5)
    def wall(offset):
        p = np.column_stack([rng.uniform(0, 1.5, 40000), rng.uniform(0, 1.2, 40000), np.full(40000, 2.0 + offset)])
        return p + rng.normal(0, 0.003, p.shape)
    K = np.array([[280.0, 0, 196], [0, 280, 259], [0, 0, 1]])
    def raw_of(offsets, f=(280, 280, 280)):
        pts = [wall(o) for o in offsets]
        Ks = np.stack([K * np.array([[x / 280], [x / 280], [1]]) for x in f])
        return {"points": np.concatenate(pts).astype(np.float32), "cam_index": np.repeat(np.arange(3), 40000).astype(np.int32),
                "cam2world": np.stack([np.eye(4)] * 3), "intrinsics": Ks}
    good = capture_quality(raw_of([0, 0, 0]))
    assert good["level"] == "good" and good["disagreement_mm"] < 11 and good["overlap"] > 0.9
    bad = capture_quality(raw_of([0, 0, 0.04]))
    assert bad["level"] == "poor" and "disagreement" in bad["reasons"]
    wide = capture_quality(raw_of([0, 0, 0], f=(280, 280, 400)))
    assert wide["level"] == "poor" and wide["reasons"] == ["focal"]
    # AR の位置を使った区間でカメラがあまり動いていない: 形がそろっていても、大きさは要確認
    r = raw_of([0, 0, 0]); r["meta"] = {"pose_check": {"status": "partial", "extent_m": 0.17, "scale_cue": "weak"}}
    ar = capture_quality(r)
    assert ar["level"] == "poor" and ar["reasons"] == ["ar_scale"] and ar["ar"]["extent_m"] == 0.17
    # 内部パラメータの無い記録（AR の深度を重ねた取り込み）でも落ちない
    r = raw_of([0, 0, 0]); del r["intrinsics"]
    assert capture_quality(r)["focal_ratio"] is None
