"""壁・床に写真を貼る処理（roomfit.textures）のテスト。作り物のカメラと壁で、貼る向きと手前の物の扱いを確かめる。"""
import numpy as np

from roomfit.textures import ViewData, make_surfaces, photo_mapping

W, H = 320, 240
F = 200.0


def _camera_view(occluder: bool = False) -> ViewData:
    """部屋の (0, 1.2, 0) から +z 向き（奥の壁 z=2）を見るカメラ。写真の左半分は赤、右半分は青。"""
    K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1.0]])
    # カメラの向き（右手系の回転）: 画像の右 = 部屋の -x、画像の下 = 部屋の下、前 = 部屋の +z
    c2r = np.eye(4); c2r[:3, :3] = np.diag([-1.0, -1.0, 1.0]); c2r[:3, 3] = [0.0, 1.2, 0.0]
    photo = np.zeros((H, W, 3), np.uint8); photo[:, : W // 2] = (220, 30, 30); photo[:, W // 2:] = (30, 30, 220)
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    z = np.full((H, W), 2.0)
    if occluder:
        photo[100:140, 140:180] = (30, 200, 30)        # 壁の手前 1m にある緑の物
        z[100:140, 140:180] = 1.0
    pc = np.stack([(u - W / 2) / F * z, (v - H / 2) / F * z, z], -1)
    grid = pc @ c2r[:3, :3].T + c2r[:3, 3]
    return ViewData(cam2room=c2r, K=K, grid_room=grid, photo=photo, to_photo=photo_mapping((H, W), (W, H), (W, H)))


OUTLINE = [[-1.5, -1.0], [1.5, -1.0], [1.5, 2.0], [-1.5, 2.0]]   # 反時計回り。辺 2 が奥の壁（(1.5,2)→(-1.5,2)）


def test_wall_orientation_matches_edge_direction():
    s = make_surfaces(OUTLINE, ["unobserved", "unobserved", "observed", "unobserved"], 2.4, [_camera_view()])
    w = s["walls"][0]
    assert w["edge"] == 2 and w["coverage"] > 0.3
    im = w["image"].astype(float)
    h, wd = im.shape[:2]
    band = im[h // 2 - 20: h // 2 + 20]
    # 辺の始点 (1.5, 2) 側（壁の画像の左）は部屋の +x 側 = 写真の左 = 赤
    left, right = band[:, wd // 2 - 120: wd // 2 - 20].mean((0, 1)), band[:, wd // 2 + 20: wd // 2 + 120].mean((0, 1))
    assert left[0] > left[2] + 100 and right[2] > right[0] + 100


def test_object_in_front_of_wall_is_not_painted():
    s = make_surfaces(OUTLINE, ["unobserved", "unobserved", "observed", "unobserved"], 2.4, [_camera_view(occluder=True)])
    im = s["walls"][0]["image"].astype(float)
    green = (im[..., 1] > 150) & (im[..., 0] < 100) & (im[..., 2] < 100)
    assert green.sum() == 0      # 手前の緑の物は壁の模様として貼られない（その部分は写真なしの灰色）


def test_floor_is_made_and_reports_coverage():
    s = make_surfaces(OUTLINE, ["observed"] * 4, 2.4, [_camera_view()])
    assert s["floor"] is not None and 0 <= s["floor"]["coverage"] <= 1
    assert s["floor"]["bbox"] == [-1.5, -1.0, 1.5, 2.0]


# 保護領域（スイッチ・コンセントの上に写真の境目を通さない）
PLATE = (-1.15, -1.05, 1.1, 1.3)          # 奥の壁の上の小さな板（x の範囲、高さの範囲）
PLATE_BOX = (2.52, 1.33, 2.68, 1.07)      # 同じ板の保護領域（辺 2 の始点 (1.5, 2) からの距離 左, 高さ 上, 右, 下）、少し広め
BACK = ["unobserved", "unobserved", "observed", "unobserved"]


def _wall_view(cx: float) -> ViewData:
    """部屋の (cx, 1.2, 0) から奥の壁 z=2 を正面に見るカメラ。写真は、壁の模様（薄い縦縞と、暗い板 PLATE）をそのまま写したもの。"""
    K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1.0]])
    c2r = np.eye(4); c2r[:3, :3] = np.diag([-1.0, -1.0, 1.0]); c2r[:3, 3] = [cx, 1.2, 0.0]
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    z = np.full((H, W), 2.0)
    pc = np.stack([(u - W / 2) / F * z, (v - H / 2) / F * z, z], -1)
    grid = pc @ c2r[:3, :3].T + c2r[:3, 3]
    x, y = grid[..., 0], grid[..., 1]
    photo = np.full((H, W, 3), 200.0) + 20 * np.sin(x * 40)[..., None]
    photo[(x > PLATE[0]) & (x < PLATE[1]) & (y > PLATE[2]) & (y < PLATE[3])] = 60
    return ViewData(cam2room=c2r, K=K, grid_room=grid, photo=photo.astype(np.uint8), to_photo=photo_mapping((H, W), (W, H), (W, H)))


def _box_px(wall: dict, box=PLATE_BOX) -> tuple[slice, slice]:
    res = wall["res"]
    return slice(int(box[0] / res), int(box[2] / res)), slice(int((2.4 - box[1]) / res), int((2.4 - box[3]) / res))


def test_protected_region_is_painted_by_one_photo_without_seam():
    # 写真 0 は広く写すので先に貼られるが、板は写真の端（x = -1.1）で切れる。写真 1 は板の全体を写している
    views = [_wall_view(0.5), _wall_view(-0.8)]
    plain = make_surfaces(OUTLINE, BACK, 2.4, views, details=True)["walls"][0]
    xs, ys = _box_px(plain)
    assert len(np.unique(plain["src"][ys, xs])) == 2          # 保護しないと、板の上を境目が通る
    prot = make_surfaces(OUTLINE, BACK, 2.4, views, protect={2: [PLATE_BOX]}, details=True)["walls"][0]
    assert prot["protected"][0]["view"] == 1
    assert (prot["src"][ys, xs] == 1).all()                    # 板は写真 1 だけで塗る
    # 板の見た目の幅が実際の幅（10cm）のまま
    row = prot["image"][int((2.4 - 1.2) / prot["res"]), :, 0]
    assert abs((row < 120).sum() * prot["res"] - 0.10) < 0.01
    assert prot["coverage"] == plain["coverage"]


def test_protected_region_without_seam_is_left_as_is():
    views = [_wall_view(0.5), _wall_view(-0.8)]
    box = (1.4, 1.33, 1.56, 1.07)                               # 写真 0 だけで塗られる所
    plain = make_surfaces(OUTLINE, BACK, 2.4, views)["walls"][0]
    prot = make_surfaces(OUTLINE, BACK, 2.4, views, protect={2: [box]})["walls"][0]
    assert prot["protected"][0].get("kept") and np.array_equal(prot["image"], plain["image"])


def test_protected_region_not_fully_seen_by_any_photo_is_not_forced():
    # どちらの写真も板を写真の端で切っている: 全体を写した 1 枚がないので、普段どおりに貼る
    views = [_wall_view(0.5), _wall_view(-2.7)]               # 写真 1 は x = -1.1 まで（板の左半分）
    plain = make_surfaces(OUTLINE, BACK, 2.4, views)["walls"][0]
    prot = make_surfaces(OUTLINE, BACK, 2.4, views, protect={2: [PLATE_BOX]})["walls"][0]
    assert prot["protected"][0]["view"] is None and np.array_equal(prot["image"], plain["image"])
