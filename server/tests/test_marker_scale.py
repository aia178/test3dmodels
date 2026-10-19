"""印刷マーカーでの縮尺合わせ（roomfit.marker_scale）のテスト。作り物のカメラ・壁・写真を使う（実物での精度の証明ではない）。"""
import numpy as np

from roomfit.marker_scale import estimate_scale
from roomfit.textures import ViewData, photo_mapping

WM, HM, FM = 320, 240, 200.0          # モデルの画像
WP, HP = 1280, 960                    # 元の写真（モデルの 4 倍）
FP = FM * 4


def _view(marker_center=(0.1, 1.25), size=0.15, wall_z=1.5, recon_scale=1.04, noise=0.003, with_marker=True):
    """部屋の (0, 1.2, 0) から +z 向きに、z = wall_z の壁を見るカメラ。壁にマーカーを貼った写真と、
    実物より recon_scale 倍に復元された 3D（モデルの画素ごと）を作る。"""
    import cv2

    c2r = np.eye(4); c2r[:3, :3] = np.diag([-1.0, -1.0, 1.0]); c2r[:3, 3] = [0.0, 1.2, 0.0]
    rng = np.random.default_rng(1)
    photo = np.clip(200 + rng.normal(0, 6, (HP, WP)), 0, 255).astype(np.uint8)
    if with_marker:
        d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        tag = np.full((600, 600), 255, np.uint8); tag[100:500, 100:500] = cv2.aruco.generateImageMarker(d, 3, 400)
        cx, cy, a = marker_center[0], marker_center[1], size / 2
        # 画像の右 = 部屋の -x、画像の下 = 部屋の -y
        room = np.array([[cx + a, cy + a], [cx - a, cy + a], [cx - a, cy - a], [cx + a, cy - a]])
        px = np.c_[-room[:, 0] / wall_z * FP + WP / 2, -(room[:, 1] - 1.2) / wall_z * FP + HP / 2].astype(np.float32)
        Hm = cv2.getPerspectiveTransform(np.float32([[100, 100], [500, 100], [500, 500], [100, 500]]), px)
        warped = cv2.warpPerspective(tag, Hm, (WP, HP), borderValue=0)
        inside = cv2.warpPerspective(np.full_like(tag, 255), Hm, (WP, HP), borderValue=0) > 0
        photo[inside] = warped[inside]
    u, v = np.meshgrid(np.arange(WM) + 0.5, np.arange(HM) + 0.5)
    true = np.stack([-(u - WM / 2) / FM * wall_z, 1.2 - (v - HM / 2) / FM * wall_z, np.full(u.shape, wall_z)], -1)
    grid = true * recon_scale + rng.normal(0, noise, true.shape)
    K = np.array([[FM, 0, WM / 2], [0, FM, HM / 2], [0, 0, 1.0]])
    return ViewData(cam2room=c2r, K=K, grid_room=grid, photo=np.repeat(photo[..., None], 3, -1),
                    to_photo=photo_mapping((HM, WM), (WM, HM), (WP, HP)))


def test_marker_brings_reconstruction_back_to_real_size():
    r = estimate_scale([_view(), _view(marker_center=(-0.3, 1.0))])
    assert r["n"] == 2 and abs(r["factor"] - 1 / 1.04) < 0.01        # 4% 大きい復元 → 倍率 約 0.962
    assert r["spread_pct"] < 2


def test_too_small_marker_is_not_used():
    # 4m 先の 8cm のマーカーは、モデルの画像で一辺 4 画素しかない
    r = estimate_scale([_view(size=0.08, wall_z=4.0)])
    assert r is None or (r["factor"] is None and r["n"] == 0)


def test_no_marker_gives_none():
    assert estimate_scale([_view(with_marker=False)]) is None
