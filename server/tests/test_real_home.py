"""実データ（自宅の一角・写真 13 枚の復元結果）での回帰テスト。

復元結果のファイル（experiments/out/home_1012/raw.npz）は室内写真に由来するためリポジトリに入れていない。
ファイルがある PC でだけ実行され、無ければスキップする。形状抽出を変えたときに、実測と合っていた結果を壊していないかを見る。
"""
from pathlib import Path

import numpy as np
import pytest

RAW = Path(__file__).resolve().parents[2] / "experiments" / "out" / "home_1012" / "raw.npz"
pytestmark = pytest.mark.skipif(not RAW.exists(), reason="実データの復元結果がこの PC に無い")


@pytest.fixture(scope="module")
def room(tmp_path_factory):
    from roomfit.reconstruct import load_raw
    from roomfit.worker import structure_from_raw

    return structure_from_raw(load_raw(RAW), tmp_path_factory.mktemp("home"))


def test_ceiling_matches_the_ceiling_pixels(room):
    # 天井が写っている画素の高さは 2.46〜2.47m（縮尺未補正）。以前は 2.41m と出る不具合があった
    assert room["ceiling_observed"] and 2.44 < room["ceiling_height"] < 2.50


def test_corner_is_detected_and_rectified(room):
    d = room["diagnostics"]
    assert d["rectified"] and 83 < d["wall_corner_angle_deg"] < 88
    R = np.array(room["T_room_from_world"])[:3, :3]
    assert np.linalg.det(R) > 0.9  # 鏡映なし


def test_two_walls_observed_and_rest_unknown(room):
    st = room["edge_status"]
    assert st.count("observed") >= 2 and "unobserved" in st
    ol = np.array(room["outline"])
    lens = sorted(float(np.linalg.norm(ol[(i + 1) % len(ol)] - ol[i])) for i in range(len(ol)) if st[i] == "observed")
    assert 1.2 < lens[0] < 1.4      # クローゼットのある壁（撮れた範囲）: 約 1.3m
    assert lens[-1] > 2.0           # 引き戸のある壁（撮れた範囲）: 2m 以上


def test_capture_quality_is_good(room):
    q = room["capture_quality"]
    assert q["level"] == "good" and q["disagreement_mm"] < 11 and q["overlap"] > 0.95
