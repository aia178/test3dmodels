"""AR の連続記録からのコマ選び（roomfit.frame_select）のテスト。作り物のカメラを使う。"""
import numpy as np

from roomfit.frame_select import chain, overlap_matrix, select_frames

K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])
WH = (640, 480)


def _turning(n, step_deg, speed_up_at=None):
    """その場で y 軸まわりに向きを変えるカメラ。speed_up_at 以降は 3 倍の速さで回る（速く振った所）。"""
    out, a = [], 0.0
    for i in range(n):
        c = np.eye(4)
        r = np.radians(a)
        c[:3, :3] = [[np.cos(r), 0, np.sin(r)], [0, 1, 0], [-np.sin(r), 0, np.cos(r)]]
        out.append(c)
        a += step_deg * (3 if speed_up_at is not None and i >= speed_up_at else 1)
    return np.array(out)


def test_overlap_is_one_for_same_view_and_falls_with_turning():
    M = overlap_matrix(_turning(10, 10), K, WH)
    assert np.allclose(np.diag(M), 1) and np.allclose(M, M.T)
    assert M[0, 1] > M[0, 3] > M[0, 6]
    assert M[0, 9] == 0                          # 90 度向きが違えば重ならない（水平の画角は約 65 度）


def test_every_gap_keeps_minimum_overlap_even_where_the_camera_turned_fast():
    c = _turning(60, 2, speed_up_at=40)          # 終わりの 20 コマは速く振った
    r = select_frames(c, K, WH, min_overlap=0.4)
    M = overlap_matrix(c, K, WH)
    assert r["idx"][0] == 0 and r["idx"][-1] == 59 and not r["capped"]
    assert all(M[a, b] >= 0.4 for a, b in zip(r["idx"], r["idx"][1:]))
    # 速く振った所ほど密に選ぶ（時間で等間隔ではない）
    gaps = np.diff(r["idx"])
    assert gaps[-1] < gaps[0]


def test_max_frames_caps_the_count_and_maximizes_the_smallest_overlap():
    c = _turning(80, 3)
    full = select_frames(c, K, WH, min_overlap=0.8)
    capped = select_frames(c, K, WH, min_overlap=0.8, max_frames=8)
    assert len(full["idx"]) > 8 and len(capped["idx"]) == 8 and capped["capped"]
    assert capped["idx"][0] == 0 and capped["idx"][-1] == 79
    even = np.round(np.linspace(0, 79, 8)).astype(int)
    M = overlap_matrix(c, K, WH)
    assert capped["overlap_min"] >= min(M[a, b] for a, b in zip(even, even[1:])) - 1e-9


def test_chain_does_not_jump_to_a_revisit_at_the_end():
    # 一周して始めと同じ向きに戻る: 終わりのコマは始めと重なるが、途中を飛ばしてはいけない
    c = _turning(37, 10)                         # 0〜360 度
    s = chain(overlap_matrix(c, K, WH), 0.3)
    assert len(s) > 3 and np.diff(s).max() < 10
