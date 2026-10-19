"""印刷用の縮尺マーカー（A4、AprilTag 36h11、黒い四角の一辺 150mm）の PDF を作る。

出力: docs/markers/roomfit_markers_A4.pdf（番号 0〜5 を 1 ページずつ）
寸法を正確にするため、黒い四角を 1600 画素（1 マス 200 画素 × 8 マス）にし、PDF の解像度をそれが 150mm になる値にしている。
印刷の大きさは、カード（ISO の ID-1、85.60×53.98mm）を重ねる枠で確かめてもらう（アプリはメジャーを使わない前提のため）。

使い方（server フォルダで）: uv run python ../docs/markers/make_markers.py
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "server"))
from roomfit.marker_scale import MARKER_SIZE_M  # noqa: E402

TAG_PX = 1600
DPI = TAG_PX / (MARKER_SIZE_M * 1000 / 25.4)          # 約 270.9
CARD_MM = (85.60, 53.98)
IDS = range(6)
FONT = "C:/Windows/Fonts/BIZ-UDGothicR.ttc"
FONT_B = "C:/Windows/Fonts/BIZ-UDGothicB.ttc"


def px(mm: float) -> int:
    return round(mm / 25.4 * DPI)


def page(tag_id: int) -> Image.Image:
    import cv2

    W, H = px(210), px(297)
    im = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(im)
    f_title, f_text, f_small = ImageFont.truetype(FONT_B, px(6)), ImageFont.truetype(FONT, px(3.6)), ImageFont.truetype(FONT, px(3.0))
    d.text((px(15), px(10)), f"RoomFit 縮尺マーカー  No.{tag_id}", font=f_title, fill=0)
    lines = [
        "1. 印刷は「実際のサイズ（倍率 100%）」で。「用紙に合わせる」「縮小」は使わない。",
        "2. 左下の枠にクレジットカードや交通系 IC カードを重ねて、ぴったり合うか確かめる。",
        "    合わなければ、印刷の設定を見直して印刷し直す（大きさがずれると、部屋の大きさも同じだけずれる）。",
        "3. 平らな壁に、しわ・浮き・折り目がないように貼る。番号の違う紙を、部屋の別々の壁に 2〜4 枚。",
    ]
    for i, t in enumerate(lines):
        d.text((px(15), px(20) + i * px(5.2)), t, font=f_text, fill=0)
    tag = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), tag_id, TAG_PX)
    x0, y0 = (W - TAG_PX) // 2, px(64)                 # 周りに 1 マス（18.75mm）以上の白を残す
    im.paste(Image.fromarray(tag), (x0, y0))
    # カードを重ねる枠（線の内側の寸法がカードの大きさ）
    cx0, cy0 = px(15), px(238)
    lw = max(2, px(0.3))
    d.rectangle([cx0 - lw, cy0 - lw, cx0 + px(CARD_MM[0]) + lw - 1, cy0 + px(CARD_MM[1]) + lw - 1], outline=0, width=lw)
    d.text((cx0 + px(4), cy0 + px(20)), "カードをここに重ねる", font=f_text, fill=0)
    d.text((cx0 + px(4), cy0 + px(27)), "（85.6 × 54.0 mm）", font=f_small, fill=0)
    notes = [
        f"黒い四角の一辺 {MARKER_SIZE_M * 1000:.0f} mm",
        f"AprilTag 36h11  番号 {tag_id}",
        "",
        "撮るときは、マーカー全体が",
        "はっきり写るように（斜めすぎない・",
        "2m くらいまでの距離から）。",
        "マーカーが無くても部屋は復元できます",
        "（大きさは推定のまま）。",
    ]
    for i, t in enumerate(notes):
        d.text((px(110), px(240) + i * px(5.4)), t, font=f_text if i < 2 else f_small, fill=0)
    return im


def main() -> None:
    pages = [page(i) for i in IDS]
    out = Path(__file__).resolve().parent / "roomfit_markers_A4.pdf"
    pages[0].save(out, "PDF", resolution=DPI, save_all=True, append_images=pages[1:])
    # 寸法の確認: 黒い四角とカード枠の大きさ（mm）
    a = np.asarray(pages[0])
    ys, xs = np.nonzero(a[px(60):px(232), :] < 128)
    print(f"保存: {out}（{len(pages)} ページ、解像度 {DPI:.2f}dpi）")
    print(f"黒い四角: {(xs.max() - xs.min() + 1) / DPI * 25.4:.2f} × {(ys.max() - ys.min() + 1) / DPI * 25.4:.2f} mm")
    print(f"カード枠の内側: {px(CARD_MM[0]) / DPI * 25.4:.2f} × {px(CARD_MM[1]) / DPI * 25.4:.2f} mm")


if __name__ == "__main__":
    main()
