import { describe, expect, it } from 'vitest';
import { snapToEdge, XYZ_INVALID, type SnapImage } from './snap';

// 幅 100×高さ 40 の画像。x が 30〜70 の範囲が暗い「扉」（明るさ 80）、ほかは白い壁（220）。
// 3D: 扉は壁より 5cm 手前。1 画素 = 1cm。端の画素は手前と奥が混ざった値にしておく（実データの癖を模す）。
function scene(): SnapImage {
  const w = 100, h = 40, lum = new Float32Array(w * h), xyz = new Int16Array(w * h * 3);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const door = x >= 30 && x < 70, i = y * w + x;
    lum[i] = door ? 80 : 220;
    xyz[i * 3] = (x + 0.5) * 10; xyz[i * 3 + 1] = (h - y - 0.5) * 10; xyz[i * 3 + 2] = door ? 1950 : 2000;
    if (x === 29 || x === 30 || x === 69 || x === 70) xyz[i * 3 + 2] = 1975;
  }
  return { w, h, lum, xyz };
}

describe('端への吸い付け', () => {
  it('タップが数画素ずれていても、扉の両端（x=30, x=70）へ寄り、幅 40cm が出る', () => {
    const img = scene();
    for (const [da, db] of [[-3, 2], [3, -4], [0, 0], [4, 4], [-10, 9], [8, -10]]) {
      const A = { u: 30 + da, v: 20 }, B = { u: 70 + db, v: 21 };
      const a = snapToEdge(img, A, B)!, b = snapToEdge(img, B, A)!;
      expect(a.u).toBeCloseTo(30, 0); expect(b.u).toBeCloseTo(70, 0);
      expect(Math.abs(Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z) - 0.4)).toBeLessThan(0.012);
      expect(a.z).toBeCloseTo(1.95, 2); // 端の混ざった値(1.975)ではなく、扉の面の奥行き
    }
  });
  it('端が無い所（一面の白い壁）では null を返し、作り物の値を出さない', () => {
    const img = scene();
    expect(snapToEdge(img, { u: 85, v: 20 }, { u: 95, v: 20 })).toBeNull();
  });
  it('内側の 3D 位置が取れていない場合も null', () => {
    const img = scene();
    for (let i = 0; i < img.xyz.length; i++) img.xyz[i] = XYZ_INVALID;
    expect(snapToEdge(img, { u: 30, v: 20 }, { u: 70, v: 20 })).toBeNull();
  });
  it('2 点が近すぎるときは使わない', () => {
    expect(snapToEdge(scene(), { u: 30, v: 20 }, { u: 34, v: 20 })).toBeNull();
  });
});

import { findNearest, measureInOtherViews, median } from './snap';

describe('ほかの写真でも測ってまとめる', () => {
  // 同じ扉を写した別の写真: 3 画素横にずれて写り、復元の大きさが 5% 大きい
  function otherView(): SnapImage {
    const s = scene(), w = s.w, h = s.h, lum = new Float32Array(w * h), xyz = new Int16Array(w * h * 3);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const sx = Math.min(w - 1, Math.max(0, x - 3)), i = y * w + x, j = y * w + sx;
      lum[i] = s.lum[j];
      // 扉の左端(x=300mm)を基準に 5% 引き伸ばす
      xyz[i * 3] = Math.round(300 + (s.xyz[j * 3] - 300) * 1.05); xyz[i * 3 + 1] = s.xyz[j * 3 + 1]; xyz[i * 3 + 2] = s.xyz[j * 3 + 2];
    }
    return { w, h, lum, xyz };
  }
  it('3D の位置から、別の写真の中の同じ場所の画素を見つける', () => {
    const q = findNearest(scene(), { x: 0.505, y: 0.195, z: 1.95 })!;
    expect(q.u).toBeCloseTo(50.5, 0); expect(q.v).toBeCloseTo(20.5, 0);
    expect(findNearest(scene(), { x: 5, y: 5, z: 5 })).toBeNull();
  });
  it('別の写真での長さが出て、中央値でまとめられる', () => {
    const img = scene();
    const a = snapToEdge(img, { u: 31, v: 20 }, { u: 69, v: 20 })!, b = snapToEdge(img, { u: 69, v: 20 }, { u: 31, v: 20 })!;
    const ws = measureInOtherViews([otherView()], a, b);
    expect(ws.length).toBe(1);
    expect(ws[0]).toBeGreaterThan(0.405); expect(ws[0]).toBeLessThan(0.435); // 5% 大きい写真では約 42cm
    expect(median([0.4, ...ws])).toBeCloseTo((0.4 + ws[0]) / 2, 6);
    expect(median([3, 1, 2])).toBe(2);
  });
  it('その場所が写っていない写真は数えない', () => {
    const img = scene(), far = scene();
    for (let i = 0; i < far.xyz.length; i += 3) if (far.xyz[i] !== XYZ_INVALID) far.xyz[i] += 5000; // 5m 離れた別の場所
    const a = snapToEdge(img, { u: 31, v: 20 }, { u: 69, v: 20 })!, b = snapToEdge(img, { u: 69, v: 20 }, { u: 31, v: 20 })!;
    expect(measureInOtherViews([far], a, b)).toEqual([]);
  });
});
