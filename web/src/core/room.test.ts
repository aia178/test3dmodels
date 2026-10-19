import { describe, expect, it } from 'vitest';
import { accuracyTable, addMeasurement, applyJobResult, moveEdge, newFixture, newRoomBase, setManualRect, upsertFixture } from './room';
import { fixturePosition, judgeLayout, roomTolerance } from './judge';
import { dist, signedArea, v } from './geom';

let n = 0;
const id = () => `id${n++}`;

function photoRoom() {
  // 実際は 4.0m × 3.0m の部屋が、5% 小さく復元されたとする
  const s = 0.95;
  return applyJobResult(newRoomBase('r', 'r', '', 'viewing'), 'job', {
    outline: [[0, 0], [4 * s, 0], [4 * s, 3 * s], [0, 3 * s]], edge_status: ['observed', 'observed', 'unobserved', 'observed'],
    ceiling_height: 2.4 * s, ceiling_observed: true, obstacles: [{ kind: 'unknown_vertical', rect: [1 * s, 1 * s, 1.3 * s, 1.3 * s], yMin: 0, yMax: 2 * s }],
    meta: { model: 'm', n_views: 8 },
  }, id);
}

describe('写真から作った部屋の取り込み', () => {
  it('縮尺は未確認、観測できなかった壁は未確認のまま、版が上がる', () => {
    const r = photoRoom();
    expect(r.scale.status).toBe('model_estimate');
    expect(r.edges[2].status).toBe('unobserved');
    expect(r.obstacles[0].confirm).toBe('auto');
    expect(r.geometryVersion).toBe(1);
    expect(roomTolerance(r).tol).toBeCloseTo(0.06 * 3.8); // いちばん長い辺 3.8m の 6%
  });
  it('時計回りの輪郭は受け付けない（左右反転の検出）', () => {
    expect(() => applyJobResult(newRoomBase('r', 'r', '', 'current'), 'j', {
      outline: [[0, 0], [0, 3], [4, 3], [4, 0]], edge_status: ['observed', 'observed', 'observed', 'observed'], ceiling_height: null, ceiling_observed: false, obstacles: [], meta: { model: 'm', n_views: 3 },
    }, id)).toThrow();
  });
});

describe('縮尺合わせと検証', () => {
  it('1 か所の実測で全体が同じ比率で変わり、人が入れた寸法は変わらない', () => {
    let r = photoRoom();
    r = upsertFixture(r, newFixture('o', 'outlet', 0, 1.9, false)); // 幅 7cm は人が入れた実寸
    const before = fixturePosition(r, r.fixtures[0]).center;
    r = addMeasurement(r, { id: 'c', label: '壁1', a: r.outline[0], b: r.outline[1], measuredM: 4.0, role: 'calibration' });
    expect(r.scale.status).toBe('calibrated');
    expect(dist(r.outline[0], r.outline[1])).toBeCloseTo(4.0, 6);
    expect(dist(r.outline[1], r.outline[2])).toBeCloseTo(3.0, 6);
    expect(r.ceiling.height).toBeCloseTo(2.4, 6);
    expect(r.obstacles[0].rect[2] - r.obstacles[0].rect[0]).toBeCloseTo(0.3, 6);
    expect(r.fixtures[0].width).toBeCloseTo(0.07, 9);
    expect(fixturePosition(r, r.fixtures[0]).center.x).toBeCloseTo(before.x / 0.95, 6);
    expect(r.scale.factor).toBeCloseTo(1 / 0.95, 6);
    expect(signedArea(r.outline)).toBeGreaterThan(0);
  });
  it('縮尺合わせに使った測定は精度の評価に数えない', () => {
    let r = photoRoom();
    r = addMeasurement(r, { id: 'c', label: '壁1', a: r.outline[0], b: r.outline[1], measuredM: 4.0, role: 'calibration' });
    expect(accuracyTable(r).nValidation).toBe(0);
    expect(roomTolerance(r).tol).toBeCloseTo(0.05);
    r = addMeasurement(r, { id: 'v1', label: '壁2', a: r.outline[1], b: r.outline[2], measuredM: 3.04, role: 'validation' });
    const t = accuracyTable(r);
    expect(t.nValidation).toBe(1);
    expect(t.maxAbs).toBeCloseTo(0.04, 6);
    expect(roomTolerance(r).tol).toBeCloseTo(0.04, 6);
  });
  it('形や縮尺が変わると版が上がり、古い判定結果と区別できる', () => {
    let r = setManualRect(newRoomBase('m', 'm', '', 'current'), 4, 3, 2.4);
    const res = judgeLayout(r, [], { id: 'l', roomId: 'm', name: '', items: [], updatedAt: '' });
    r = moveEdge(r, 0, 0.05);
    expect(r.geometryVersion).toBeGreaterThan(res.geometryVersion);
    expect(r.outline[0].z).toBeCloseTo(0.05);
    expect(r.edges[0].status).toBe('confirmed');
  });
  it('極端な補正（測り間違いの疑い）は断る', () => {
    const r = photoRoom();
    expect(() => addMeasurement(r, { id: 'c', label: 'x', a: v(0, 0), b: v(0.06, 0), measuredM: 10, role: 'calibration' })).toThrow();
  });
});

import { measurementLength } from './room';
import { lookupXyz } from '../ui/PhotoMeasure';

describe('写真の上で選んだ 2 点の測定', () => {
  it('高さのある 2 点は 3D の距離で測る（扉の高さのような縦の長さ）', () => {
    expect(measurementLength({ a: v(1, 1), b: v(1, 1), ya: 0, yb: 2.19 })).toBeCloseTo(2.19);
    expect(measurementLength({ a: v(0, 0), b: v(3, 0), ya: 1, yb: 5 })).toBeCloseTo(5);
    expect(measurementLength({ a: v(0, 0), b: v(3, 4) })).toBeCloseTo(5);
  });
  it('縦の長さでも縮尺を合わせられ、高さも同じ比率で変わる', () => {
    let r = photoRoom();
    r = addMeasurement(r, { id: 'c', label: '扉の高さ', a: v(1, 0), b: v(1, 0), ya: 0.02, yb: 2.3, measuredM: 2.19, role: 'calibration' });
    const m = r.measurements[0];
    expect(measurementLength(m)).toBeCloseTo(2.19, 6);
    expect(r.scale.factor).toBeCloseTo(2.19 / 2.28, 6);
    expect(m.ya! / 0.02).toBeCloseTo(r.scale.factor, 6);
  });
  it('画素から 3D の位置を引く: mm の int16、値なしは飛ばし、縮尺を掛ける', () => {
    const w = 6, h = 6, a = new Int16Array(w * h * 3).fill(-32768);
    for (let y = 1; y < 6; y++) for (let x = 1; x < 6; x++) { const i = (y * w + x) * 3; a[i] = 1000; a[i + 1] = 500; a[i + 2] = -2000; }
    a[(3 * w + 3) * 3] = 30000; // 1 画素だけのはずれ値は中央値で消える
    expect(lookupXyz(a, w, h, 3, 3, 1)).toEqual({ x: 1, y: 0.5, z: -2 });
    expect(lookupXyz(a, w, h, 3, 3, 0.96)!.x).toBeCloseTo(0.96);
    expect(lookupXyz(new Int16Array(w * h * 3).fill(-32768), w, h, 3, 3, 1)).toBeNull();
  });
});

import { calibrateByCeiling } from './room';

describe('天井高で大きさを合わせる', () => {
  it('復元した天井高と実際の天井高の比で、部屋全体が変わる', () => {
    const r0 = photoRoom(); // 天井 2.4m が 5% 小さく復元されている
    const r = calibrateByCeiling(r0, 2.4);
    expect(r.scale.status).toBe('calibrated');
    expect(r.ceiling).toEqual({ height: 2.4, status: 'manual' });
    expect(dist(r.outline[0], r.outline[1])).toBeCloseTo(4.0, 6);
    expect(r.geometryVersion).toBeGreaterThan(r0.geometryVersion);
  });
  it('天井が復元できていない部屋では使えない', () => {
    const r = { ...photoRoom(), ceiling: { height: null, status: 'unobserved' as const } };
    expect(() => calibrateByCeiling(r, 2.4)).toThrow();
  });
});

describe('復元の質が要注意の部屋', () => {
  const q = { level: 'poor' as const, reasons: ['disagreement'], disagreementMm: 15.2, overlap: 0.95, nPhotos: 13 };
  it('誤差を大きく見込み、縮尺を合わせても形のゆがみを見込む', () => {
    const r = { ...photoRoom(), captureQuality: q };
    expect(roomTolerance(r).tol).toBeCloseTo(0.15 * 3.8);
    const c = addMeasurement(r, { id: 'c', label: '壁1', a: r.outline[0], b: r.outline[1], measuredM: 4.0, role: 'calibration' });
    expect(roomTolerance(c).tol).toBeCloseTo(0.08 * 4.0);
    expect(roomTolerance(c).basis).toContain('撮り直し');
  });
  it('検証の実測があれば、その差を優先する', () => {
    let r: ReturnType<typeof photoRoom> = { ...photoRoom(), captureQuality: q };
    r = addMeasurement(r, { id: 'c', label: '壁1', a: r.outline[0], b: r.outline[1], measuredM: 4.0, role: 'calibration' });
    r = addMeasurement(r, { id: 'v', label: '壁2', a: r.outline[1], b: r.outline[2], measuredM: 3.03, role: 'validation' });
    expect(roomTolerance(r).tol).toBeCloseTo(0.03, 6);
  });
  it('配置の確認結果に、撮り直しを勧める項目が出る', () => {
    const r = { ...photoRoom(), captureQuality: q };
    const res = judgeLayout(r, [], { id: 'l', roomId: 'r', name: '', items: [], updatedAt: '' });
    expect(res.issues.some((i) => i.title.includes('信頼しにくい'))).toBe(true);
  });
});

describe('復元の質が特に悪い部屋（近距離・部分写真など）', () => {
  it('1 か所合わせても、形のゆがみを大きく見込む', () => {
    const q = { level: 'poor' as const, reasons: ['disagreement', 'focal'], disagreementMm: 15.7, overlap: 0.95, nPhotos: 13 };
    const r = { ...photoRoom(), captureQuality: q };
    expect(roomTolerance(r).tol).toBeCloseTo(0.2 * 3.8);
    const c = addMeasurement(r, { id: 'c', label: '壁1', a: r.outline[0], b: r.outline[1], measuredM: 4.0, role: 'calibration' });
    expect(roomTolerance(c).tol).toBeCloseTo(0.2 * 4.0);
  });
});

describe('写真が多く、範囲に分けて復元してつないだ部屋', () => {
  const q = { level: 'good' as const, reasons: [], disagreementMm: 6.0, overlap: 0.95, nPhotos: 26, chunks: 3 };
  it('質の判定が良好でも、つなぎ目のずれを見込んで誤差を広く取る', () => {
    const r = { ...photoRoom(), captureQuality: q };
    const t = roomTolerance(r);
    expect(t.tol).toBeCloseTo(0.15 * 3.8);
    expect(t.basis).toContain('3 つの範囲');
    const c = addMeasurement(r, { id: 'c', label: '壁1', a: r.outline[0], b: r.outline[1], measuredM: 4.0, role: 'calibration' });
    expect(roomTolerance(c).tol).toBeCloseTo(0.08 * 4.0);
  });
  it('1 回で復元した良好な部屋は、これまでどおり', () => {
    const r = { ...photoRoom(), captureQuality: { ...q, chunks: undefined } };
    expect(roomTolerance(r).tol).toBeCloseTo(0.06 * 3.8);
  });
});

import { surfacesUsable } from './room';

describe('壁・床に写真を貼った画像', () => {
  const withSurfaces = () => applyJobResult(newRoomBase('r', 'r', '', 'viewing'), 'job', {
    outline: [[0, 0], [4, 0], [4, 3], [0, 3]], edge_status: ['observed', 'observed', 'unobserved', 'observed'],
    ceiling_height: 2.4, ceiling_observed: true, obstacles: [], meta: { model: 'm', n_views: 8 },
    surfaces: { walls: [{ edge: 0, file: 'wall_0.jpg', coverage: 0.9 }], floor: { file: 'floor.jpg', coverage: 0.3, bbox: [0, 0, 4, 3] } },
  }, () => 'x');
  it('復元結果から取り込み、そのままの形なら貼れる', () => {
    const r = withSurfaces();
    expect(r.surfaces?.walls).toEqual([{ edge: 0, file: 'wall_0.jpg' }]);
    expect(surfacesUsable(r)).toBe(true);
  });
  it('1 か所測って大きさを合わせると、画像の位置も一緒に伸び縮みし、貼れるまま', () => {
    const r = addMeasurement(withSurfaces(), { id: 'c', label: '壁', a: v(0, 0), b: v(4, 0), measuredM: 4.2, role: 'calibration' });
    expect(surfacesUsable(r)).toBe(true);
    expect(r.surfaces!.floor!.bbox[2]).toBeCloseTo(4.2, 6);
  });
  it('壁を手で動かした後は、画像が合わないので貼らない', () => {
    const r = moveEdge(withSurfaces(), 1, 0.3);
    expect(surfacesUsable(r)).toBe(false);
  });
});

describe('印刷マーカーでの縮尺合わせ', () => {
  const job = (marker_scale?: { factor: number | null; n: number; spread_pct?: number; size_m: number }) => applyJobResult(newRoomBase('r', 'r', '', 'viewing'), 'job', {
    outline: [[0, 0], [4.2, 0], [4.2, 3.15], [0, 3.15]], edge_status: ['observed', 'observed', 'observed', 'observed'],
    ceiling_height: 2.52, ceiling_observed: true, obstacles: [], meta: { model: 'm', n_views: 8 }, marker_scale,
  }, id);

  it('2 か所以上でそろって測れたら、その倍率で部屋全体を合わせ、マーカーで合わせたと記録する', () => {
    const r = job({ factor: 1 / 1.05, n: 3, spread_pct: 2.1, size_m: 0.15 });
    expect(r.scale.status).toBe('marker');
    expect(r.outline[1].x).toBeCloseTo(4.0);
    expect(r.ceiling.height).toBeCloseTo(2.4);
  });

  it('1 か所だけ・ばらつきが大きい・測れなかったときは当てない', () => {
    expect(job({ factor: 0.95, n: 1, spread_pct: 0, size_m: 0.15 }).scale.status).toBe('model_estimate');
    expect(job({ factor: 0.95, n: 4, spread_pct: 12, size_m: 0.15 }).scale.status).toBe('model_estimate');
    expect(job({ factor: null, n: 0, size_m: 0.15 }).scale.status).toBe('model_estimate');
    expect(job().outline[1].x).toBeCloseTo(4.2);
  });
});
