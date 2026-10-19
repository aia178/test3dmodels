import { describe, expect, it } from 'vitest';
import type { Furniture, Layout, Room } from './types';
import { judgeLayout, qualityReason, roomTolerance } from './judge';
import { convexRelation, convexVsOutline, footprint, frontNormal, rotY, signedArea, v } from './geom';
import { buildPassageGrid, widestPath } from './passage';
import { fmtCm, parseCm } from './units';
import { newRoomBase } from './room';

// 4m × 3m の部屋。右下の角に 30cm × 40cm の柱の出っ張りがある（輪郭の凹み）。
function room(over: Partial<Room> = {}): Room {
  const outline = [v(0, 0), v(3.7, 0), v(3.7, 0.4), v(4, 0.4), v(4, 3), v(0, 3)];
  return {
    id: 'r', name: 't', purpose: '', kind: 'current', source: 'manual', createdAt: '', updatedAt: '',
    outline, edges: outline.map(() => ({ status: 'manual' as const })),
    ceiling: { height: 2.4, status: 'manual' }, obstacles: [], fixtures: [], measurements: [],
    scale: { factor: 1, status: 'manual' }, geometryVersion: 1, hasPointCloud: false, history: [], ...over,
  };
}
function furn(over: Partial<Furniture> = {}): Furniture {
  return {
    id: 'f1', name: '棚', category: 'shelf', w: 0.8, d: 0.4, h: 1.8, dimSource: 'measured', dimConfirmed: true, shape: 'box',
    legClearance: null, ownership: 'owned', useSpace: { kind: 'none', depth: 0 }, photoIds: [], color: '#888', createdAt: '', ...over,
  };
}
const layout = (items: Layout['items']): Layout => ({ id: 'l', roomId: 'r', name: 'a', items, updatedAt: '' });
const place = (x: number, z: number, rot = 0, id = 'p1', furnitureId = 'f1') => ({ id, furnitureId, x, z, rotY: rot, elevation: 0 });

describe('単位と座標', () => {
  it('cm 入力を m に変換する（全角も）', () => {
    expect(parseCm('120')).toBeCloseTo(1.2);
    expect(parseCm('４５．５')).toBeCloseTo(0.455);
    expect(parseCm('abc')).toBeNull();
    expect(parseCm('-3')).toBeNull();
  });
  it('推定値は cm の整数で表示する', () => expect(fmtCm(0.0549)).toBe('5cm'));
  it('rotY は Three.js と同じ向き（+90° で +x が -z へ）', () => {
    const p = rotY(v(1, 0), Math.PI / 2);
    expect(p.x).toBeCloseTo(0); expect(p.z).toBeCloseTo(-1);
  });
  it('回転 0 の家具の正面は +z、輪郭は反時計回り', () => {
    expect(frontNormal(place(0, 0))).toEqual(v(0, 1));
    expect(signedArea(room().outline)).toBeGreaterThan(0);
    expect(signedArea(footprint(place(1, 1, 0.3), furn()))).toBeGreaterThan(0);
  });
  it('左右が反転しない: 幅方向 +x の角は回転 0 で x が大きい側にある', () => {
    const fp = footprint(place(2, 1), furn({ w: 1, d: 0.2 }));
    expect(Math.max(...fp.map((p) => p.x))).toBeCloseTo(2.5);
    expect(Math.max(...fp.map((p) => p.z))).toBeCloseTo(1.1);
  });
});

describe('置けるか', () => {
  it('壁に重なる位置は問題あり、十分離すと解消する', () => {
    const r = room(), f = [furn()];
    const bad = judgeLayout(r, f, layout([place(0.3, 1.5, Math.PI / 2)])); // 幅 0.8 を z 方向に、奥行き 0.4 → x は 0.1〜0.5 … 壁から 10cm 内側
    expect(bad.issues.find((i) => i.check === 'wall')).toBeUndefined();
    const hit = judgeLayout(r, f, layout([place(0.1, 1.5, Math.PI / 2)])); // x は -0.1〜0.3 → 10cm はみ出し
    const w = hit.issues.find((i) => i.check === 'wall')!;
    expect(w.status).toBe('problem');
    expect(w.dims[0].valueM).toBeCloseTo(0.1, 2);
    // 提案どおりに動かすと解消する
    const fixed = place(0.1 + w.fix!.dx, 1.5 + w.fix!.dz, Math.PI / 2);
    expect(judgeLayout(r, f, layout([fixed])).issues.find((i) => i.check === 'wall')).toBeUndefined();
  });
  it('柱の出っ張り（輪郭の凹み）に当たる配置を検出する', () => {
    // 棚を右下の角へ。出っ張りが無ければ収まる位置だが、柱の角が家具に食い込む
    const res = judgeLayout(room(), [furn()], layout([place(3.55, 0.25)])); // x 3.15〜3.95, z 0.05〜0.45
    expect(res.issues.find((i) => i.check === 'wall')?.status).toBe('problem');
  });
  it('誤差を見込むと逆転し得る隙間は要確認にする', () => {
    const r = room({ source: 'photo', scale: { factor: 1, status: 'calibrated' }, edges: room().edges.map(() => ({ status: 'confirmed' as const })) });
    expect(roomTolerance(r).tol).toBeCloseTo(0.05);
    const res = judgeLayout(r, [furn()], layout([place(1, 0.23)])); // 壁との隙間 3cm < 誤差 5cm
    expect(res.issues.find((i) => i.check === 'wall')?.status).toBe('needs_check');
  });
  it('未確認の壁へのはみ出しは、問題なしにも問題ありにもせず要確認にする', () => {
    const r = room();
    r.edges[5] = { status: 'unobserved' }; // x=0 の壁
    const res = judgeLayout(r, [furn()], layout([place(0.1, 1.5, Math.PI / 2)]));
    const w = res.issues.find((i) => i.check === 'wall')!;
    expect(w.status).toBe('needs_check');
    expect(w.unverified.join('')).toContain('未確認');
    expect(res.issues.some((i) => i.check === 'input' && i.title.includes('形の分からない壁'))).toBe(true);
  });
  it('寸法が未確認の家具は問題ありと断定しない', () => {
    const res = judgeLayout(room(), [furn({ dimConfirmed: false, dimSource: 'estimated' })], layout([place(0.1, 1.5, Math.PI / 2)]));
    expect(res.issues.find((i) => i.check === 'wall')?.status).toBe('needs_check');
  });
  it('家具どうしの重なりと、高さが重ならない場合', () => {
    const fs = [furn(), furn({ id: 'f2', name: '机', w: 1.2, d: 0.6, h: 0.7 })];
    const res = judgeLayout(room(), fs, layout([place(2, 1.5), place(2.2, 1.6, 0, 'p2', 'f2')]));
    expect(res.issues.find((i) => i.check === 'furniture')?.status).toBe('problem');
    const shelf = { ...place(2.2, 1.6, 0, 'p1', 'f1'), elevation: 1.0 }; // 机(高さ0.7)の上方にある吊り棚
    const res2 = judgeLayout(room({ ceiling: { height: 3, status: 'manual' } }), fs, layout([shelf, place(2.2, 1.6, 0, 'p2', 'f2')]));
    expect(res2.issues.find((i) => i.check === 'furniture')).toBeUndefined();
  });
  it('梁（床から浮いた障害物）は高さが重なる家具だけに当たる', () => {
    const r = room({ obstacles: [{ id: 'b', kind: 'beam', rect: [0, 2.6, 4, 3], yMin: 2.0, yMax: 2.4, confirm: 'confirmed' }] });
    const tall = judgeLayout(r, [furn({ h: 2.1 })], layout([place(2, 2.75)]));
    expect(tall.issues.find((i) => i.check === 'obstacle')?.status).toBe('problem');
    const low = judgeLayout(r, [furn({ h: 1.8 })], layout([place(2, 2.75)]));
    expect(low.issues.find((i) => i.check === 'obstacle')).toBeUndefined();
  });
  it('天井より高い家具', () => {
    const res = judgeLayout(room(), [furn({ h: 2.5 })], layout([place(2, 1.5)]));
    expect(res.issues.find((i) => i.check === 'ceiling')?.status).toBe('problem');
  });
});

describe('使えるか', () => {
  const doorRoom = () => room({
    fixtures: [{ id: 'd', type: 'door', edgeIndex: 5, offset: 1.5, width: 0.8, height: 2, bottom: 0, source: 'manual', confirmed: true,
      door: { hinge: 'start', opensInto: 'room', maxAngleDeg: 90, isEntrance: true } }],
  });
  it('扉の開閉途中の干渉を検出し、開く角度を報告する', () => {
    // 辺5 は (0,3)→(0,0)。中心 offset 1.5 → z=1.5、蝶番は z=1.9。扉は +x 側へ開く
    const res = judgeLayout(doorRoom(), [furn({ w: 0.4, d: 0.4, h: 0.5 })], layout([place(0.6, 1.3)]));
    const d = res.issues.find((i) => i.check === 'door_swing')!;
    expect(d.status).toBe('problem');
    expect(d.reason).toMatch(/約 \d+° までしか開きません/);
    const far = judgeLayout(doorRoom(), [furn({ w: 0.4, d: 0.4, h: 0.5 })], layout([place(2.5, 1.3)]));
    expect(far.issues.find((i) => i.check === 'door_swing')).toBeUndefined();
  });
  it('引き出しの前がふさがれていると問題あり、寸法未入力なら要確認', () => {
    const chest = furn({ id: 'f1', name: 'チェスト', w: 0.8, d: 0.4, h: 0.8, useSpace: { kind: 'drawer', depth: 0.5 } });
    // 正面(+z)が z=3 の壁向き。前の空きは 3 - 2.9 = 10cm しかない
    const res = judgeLayout(room(), [chest], layout([place(2, 2.7)]));
    expect(res.issues.find((i) => i.check === 'use_space')?.status).toBe('problem');
    const unk = judgeLayout(room(), [{ ...chest, useSpace: { kind: 'unknown', depth: 0 } }], layout([place(2, 1.5)]));
    expect(unk.issues.find((i) => i.check === 'use_space')?.status).toBe('needs_check');
  });
  it('通路: 隙間はあっても連続した経路が無ければ到達不可と判定する', () => {
    const r = doorRoom();
    // 部屋を横切る壁のような家具 2 つで、入口側と奥を完全に仕切る（家具間に隙間なし）
    const wallA = furn({ id: 'fa', name: '間仕切りA', w: 0.3, d: 1.6, h: 1.8 });
    const wallB = furn({ id: 'fb', name: '間仕切りB', w: 0.3, d: 1.5, h: 1.8 });
    const bed = furn({ id: 'f1', name: 'ベッド', w: 1, d: 2, h: 0.4 });
    const res = judgeLayout(r, [wallA, wallB, bed], layout([
      place(1.6, 0.8, 0, 'pa', 'fa'), place(1.6, 2.25, 0, 'pb', 'fb'), place(3.4, 1.4, -Math.PI / 2, 'p1', 'f1'),
    ]));
    const p = res.issues.find((i) => i.check === 'passage' && i.targets.includes('p1'))!;
    expect(p.title).toContain('たどり着けません');
  });
  it('通路: 狭い所の幅を経路で評価する', () => {
    const outline = [v(0, 0), v(4, 0), v(4, 3), v(0, 3)];
    // 中央に 50cm の隙間だけ残して仕切る
    const blockers = [[v(1.9, 0), v(2.1, 0), v(2.1, 1.25), v(1.9, 1.25)], [v(1.9, 1.75), v(2.1, 1.75), v(2.1, 3), v(1.9, 3)]];
    const g = buildPassageGrid(outline, blockers);
    const r = widestPath(g, v(0.5, 1.5), v(3.5, 1.5));
    expect(r.reachable).toBe(true);
    expect(r.widthM).toBeGreaterThan(0.42);
    expect(r.widthM).toBeLessThanOrEqual(0.5);
    expect(r.bottleneck!.x).toBeGreaterThan(1.8);
    expect(r.bottleneck!.x).toBeLessThan(2.2);
    // 隙間を閉じると到達不可
    const g2 = buildPassageGrid(outline, [[v(1.9, 0), v(2.1, 0), v(2.1, 3), v(1.9, 3)]]);
    expect(widestPath(g2, v(0.5, 1.5), v(3.5, 1.5)).reachable).toBe(false);
  });
  it('入口が未登録なら通路を「問題なし」とせず、未判定と表示する', () => {
    const res = judgeLayout(room(), [furn()], layout([place(2, 1.5)]));
    expect(res.issues.find((i) => i.check === 'passage')?.title).toContain('判定していません');
  });
});

describe('設備', () => {
  const outletRoom = () => room({
    fixtures: [{ id: 'o', type: 'outlet', edgeIndex: 4, offset: 2, width: 0.07, height: 0.12, bottom: 0.2, source: 'manual', confirmed: true }],
  });
  it('コンセントを家具でふさぐ配置を指摘する', () => {
    // 辺4 は (4,3)→(0,3)。offset 2 → (2,3)。棚の背面を z=3 の壁へ付ける
    const res = judgeLayout(outletRoom(), [furn()], layout([place(2, 2.8, Math.PI)]));
    const o = res.issues.find((i) => i.check === 'fixture')!;
    expect(o.status).toBe('problem');
    expect(o.title).toContain('隠れます');
    expect(o.unverified.join('')).toContain('箱として判定');
  });
  it('離れた家具・高さが重ならない家具では指摘しない', () => {
    expect(judgeLayout(outletRoom(), [furn()], layout([place(0.6, 2.8, Math.PI)])).issues.find((i) => i.check === 'fixture')).toBeUndefined();
    const hung = { ...place(2, 2.8, Math.PI), elevation: 1.0 };
    expect(judgeLayout(outletRoom(), [furn({ h: 0.5 })], layout([hung])).issues.find((i) => i.check === 'fixture')).toBeUndefined();
  });
  it('位置が未確認のコンセントは断定しない', () => {
    const r = outletRoom();
    r.fixtures[0].confirmed = false;
    expect(judgeLayout(r, [furn()], layout([place(2, 2.8, Math.PI)])).issues.find((i) => i.check === 'fixture')?.status).toBe('needs_check');
  });
});

describe('誤差の見込み', () => {
  it('検証用の測定の最大誤差を使い、縮尺合わせに使った測定は数えない', () => {
    const r = room({
      source: 'photo', scale: { factor: 1, status: 'calibrated' },
      measurements: [
        { id: 'c', label: '壁', a: v(0, 0), b: v(3.7, 0), measuredM: 3.7, role: 'calibration', at: '' },
        { id: 'v1', label: '奥行き', a: v(0, 0), b: v(0, 3), measuredM: 2.93, role: 'validation', at: '' },
      ],
    });
    const t = roomTolerance(r);
    expect(t.tol).toBeCloseTo(0.07);
    expect(t.basis).toContain('1 か所');
  });
  it('縮尺未確認の部屋は必ず要確認の項目が出る', () => {
    const r = room({ source: 'photo', scale: { factor: 1, status: 'model_estimate' } });
    expect(judgeLayout(r, [furn()], layout([place(2, 1.5)])).issues.some((i) => i.check === 'input')).toBe(true);
  });
});

describe('幾何の基礎', () => {
  it('凸多角形の重なり深さと隙間', () => {
    const a = [v(0, 0), v(1, 0), v(1, 1), v(0, 1)], b = [v(0.9, 0), v(2, 0), v(2, 1), v(0.9, 1)];
    const r = convexRelation(a, b);
    expect(r.overlap).toBe(true); expect(r.depth).toBeCloseTo(0.1); expect(r.push.x).toBeCloseTo(-1);
    const c = [v(1.3, 0), v(2, 0), v(2, 1), v(1.3, 1)];
    expect(convexRelation(a, c).gap).toBeCloseTo(0.3);
  });
  it('輪郭の内側にあるときの隙間', () => {
    const r = convexVsOutline([v(0.2, 0.2), v(1, 0.2), v(1, 1), v(0.2, 1)], room().outline);
    expect(r.outside).toBe(0); expect(r.gap).toBeCloseTo(0.2);
  });
});

describe('qualityReason', () => {
  it('AR の動きが小さいときは、大きさの手がかりが弱いことを伝える', () => {
    const s = qualityReason(['ar_scale']);
    expect(s).toContain('大きさの手がかりが弱い');
    expect(qualityReason(['overlap'])).not.toContain('AR');
  });
});

describe('印刷マーカーで縮尺を合わせた部屋の誤差の見込み', () => {
  it('実測の検証が無ければ、いちばん長い辺の 3%（最低 5cm）を見込み、根拠にマーカーと書く', () => {
    const r = { ...newRoomBase('r', 'r', '', 'viewing'), source: 'photo' as const, outline: [v(0, 0), v(4, 0), v(4, 3), v(0, 3)],
      scale: { factor: 0.95, status: 'marker' as const } };
    const t = roomTolerance(r);
    expect(t.tol).toBeCloseTo(0.12);
    expect(t.basis).toContain('印刷マーカー');
  });
});
