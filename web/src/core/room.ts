// 部屋データの作成と補正。形や縮尺が変わる操作は必ず geometryVersion を上げる（古い判定を有効扱いにしない）。
import type { Fixture, FixtureType, Measurement, Obstacle, Room, Vec2 } from './types';
import { dist, inwardNormal, mul, add, signedArea, sub, norm, dot, v } from './geom';

const stamp = () => new Date().toISOString();

function touch(room: Room, action: string, geometryChanged: boolean): Room {
  return {
    ...room,
    updatedAt: stamp(),
    geometryVersion: room.geometryVersion + (geometryChanged ? 1 : 0),
    history: [...room.history.slice(-49), { at: stamp(), action }],
  };
}

export function newRoomBase(id: string, name: string, purpose: string, kind: Room['kind']): Room {
  return {
    id, name, purpose, kind, source: 'manual', createdAt: stamp(), updatedAt: stamp(),
    outline: [], edges: [], ceiling: { height: null, status: 'unobserved' }, obstacles: [], fixtures: [], measurements: [],
    scale: { factor: 1, status: 'manual' }, geometryVersion: 0, hasPointCloud: false, history: [],
  };
}

/** 寸法入力による長方形の部屋（撮影できないときの補助）。 */
export function setManualRect(room: Room, w: number, d: number, h: number | null): Room {
  const outline = [v(0, 0), v(w, 0), v(w, d), v(0, d)];
  return touch({
    ...room, source: 'manual', outline, edges: outline.map(() => ({ status: 'manual' as const })),
    ceiling: { height: h, status: h == null ? 'unobserved' : 'manual' }, scale: { factor: 1, status: 'manual' },
  }, '寸法を入力して部屋を作成', true);
}

export interface JobRoomLike {
  outline: [number, number][];
  edge_status: ('observed' | 'unobserved')[];
  ceiling_height: number | null;
  ceiling_observed: boolean;
  obstacles: { kind: string; rect: [number, number, number, number]; yMin: number; yMax: number }[];
  meta: { model: string; n_views: number };
  views?: { n: number; w: number; h: number }[];
  capture_quality?: { level: 'good' | 'poor'; reasons: string[]; disagreement_mm: number | null; overlap: number; n_photos: number; chunks?: number };
  surfaces?: { walls: { edge: number; file: string; coverage: number }[]; floor: { file: string; coverage: number; bbox: [number, number, number, number] } | null };
  /** 写真に写った印刷マーカーから出した、部屋に掛ける倍率（サーバーの roomfit.marker_scale） */
  marker_scale?: { factor: number | null; n: number; spread_pct?: number; size_m: number };
}

/** マーカーの倍率を当てる条件: 2 か所以上（別の写真か別のマーカー）で測れて、ばらつきが小さいこと。1 か所だけだと外れに気づけない。 */
export const MARKER_MIN_N = 2;
export const MARKER_MAX_SPREAD_PCT = 6;

/** 写真からの復元結果を取り込む。人が確認・入力した設備や測定は消さない。
 * 写真に印刷マーカーが写っていて、条件（MARKER_MIN_N・MARKER_MAX_SPREAD_PCT）を満たせば、その倍率で部屋の大きさを合わせる。 */
export function applyJobResult(room: Room, jobId: string, jr: JobRoomLike, newId: () => string): Room {
  const base = applyJobGeometry(room, jobId, jr, newId);
  const ms = jr.marker_scale;
  if (!ms || ms.factor == null || ms.n < MARKER_MIN_N || (ms.spread_pct ?? 0) > MARKER_MAX_SPREAD_PCT) return base;
  const scaled = applyScale(base, ms.factor, `写真に写した印刷マーカー（一辺 ${Math.round(ms.size_m * 1000) / 10}cm、${ms.n} か所）で縮尺を合わせた`);
  return { ...scaled, scale: { ...scaled.scale, status: 'marker' } };
}

function applyJobGeometry(room: Room, jobId: string, jr: JobRoomLike, newId: () => string): Room {
  const outline = jr.outline.map(([x, z]) => v(x, z));
  if (signedArea(outline) < 0) throw new Error('復元結果の輪郭の向きが想定と違います。');
  const obstacles: Obstacle[] = jr.obstacles.map((o) => ({
    id: newId(), kind: 'unknown_vertical', rect: o.rect, yMin: o.yMin, yMax: o.yMax, confirm: 'auto',
  }));
  return touch({
    ...room, source: 'photo', outline, edges: jr.edge_status.map((s) => ({ status: s })),
    ceiling: { height: jr.ceiling_height, status: jr.ceiling_observed ? 'observed' : 'unobserved' },
    obstacles: [...room.obstacles.filter((o) => o.confirm !== 'auto'), ...obstacles],
    // 輪郭が変わると辺の番号が変わるため、辺に付けた設備と測定は引き継げない
    fixtures: [], measurements: [],
    scale: { factor: 1, status: 'model_estimate' }, hasPointCloud: true,
    job: { id: jobId, model: jr.meta.model, nPhotos: jr.meta.n_views },
    views: (jr.views ?? []).map((v) => ({ n: v.n, w: v.w, h: v.h })),
    surfaces: jr.surfaces && (jr.surfaces.walls.length || jr.surfaces.floor) ? {
      walls: jr.surfaces.walls.map((w) => ({ edge: w.edge, file: w.file })),
      ...(jr.surfaces.floor ? { floor: { file: jr.surfaces.floor.file, bbox: jr.surfaces.floor.bbox } } : {}),
      refOutline: outline,
    } : undefined,
    captureQuality: jr.capture_quality && { level: jr.capture_quality.level, reasons: jr.capture_quality.reasons, disagreementMm: jr.capture_quality.disagreement_mm, overlap: jr.capture_quality.overlap, nPhotos: jr.capture_quality.n_photos, ...(jr.capture_quality.chunks ? { chunks: jr.capture_quality.chunks } : {}) },
  }, `写真 ${jr.meta.n_views} 枚から復元`, true);
}

/** 測定の 2 点の、いまの部屋データでの長さ。高さがある測定は 3D の距離。 */
export function measurementLength(m: Pick<Measurement, 'a' | 'b' | 'ya' | 'yb'>): number {
  const flat = dist(m.a, m.b);
  return m.ya != null && m.yb != null ? Math.hypot(flat, m.yb - m.ya) : flat;
}

/** 壁・床の写真の画像を、今の部屋の形に貼ってよいか（形を手で直していないか）。 */
export function surfacesUsable(room: Room): boolean {
  const s = room.surfaces;
  if (!s || s.refOutline.length !== room.outline.length) return false;
  return s.refOutline.every((p, i) => dist(p, room.outline[i]) < 0.01);
}

export function edgeLength(room: Room, i: number): number {
  return dist(room.outline[i], room.outline[(i + 1) % room.outline.length]);
}

export function floorArea(room: Room): number {
  return Math.abs(signedArea(room.outline));
}

/**
 * 実測 1 か所で縮尺を合わせる（原点まわりの一様な拡大縮小）。
 * 直るのは全体の大きさだけ。部屋のゆがみや一部の位置ずれは直らない。
 * 人が入力した大きさ（設備の幅・高さ、手入力の天井高・障害物の寸法）は実寸なので変えない。
 */
export function applyScale(room: Room, s: number, reason: string): Room {
  if (!(s > 0.2 && s < 5)) throw new Error('縮尺の補正量が大きすぎます。測った場所と長さを確認してください。');
  const sp = (p: Vec2) => v(p.x * s, p.z * s);
  const obstacles = room.obstacles.map((o): Obstacle => {
    if (o.confirm === 'auto') return { ...o, rect: o.rect.map((t) => t * s) as Obstacle['rect'], yMin: o.yMin * s, yMax: o.yMax * s };
    const cx = ((o.rect[0] + o.rect[2]) / 2) * s, cz = ((o.rect[1] + o.rect[3]) / 2) * s;
    const hw = (o.rect[2] - o.rect[0]) / 2, hd = (o.rect[3] - o.rect[1]) / 2;
    return { ...o, rect: [cx - hw, cz - hd, cx + hw, cz + hd] };
  });
  return touch({
    ...room,
    outline: room.outline.map(sp),
    obstacles,
    fixtures: room.fixtures.map((f) => ({ ...f, offset: f.offset * s })),
    measurements: room.measurements.map((m) => ({ ...m, a: sp(m.a), b: sp(m.b), ya: m.ya == null ? undefined : m.ya * s, yb: m.yb == null ? undefined : m.yb * s })),
    ceiling: room.ceiling.height != null && room.ceiling.status === 'observed'
      ? { ...room.ceiling, height: room.ceiling.height * s } : room.ceiling,
    scale: { factor: room.scale.factor * s, status: 'calibrated' },
    surfaces: room.surfaces && {
      ...room.surfaces,
      refOutline: room.surfaces.refOutline.map(sp),
      ...(room.surfaces.floor ? { floor: { ...room.surfaces.floor, bbox: room.surfaces.floor.bbox.map((t) => t * s) as [number, number, number, number] } } : {}),
    },
  }, reason, true);
}

export function addMeasurement(room: Room, m: Omit<Measurement, 'at'>): Room {
  const withM = touch({ ...room, measurements: [...room.measurements, { ...m, at: stamp() }] }, `測定「${m.label}」を記録（${m.role === 'calibration' ? '縮尺合わせ' : '検証'}）`, m.role === 'validation');
  if (m.role !== 'calibration') return withM;
  const est = measurementLength(m);
  if (est < 0.05) throw new Error('2 点が近すぎます。離れた 2 点を選んでください。');
  // 以前の縮尺合わせの記録は検証には数えられないので、役割はそのまま残す（評価では calibration を除外する）
  return applyScale(withM, m.measuredM / est, `「${m.label}」の実測 ${Math.round(m.measuredM * 1000) / 10}cm で縮尺を合わせた`);
}

/**
 * 天井高が分かっている場合（物件情報に書いてある、測った）に、それで部屋全体の大きさを合わせる。
 * 写真から天井が復元できている部屋でだけ使える。
 */
export function calibrateByCeiling(room: Room, realHeightM: number): Room {
  if (room.ceiling.height == null || room.ceiling.status !== 'observed') throw new Error('この部屋は、写真から天井の高さを復元できていません。');
  const scaled = applyScale(room, realHeightM / room.ceiling.height, `天井高 ${Math.round(realHeightM * 1000) / 10}cm で縮尺を合わせた`);
  return { ...scaled, ceiling: { height: realHeightM, status: 'manual' } };
}

export function removeMeasurement(room: Room, id: string): Room {
  return touch({ ...room, measurements: room.measurements.filter((m) => m.id !== id) }, '測定を削除', true);
}

export interface AccuracyRow { id: string; label: string; role: Measurement['role']; measuredM: number; estimatedM: number; errorM: number }
export function accuracyTable(room: Room): { rows: AccuracyRow[]; nValidation: number; meanAbs: number | null; maxAbs: number | null } {
  const rows = room.measurements.map((m) => {
    const est = measurementLength(m);
    return { id: m.id, label: m.label, role: m.role, measuredM: m.measuredM, estimatedM: est, errorM: est - m.measuredM };
  });
  const val = rows.filter((r) => r.role === 'validation').map((r) => Math.abs(r.errorM));
  return {
    rows, nValidation: val.length,
    meanAbs: val.length ? val.reduce((a, b) => a + b, 0) / val.length : null,
    maxAbs: val.length ? Math.max(...val) : null,
  };
}

/** 壁 1 枚を、室内側(+)・外側(-)へ平行に動かす。直交する隣の辺の長さが変わる。 */
export function moveEdge(room: Room, i: number, inwardBy: number): Room {
  const n = room.outline.length;
  const a = room.outline[i], b = room.outline[(i + 1) % n];
  const shift = mul(inwardNormal(a, b), inwardBy);
  const outline = room.outline.map((p, k) => (k === i || k === (i + 1) % n ? add(p, shift) : p));
  if (signedArea(outline) <= 0) throw new Error('その量を動かすと部屋の形が壊れます。');
  const edges = room.edges.map((e, k) => (k === i ? { status: 'confirmed' as const } : e));
  return touch({ ...room, outline, edges }, `壁 ${i + 1} を ${Math.round(inwardBy * 100)}cm 動かした`, true);
}

export function setEdgeStatus(room: Room, i: number, status: Room['edges'][number]['status']): Room {
  return touch({ ...room, edges: room.edges.map((e, k) => (k === i ? { status } : e)) }, `壁 ${i + 1} を「${status}」に変更`, true);
}

export function setCeiling(room: Room, height: number | null): Room {
  return touch({ ...room, ceiling: { height, status: height == null ? 'unobserved' : 'manual' } }, '天井高を入力', true);
}

export function upsertObstacle(room: Room, ob: Obstacle): Room {
  const exists = room.obstacles.some((o) => o.id === ob.id);
  return touch({ ...room, obstacles: exists ? room.obstacles.map((o) => (o.id === ob.id ? ob : o)) : [...room.obstacles, ob] }, exists ? '障害物を編集' : '障害物を追加', true);
}
export function removeObstacle(room: Room, id: string): Room {
  return touch({ ...room, obstacles: room.obstacles.filter((o) => o.id !== id) }, '障害物を削除', true);
}

export const FIXTURE_DEFAULTS: Record<FixtureType, { label: string; width: number; height: number; bottom: number }> = {
  door: { label: '扉', width: 0.8, height: 2.0, bottom: 0 },
  window: { label: '窓', width: 1.6, height: 1.1, bottom: 0.9 },
  outlet: { label: 'コンセント', width: 0.07, height: 0.12, bottom: 0.25 },
  lan: { label: 'LAN端子', width: 0.07, height: 0.12, bottom: 0.25 },
  tv: { label: 'テレビ端子', width: 0.07, height: 0.12, bottom: 0.25 },
  aircon: { label: 'エアコン', width: 0.8, height: 0.3, bottom: 2.0 },
};

export function newFixture(id: string, type: FixtureType, edgeIndex: number, offset: number, hasEntrance: boolean): Fixture {
  const d = FIXTURE_DEFAULTS[type];
  return {
    id, type, edgeIndex, offset, width: d.width, height: d.height, bottom: d.bottom, source: 'manual', confirmed: true,
    door: type === 'door' ? { hinge: 'start', opensInto: 'room', maxAngleDeg: 90, isEntrance: !hasEntrance } : undefined,
  };
}

export function upsertFixture(room: Room, fx: Fixture): Room {
  const len = edgeLength(room, fx.edgeIndex);
  const clamped = { ...fx, offset: Math.max(fx.width / 2, Math.min(len - fx.width / 2, fx.offset)) };
  let fixtures = room.fixtures.some((f) => f.id === fx.id) ? room.fixtures.map((f) => (f.id === fx.id ? clamped : f)) : [...room.fixtures, clamped];
  if (clamped.door?.isEntrance) fixtures = fixtures.map((f) => (f.id !== clamped.id && f.door ? { ...f, door: { ...f.door, isEntrance: false } } : f));
  return touch({ ...room, fixtures }, `${FIXTURE_DEFAULTS[fx.type].label}を登録・編集`, true);
}
export function removeFixture(room: Room, id: string): Room {
  return touch({ ...room, fixtures: room.fixtures.filter((f) => f.id !== id) }, '設備を削除', true);
}

/** 点 p に最も近い壁と、その壁に沿った位置を返す（設備を壁へ置くときに使う）。 */
export function nearestEdge(room: Room, p: Vec2): { edge: number; offset: number; distance: number; point: Vec2 } {
  let best = { edge: 0, offset: 0, distance: Infinity, point: p };
  for (let i = 0; i < room.outline.length; i++) {
    const a = room.outline[i], b = room.outline[(i + 1) % room.outline.length];
    const d = norm(sub(b, a)), L = dist(a, b);
    const t = Math.max(0, Math.min(L, dot(sub(p, a), d)));
    const q = add(a, mul(d, t));
    const dd = dist(p, q);
    if (dd < best.distance) best = { edge: i, offset: t, distance: dd, point: q };
  }
  return best;
}
