// 配置の判定。結果は保証ではなく、入力の範囲で分かったことの報告。
// 状態: problem=根拠のある干渉・不足 / needs_check=誤差・欠損・未入力で判断できない / clear は一覧に出さない。
import type { Room, Furniture, Layout, Placement, Issue, IssueStatus, JudgeResult, JudgeSettings, Vec2, Fixture } from './types';
import {
  add, mul, sub, norm, dist, v, footprint, frontZone, frontNormal, rectPoly, convexRelation, convexVsOutline,
  inwardNormal, intervalsOverlap, segIntersect, pointInPolygon, polyCentroid, bbox,
} from './geom';
import { fmtCm } from './units';
import { buildPassageGrid, widestPath } from './passage';
import { measurementLength } from './room';

export const DEFAULT_SETTINGS: JudgeSettings = { passageWidthM: 0.6, outletAccessM: 0.15 };

/** 縮尺を実測で確かめていない部屋で見込む、長さに対する誤差の割合。 */
export const UNCALIBRATED_SCALE_ERROR = 0.06;
/** 復元の質が「要注意」の部屋で見込む割合。実測との比較で、そうした復元は 8〜15% ずれ、1 か所合わせても形が合わなかった。 */
export const POOR_UNCALIBRATED_ERROR = 0.15;
export const POOR_CALIBRATED_ERROR = 0.08;
/** 要注意のうち、写真ごとの見え方の推定がばらついた復元（理由に focal を含む）。実測との比較で、1 か所合わせてもほかが 18% 合わなかった。 */
export const SEVERE_ERROR = 0.2;

/** 人の体が通る高さ。これより上にある物（吊り棚・梁）は通路をふさがないとみなす。 */
const BODY_HEIGHT = 1.9;

/**
 * 部屋形状にどれだけの誤差を見込むか。
 * 推定された「信頼度」からは作らない。実測との比較（検証用の測定）だけを根拠にする。
 */
export function roomTolerance(room: Room): { tol: number; basis: string } {
  if (room.source === 'manual') return { tol: 0.01, basis: '寸法を手入力した部屋のため、1cm の誤差を見込んでいます。' };
  const errs = room.measurements
    .filter((m) => m.role === 'validation')
    .map((m) => Math.abs(measurementLength(m) - m.measuredM));
  if (room.scale.status === 'calibrated' && errs.length >= 1) {
    const worst = Math.max(...errs);
    const tol = Math.max(0.02, worst);
    return {
      tol,
      basis: `縮尺合わせに使っていない ${errs.length} か所の実測との差（最大 ${fmtCm(worst)}）を誤差として見込んでいます。${errs.length < 5 ? '検証は5か所以上を推奨します。' : ''}`,
    };
  }
  const poor = room.captureQuality?.level === 'poor';
  const severe = poor && room.captureQuality!.reasons.includes('focal');
  // 範囲に分けて復元してつないだ部屋: 公開データ（docs/検証記録.md §14）で、つなぎ目をまたぐ 3m 以上の長さが 8 割の組で −6〜+13% だった。
  // 質の判定が良好でも、要注意と同じ幅の誤差を見込む。
  const stitched = !poor && (room.captureQuality?.chunks ?? 0) > 1;
  const stitchedBasis = (t: number) => `この部屋は、写真を ${room.captureQuality!.chunks} つの範囲に分けて復元し、つないだものです。つなぎ目をまたぐ寸法は、実測のある公開データで 1 割前後ずれることがありました。${fmtCm(t)} の誤差を見込んでいます。`;
  const bb = bbox(room.outline);
  const maxDim = Math.max(bb.x1 - bb.x0, bb.z1 - bb.z0);
  if (room.scale.status === 'calibrated') {
    if (poor) {
      const t = Math.max(0.05, (severe ? SEVERE_ERROR : POOR_CALIBRATED_ERROR) * maxDim);
      return { tol: t, basis: `縮尺は1か所で合わせましたが、この復元は写真どうしの食い違いが大きく、形そのものがゆがんでいる可能性があります。${fmtCm(t)} の誤差を見込んでいます。撮り直しをおすすめします。` };
    }
    if (stitched) { const t = Math.max(0.05, POOR_CALIBRATED_ERROR * maxDim); return { tol: t, basis: stitchedBasis(t) }; }
    return { tol: 0.05, basis: '縮尺は1か所で合わせましたが、別の場所での検証がまだありません。仮に 5cm の誤差を見込んでいます。' };
  }
  // 推定の縮尺のずれは長さに比例する。実測との比較（2026-10-12・1 部屋・4 か所）では 4〜7% 大きく出た。
  // 例数が少ないので補正はせず、部屋のいちばん長い辺の 6% を誤差として見込む（docs/検証記録.md §8）。
  if (poor) {
    const t = Math.max(0.05, (severe ? SEVERE_ERROR : POOR_UNCALIBRATED_ERROR) * maxDim);
    return { tol: t, basis: `この復元は写真どうしの食い違いが大きく、信頼しにくい状態です。${fmtCm(t)} の誤差を見込んで判定していますが、撮り直しをおすすめします（測りたい物の全体が入る距離から、2 か所の立ち位置で）。` };
  }
  if (stitched) { const t = Math.max(0.05, POOR_UNCALIBRATED_ERROR * maxDim); return { tol: t, basis: stitchedBasis(t) }; }
  const tol = Math.max(0.05, UNCALIBRATED_SCALE_ERROR * maxDim);
  return { tol, basis: `部屋の大きさは写真からの推定です。推定は数 % ずれることがあるため、この部屋では ${fmtCm(tol)} の誤差を見込んで判定しています。寸法の分かる物かメジャーで 1 か所確かめると、見込む誤差を小さくできます。` };
}

function byTolerance(clearance: number, tol: number): IssueStatus {
  if (clearance < -tol) return 'problem';
  if (clearance <= tol) return 'needs_check';
  return 'clear';
}

function edgePoints(room: Room, i: number): [Vec2, Vec2] {
  return [room.outline[i], room.outline[(i + 1) % room.outline.length]];
}

export function fixturePosition(room: Room, fx: Fixture): { center: Vec2; dir: Vec2; inward: Vec2; a: Vec2; b: Vec2 } {
  const [p, q] = edgePoints(room, fx.edgeIndex);
  const dir = norm(sub(q, p));
  const center = add(p, mul(dir, fx.offset));
  return { center, dir, inward: inwardNormal(p, q), a: add(center, mul(dir, -fx.width / 2)), b: add(center, mul(dir, fx.width / 2)) };
}

function zoneInFront(room: Room, fx: Fixture, depth: number, sidePad = 0): Vec2[] {
  const { a, b, dir, inward } = fixturePosition(room, fx);
  const a2 = add(a, mul(dir, -sidePad)), b2 = add(b, mul(dir, sidePad));
  return [a2, b2, add(b2, mul(inward, depth)), add(a2, mul(inward, depth))];
}

const FIXTURE_NAME: Record<Fixture['type'], string> = {
  outlet: 'コンセント', door: '扉', window: '窓', lan: 'LAN端子', tv: 'テレビ端子', aircon: 'エアコン',
};

interface Item { p: Placement; f: Furniture; poly: Vec2[]; y0: number; y1: number; unverified: string[] }

export function judgeLayout(room: Room, furniture: Furniture[], layout: Layout, settings: JudgeSettings = DEFAULT_SETTINGS): JudgeResult {
  const { tol, basis } = roomTolerance(room);
  const issues: Issue[] = [];
  let n = 0;
  const push = (i: Omit<Issue, 'id'>) => { if (i.status !== 'clear') issues.push({ ...i, id: `i${n++}` }); };
  const fmap = new Map(furniture.map((f) => [f.id, f]));

  const items: Item[] = [];
  for (const p of layout.items) {
    const f = fmap.get(p.furnitureId);
    if (!f) continue;
    const unverified: string[] = [];
    if (!f.dimConfirmed) unverified.push(`「${f.name}」の寸法が未確認です（根拠: ${dimSourceLabel(f.dimSource)}）。`);
    items.push({ p, f, poly: footprint(p, f), y0: p.elevation, y1: p.elevation + f.h, unverified });
  }

  if (room.scale.status === 'model_estimate')
    push({
      check: 'input', status: 'needs_check', targets: [], title: '部屋の大きさは写真からの推定です',
      reason: 'このまま配置を試せます。余裕のある配置はそのまま判断でき、ぎりぎりの配置は「要確認」として知らせます。数 cm が効く場所だけ、寸法の分かる物（登録済みの家具、扉の幅など）かメジャーで 1 か所確かめてください。',
      dims: [], unverified: ['部屋全体の大きさ（推定）'], suggestion: '確かめる場合は「部屋の確認」→「縮尺と精度」（任意）',
    });
  if (room.captureQuality?.level === 'poor')
    push({
      check: 'input', status: 'needs_check', targets: [], title: 'この部屋の復元は信頼しにくい状態です',
      reason: qualityReason(room.captureQuality.reasons),
      dims: [], unverified: ['部屋の形と大きさ'], suggestion: '測りたい物の全体が入る距離（1.5〜2m）から、2 か所の立ち位置で 4 枚以上撮り直す',
    });
  const nUnobs = room.edges.filter((e) => e.status === 'unobserved').length;
  if (nUnobs > 0)
    push({
      check: 'input', status: 'needs_check', targets: [], title: `形の分からない壁が ${nUnobs} か所あります`,
      reason: '撮影されていない、または復元できなかった範囲です。そこに柱や出っ張りが無いとは言えません。',
      dims: [], unverified: ['未確認の壁の形'], suggestion: '追加で撮影するか、「部屋の確認」で形を確認・修正する',
    });

  for (const it of items) {
    // ---- 壁・部屋の外
    const rel = convexVsOutline(it.poly, room.outline);
    const edgeStatus = rel.edge >= 0 ? room.edges[rel.edge]?.status : undefined;
    const clearance = rel.outside > 0 ? -rel.outside : rel.gap;
    let st = byTolerance(clearance, tol);
    const unv = [...it.unverified];
    if (st !== 'clear') {
      const [ea, eb] = edgePoints(room, Math.max(0, rel.edge));
      const inward = inwardNormal(ea, eb);
      const moveBy = rel.outside + tol + 0.005;
      if (edgeStatus === 'unobserved') {
        if (st === 'problem') st = 'needs_check';
        unv.push('この壁は未確認のため、残りの隙間を確定できません。');
      } else if (edgeStatus === 'observed') unv.push('この壁は写真からの自動復元で、人の確認がまだです。');
      if (st === 'problem' && it.unverified.length) st = 'needs_check';
      push({
        check: 'wall', status: st, targets: [it.p.id],
        title: rel.outside > 0 ? `「${it.f.name}」が壁に重なっています` : `「${it.f.name}」と壁の隙間が誤差の範囲内です`,
        reason: rel.outside > 0
          ? `壁から約 ${fmtCm(rel.outside)} はみ出しています（見込んでいる誤差 ${fmtCm(tol)}）。`
          : `壁までの隙間は約 ${fmtCm(rel.gap)} で、見込んでいる誤差 ${fmtCm(tol)} 以内です。実際には当たる可能性があります。`,
        dims: [{ label: rel.outside > 0 ? 'はみ出し' : '壁との隙間', valueM: rel.outside > 0 ? rel.outside : rel.gap }, { label: '見込み誤差', valueM: tol }],
        unverified: unv,
        suggestion: rel.outside > 0 ? `壁から約 ${fmtCm(moveBy)} 離すと解消する可能性があります。` : '現地で実寸を測って確認してください。',
        fix: rel.outside > 0 ? { placementId: it.p.id, dx: inward.x * moveBy, dz: inward.z * moveBy } : undefined,
        highlight: { polygon: it.poly },
      });
    }

    // ---- 柱・梁・出っ張り
    for (const ob of room.obstacles) {
      if (intervalsOverlap(it.y0, it.y1, ob.yMin, ob.yMax) <= 0) continue;
      const r = convexRelation(it.poly, rectPoly(ob.rect));
      let s = byTolerance(r.overlap ? -r.depth : r.gap, tol);
      if (s === 'clear') continue;
      const u = [...it.unverified];
      if (ob.confirm === 'auto') { u.push('この障害物は自動復元で、種類と大きさの確認がまだです。'); if (s === 'problem') s = 'needs_check'; }
      if (s === 'problem' && it.unverified.length) s = 'needs_check';
      const name = ob.label ?? ({ pillar: '柱', beam: '梁', protrusion: '出っ張り', unknown_vertical: '縦の構造物（種類未確認）' } as const)[ob.kind];
      const moveBy = r.depth + tol + 0.005;
      push({
        check: 'obstacle', status: s, targets: [it.p.id, ob.id],
        title: r.overlap ? `「${it.f.name}」が${name}に重なっています` : `「${it.f.name}」と${name}の隙間が誤差の範囲内です`,
        reason: r.overlap ? `約 ${fmtCm(r.depth)} 重なっています。` : `隙間は約 ${fmtCm(r.gap)} で、見込んでいる誤差 ${fmtCm(tol)} 以内です。`,
        dims: [{ label: r.overlap ? '重なり' : '隙間', valueM: r.overlap ? r.depth : r.gap }, { label: '見込み誤差', valueM: tol }],
        unverified: u,
        suggestion: r.overlap ? `約 ${fmtCm(moveBy)} ずらすと解消する可能性があります。` : undefined,
        fix: r.overlap ? { placementId: it.p.id, dx: r.push.x * moveBy, dz: r.push.z * moveBy } : undefined,
        highlight: { polygon: rectPoly(ob.rect) },
      });
    }

    // ---- 天井
    if (room.ceiling.height != null) {
      const over = it.y1 - room.ceiling.height;
      let s = byTolerance(-over, tol);
      if (s !== 'clear') {
        if (s === 'problem' && (room.ceiling.status === 'observed' || it.unverified.length)) s = 'needs_check';
        push({
          check: 'ceiling', status: s, targets: [it.p.id],
          title: over > 0 ? `「${it.f.name}」が天井より高くなっています` : `「${it.f.name}」と天井の隙間が誤差の範囲内です`,
          reason: `家具の上端 ${fmtCm(it.y1)}、天井高 ${fmtCm(room.ceiling.height)}。`,
          dims: [{ label: '家具の上端', valueM: it.y1 }, { label: '天井高', valueM: room.ceiling.height }],
          unverified: [...it.unverified, ...(room.ceiling.status === 'observed' ? ['天井高は自動復元の値で、未確認です。'] : [])],
          highlight: { polygon: it.poly },
        });
      }
    } else if (it.y1 > 1.8) {
      push({
        check: 'ceiling', status: 'needs_check', targets: [it.p.id], title: `「${it.f.name}」が天井に収まるか確認できません`,
        reason: `家具の上端は ${fmtCm(it.y1)} ですが、天井高が分かっていません。`,
        dims: [{ label: '家具の上端', valueM: it.y1 }], unverified: ['天井高'], suggestion: '「部屋の確認」で天井高を入力する', highlight: { polygon: it.poly },
      });
    }

    // ---- 使用空間（引き出し・開き戸・椅子など）
    if (it.f.useSpace.kind === 'unknown') {
      push({
        check: 'use_space', status: 'needs_check', targets: [it.p.id], title: `「${it.f.name}」を使うための空間が未入力です`,
        reason: '引き出しや扉を開けるのに必要な奥行きが登録されていないため、使えるかどうかは判定していません。',
        dims: [], unverified: ['開閉・使用に必要な奥行き'], suggestion: '家具の編集で「使うときに必要な空間」を入力する', highlight: { polygon: it.poly },
      });
    } else if (it.f.useSpace.kind !== 'none' && it.f.useSpace.depth > 0) {
      const zone = frontZone(it.p, it.f, it.f.useSpace.depth);
      const kindName = { drawer: '引き出しを引く', swing_door: '扉を開く', chair: '椅子を引く', access: '前に立つ' }[it.f.useSpace.kind];
      let worst = 0, who = '', whoId = '';
      const zr = convexVsOutline(zone, room.outline);
      let unobs = false;
      if (zr.outside > worst) { worst = zr.outside; who = '壁'; whoId = ''; unobs = room.edges[zr.edge]?.status === 'unobserved'; }
      for (const ob of room.obstacles) {
        if (intervalsOverlap(it.y0, it.y1, ob.yMin, ob.yMax) <= 0) continue;
        const r = convexRelation(zone, rectPoly(ob.rect));
        if (r.overlap && r.depth > worst) { worst = r.depth; who = ob.label ?? '柱・出っ張り'; whoId = ob.id; unobs = false; }
      }
      for (const other of items) {
        if (other === it || intervalsOverlap(it.y0, it.y1, other.y0, other.y1) <= 0) continue;
        const r = convexRelation(zone, other.poly);
        if (r.overlap && r.depth > worst) { worst = r.depth; who = `「${other.f.name}」`; whoId = other.p.id; unobs = false; }
      }
      if (worst > 0) {
        let s = byTolerance(-worst, tol);
        if (s === 'problem' && (unobs || it.unverified.length)) s = 'needs_check';
        const front = frontNormal(it.p);
        push({
          check: 'use_space', status: s, targets: whoId ? [it.p.id, whoId] : [it.p.id],
          title: `「${it.f.name}」の前の空間が足りません`,
          reason: `${kindName}ために正面に ${fmtCm(it.f.useSpace.depth)} を見込んでいますが、${who}が約 ${fmtCm(worst)} 入り込んでいます。`,
          dims: [{ label: '必要な奥行き', valueM: it.f.useSpace.depth }, { label: '不足', valueM: worst }],
          unverified: [...it.unverified, ...(unobs ? ['この壁は未確認です。'] : [])],
          suggestion: `「${it.f.name}」を正面と反対の向きへ約 ${fmtCm(worst + tol)} 動かすか、向きを変えてください。`,
          fix: { placementId: it.p.id, dx: -front.x * (worst + tol + 0.005), dz: -front.z * (worst + tol + 0.005) },
          highlight: { polygon: zone },
        });
      }
    }
  }

  // ---- 家具どうし
  for (let a = 0; a < items.length; a++)
    for (let b = a + 1; b < items.length; b++) {
      const A = items[a], B = items[b];
      if (intervalsOverlap(A.y0, A.y1, B.y0, B.y1) <= 0) continue;
      const r = convexRelation(A.poly, B.poly);
      if (!r.overlap) continue; // 家具どうしはぴったり付けて置くことがあるため、隙間 0 は問題にしない
      const u = [...A.unverified, ...B.unverified];
      const dimTol = 0.005;
      if (r.depth <= dimTol) continue;
      const moveBy = r.depth + 0.005;
      push({
        check: 'furniture', status: u.length ? 'needs_check' : 'problem', targets: [A.p.id, B.p.id],
        title: `「${A.f.name}」と「${B.f.name}」が重なっています`,
        reason: `約 ${fmtCm(r.depth)} 重なっています。${u.length ? '寸法が未確認の家具を含むため、確定はできません。' : ''}`,
        dims: [{ label: '重なり', valueM: r.depth }], unverified: u,
        suggestion: `どちらかを約 ${fmtCm(moveBy)} ずらすと解消する可能性があります。`,
        fix: { placementId: A.p.id, dx: r.push.x * moveBy, dz: r.push.z * moveBy },
        highlight: { polygon: A.poly },
      });
    }

  // ---- 設備
  const blockers = () => [
    ...items.filter((i) => i.y0 < BODY_HEIGHT).map((i) => i.poly),
    ...room.obstacles.filter((o) => o.yMin < BODY_HEIGHT).map((o) => rectPoly(o.rect)),
  ];
  for (const fx of room.fixtures) {
    const name = FIXTURE_NAME[fx.type];
    const fu = fx.confirmed ? [] : [`この${name}の位置は未確認です。`];
    if (fx.type === 'door') {
      checkDoor(room, fx, items, tol, push);
      continue;
    }
    if (fx.type === 'outlet' || fx.type === 'lan' || fx.type === 'tv') {
      const plug = zoneInFront(room, fx, 0.05, 0.01);
      const access = zoneInFront(room, fx, settings.outletAccessM, 0.03);
      for (const it of items) {
        if (intervalsOverlap(it.y0, it.y1, fx.bottom, fx.bottom + fx.height) <= 0) continue;
        const hit = convexRelation(plug, it.poly).overlap;
        const near = convexRelation(access, it.poly).overlap;
        if (!hit && !near) continue;
        const legNote = it.f.legClearance == null
          ? ['家具は箱として判定しています。脚の下の隙間は考慮していません。']
          : it.f.legClearance >= fx.bottom + fx.height ? null : [];
        if (legNote === null) continue; // 脚の下に収まる高さ
        push({
          check: 'fixture', status: hit ? (fu.length || it.unverified.length ? 'needs_check' : 'problem') : 'needs_check',
          targets: [it.p.id, fx.id],
          title: hit ? `${name}が「${it.f.name}」の後ろに隠れます` : `${name}の前の空間が足りない可能性があります`,
          reason: hit
            ? `${name}（床から ${fmtCm(fx.bottom)}〜${fmtCm(fx.bottom + fx.height)}）の正面を家具がふさいでいます。プラグを挿したまま置けるかは奥行き次第です。`
            : `${name}の前 ${fmtCm(settings.outletAccessM)} 以内に家具があります。抜き差しがしにくい可能性があります。`,
          dims: [{ label: `${name}の高さ（下端）`, valueM: fx.bottom }, { label: '前に空けたい奥行き（設定）', valueM: settings.outletAccessM }],
          unverified: [...fu, ...it.unverified, ...legNote],
          suggestion: '家具をずらすか、延長コードの利用を検討してください。',
          highlight: { polygon: access, point: fixturePosition(room, fx).center },
        });
      }
    }
    if (fx.type === 'window') {
      const zone = zoneInFront(room, fx, 0.1);
      for (const it of items) {
        if (it.y1 <= fx.bottom || !convexRelation(zone, it.poly).overlap) continue;
        push({
          check: 'fixture', status: 'needs_check', targets: [it.p.id, fx.id], title: `窓の前に「${it.f.name}」があります`,
          reason: `家具の上端 ${fmtCm(it.y1)} が窓の下端 ${fmtCm(fx.bottom)} より高く、窓の一部が隠れます。開閉や採光への影響は現地で確認してください。`,
          dims: [{ label: '家具の上端', valueM: it.y1 }, { label: '窓の下端', valueM: fx.bottom }], unverified: [...fu, ...it.unverified],
          highlight: { polygon: zone },
        });
      }
    }
    if (fx.type === 'aircon') {
      const zone = zoneInFront(room, fx, 0.5);
      for (const it of items) {
        if (it.y1 < fx.bottom - 0.3 || !convexRelation(zone, it.poly).overlap) continue;
        push({
          check: 'fixture', status: 'needs_check', targets: [it.p.id, fx.id], title: `エアコンの近くに「${it.f.name}」があります`,
          reason: `エアコンの下端 ${fmtCm(fx.bottom)} に対して家具の上端が ${fmtCm(it.y1)} です。吹き出しや掃除に必要な空間はメーカーの条件によるため、ここでは判定していません。`,
          dims: [{ label: '家具の上端', valueM: it.y1 }, { label: 'エアコンの下端', valueM: fx.bottom }],
          unverified: ['メーカーが指定する必要な空間', ...fu, ...it.unverified], highlight: { polygon: zone },
        });
      }
    }
  }

  // ---- 通路（入口から各家具の正面まで）
  const entrance = room.fixtures.find((f) => f.type === 'door' && f.door?.isEntrance);
  if (items.length > 0) {
    if (!entrance) {
      push({
        check: 'passage', status: 'needs_check', targets: [], title: '通路は判定していません（入口が未登録）',
        reason: '部屋の入口になる扉が登録されていないため、入口から家具までたどり着けるかは調べていません。',
        dims: [], unverified: ['入口の位置'], suggestion: '「部屋の確認」で扉を登録し、入口に指定する',
      });
    } else {
      const W = settings.passageWidthM;
      const grid = buildPassageGrid(room.outline, blockers());
      const ep = fixturePosition(room, entrance);
      const start = add(ep.center, mul(ep.inward, Math.max(0.35, W / 2 + 0.05)));
      for (const it of items) {
        if (it.y0 >= BODY_HEIGHT) continue;
        const front = frontNormal(it.p);
        const frontMid = add(v(it.p.x, it.p.z), mul(front, it.f.d / 2));
        const goal = add(frontMid, mul(front, W / 2 + 0.05));
        const r = widestPath(grid, start, goal);
        const u = [...it.unverified, ...(nUnobs ? ['形の分からない壁があるため、経路の幅は確定ではありません。'] : [])];
        if (!r.reachable) {
          push({
            check: 'passage', status: u.length ? 'needs_check' : 'problem', targets: [it.p.id, entrance.id],
            title: `入口から「${it.f.name}」の正面までたどり着けません`,
            reason: '家具・壁でふさがれていて、連続した経路が見つかりませんでした。正面が壁を向いている場合もこの表示になります。',
            dims: [{ label: '確保したい幅（設定）', valueM: W }], unverified: u,
            suggestion: '家具の向きを変えるか、途中をふさいでいる家具を動かしてください。', highlight: { polygon: it.poly, point: goal },
          });
          continue;
        }
        // 幅の誤差は両側の物の位置誤差が効くので 2 倍を見込む
        let s = byTolerance(r.widthM - W, tol * 2);
        if (s === 'clear') continue;
        if (s === 'problem' && u.length) s = 'needs_check';
        push({
          check: 'passage', status: s, targets: [it.p.id, entrance.id],
          title: `入口から「${it.f.name}」までの通路が狭くなっています`,
          reason: `経路のいちばん狭い所で確保できる幅は約 ${fmtCm(r.widthM)} です（確保したい幅の設定 ${fmtCm(W)}）。幅は、人を円とみなして通れる最大の直径で測っています。`,
          dims: [{ label: '確保できる幅', valueM: r.widthM }, { label: '確保したい幅（設定）', valueM: W }, { label: '見込み誤差', valueM: tol * 2 }],
          unverified: u, suggestion: '印の付いた狭い所の近くにある家具を動かしてください。',
          highlight: { path: r.path, point: r.bottleneck ?? undefined },
        });
      }
    }
  }

  const order: Record<IssueStatus, number> = { problem: 0, needs_check: 1, clear: 2 };
  issues.sort((a, b) => order[a.status] - order[b.status]);
  return { issues, geometryVersion: room.geometryVersion, toleranceM: tol, toleranceBasis: basis, checkedAt: new Date().toISOString() };
}

/** 扉を少しずつ開いていき、途中で家具・柱に当たる角度を調べる。 */
function checkDoor(room: Room, fx: Fixture, items: Item[], tol: number, push: (i: Omit<Issue, 'id'>) => void) {
  const door = fx.door;
  const fu = fx.confirmed ? [] : ['この扉の位置・幅は未確認です。'];
  if (!door) {
    push({
      check: 'door_swing', status: 'needs_check', targets: [fx.id], title: '扉の開き方が未入力です',
      reason: '開く向きが登録されていないため、開閉の干渉は判定していません。', dims: [], unverified: ['扉の開き方'],
    });
    return;
  }
  const pos = fixturePosition(room, fx);
  // 開口部そのものをふさいでいないか（引き戸・外開きでも必要）
  const opening = zoneInFront(room, fx, 0.05);
  for (const it of items) {
    if (it.y0 >= fx.bottom + fx.height) continue;
    if (convexRelation(opening, it.poly).overlap)
      push({
        check: 'door_swing', status: fu.length || it.unverified.length ? 'needs_check' : 'problem', targets: [it.p.id, fx.id],
        title: `扉の前を「${it.f.name}」がふさいでいます`, reason: '扉の開口部の正面に家具があります。',
        dims: [{ label: '扉の幅', valueM: fx.width }], unverified: [...fu, ...it.unverified], highlight: { polygon: opening },
      });
  }
  if (door.opensInto !== 'room') return;
  const hinge = door.hinge === 'start' ? pos.a : pos.b;
  const closedDir = door.hinge === 'start' ? pos.dir : mul(pos.dir, -1);
  const leafAt = (degv: number): Vec2 => {
    const t = (degv * Math.PI) / 180;
    return add(hinge, add(mul(closedDir, Math.cos(t) * fx.width), mul(pos.inward, Math.sin(t) * fx.width)));
  };
  const sector: Vec2[] = [hinge];
  for (let a = 0; a <= door.maxAngleDeg; a += 5) sector.push(leafAt(a));
  const polys: { poly: Vec2[]; id: string; name: string; unv: string[] }[] = [
    ...items.filter((it) => it.y0 < fx.bottom + fx.height).map((it) => ({ poly: it.poly, id: it.p.id, name: `「${it.f.name}」`, unv: it.unverified })),
    ...room.obstacles.filter((o) => o.yMin < fx.bottom + fx.height).map((o) => ({
      poly: rectPoly(o.rect), id: o.id, name: o.label ?? '柱・出っ張り', unv: o.confirm === 'auto' ? ['この障害物は自動復元で未確認です。'] : [],
    })),
  ];
  for (const t of polys) {
    let blockedAt: number | null = null;
    for (let a = 1; a <= door.maxAngleDeg && blockedAt === null; a += 1) {
      const tip = leafAt(a);
      if (pointInPolygon(tip, t.poly) || pointInPolygon(add(hinge, mul(sub(tip, hinge), 0.5)), t.poly)) { blockedAt = a; break; }
      for (let k = 0; k < t.poly.length; k++)
        if (segIntersect(hinge, tip, t.poly[k], t.poly[(k + 1) % t.poly.length])) { blockedAt = a; break; }
    }
    if (blockedAt !== null) {
      const u = [...fu, ...t.unv];
      push({
        check: 'door_swing', status: u.length ? 'needs_check' : 'problem', targets: [t.id, fx.id],
        title: `扉を開く範囲に${t.name}があります`,
        reason: `扉は約 ${blockedAt - 1}° までしか開きません（全開は ${door.maxAngleDeg}°）。開く途中で${t.name}に当たります。`,
        dims: [{ label: '扉の幅', valueM: fx.width }], unverified: u,
        suggestion: `${t.name}を扉の開く範囲（扇形の表示）の外へ動かしてください。`,
        highlight: { polygon: sector, point: polyCentroid(t.poly) },
      });
    } else {
      // 当たらないが、誤差の範囲内まで近い
      let gap = Infinity;
      for (let a = 0; a <= door.maxAngleDeg; a += 2) {
        const tip = leafAt(a);
        for (let k = 0; k < t.poly.length; k++) {
          const p1 = t.poly[k], p2 = t.poly[(k + 1) % t.poly.length];
          gap = Math.min(gap, Math.min(distPointSeg(p1, hinge, tip), distPointSeg(p2, hinge, tip), distPointSeg(tip, p1, p2)));
        }
      }
      if (gap <= tol)
        push({
          check: 'door_swing', status: 'needs_check', targets: [t.id, fx.id], title: `扉を開く範囲と${t.name}の隙間が誤差の範囲内です`,
          reason: `隙間は約 ${fmtCm(gap)} で、見込んでいる誤差 ${fmtCm(tol)} 以内です。`,
          dims: [{ label: '隙間', valueM: gap }, { label: '見込み誤差', valueM: tol }], unverified: [...fu, ...t.unv], highlight: { polygon: sector },
        });
    }
  }
}

function distPointSeg(p: Vec2, a: Vec2, b: Vec2): number {
  const ab = sub(b, a);
  const l2 = ab.x * ab.x + ab.z * ab.z;
  const t = l2 < 1e-18 ? 0 : Math.max(0, Math.min(1, ((p.x - a.x) * ab.x + (p.z - a.z) * ab.z) / l2));
  return dist(p, add(a, mul(ab, t)));
}

/** 復元の質が要注意になった理由を、撮り方の言葉で説明する。 */
export function qualityReason(reasons: string[]): string {
  const parts: string[] = [];
  if (reasons.includes('overlap')) parts.push('写真どうしの重なりが少ない（同じ場所が 2 枚以上に写っていない）');
  if (reasons.includes('disagreement')) parts.push('同じ場所の位置が、写真によって食い違っている');
  if (reasons.includes('stitch')) parts.push('範囲に分けて復元した結果のつなぎ目が、うまく合っていない');
  if (reasons.includes('focal')) parts.push('写真ごとに見え方の推定がばらついている（近すぎる・物の一部しか写っていない写真があると起きやすい）');
  if (reasons.includes('ar_scale')) parts.push('スマホの AR の位置を使えた区間で、カメラがあまり横に動いていない（形はそろっても、大きさの手がかりが弱い。1 か所測ると確実）');
  return `${parts.join('。')}。このような復元は、実測と比べたとき 1 割前後ずれることがありました。`;
}

export function dimSourceLabel(s: Furniture['dimSource']): string {
  return { measured: '自分で測った', product_page: '商品ページの記載', estimated: 'おおよその見積もり', unknown: '不明' }[s];
}

/** 扉の開く範囲（表示用の扇形）。室内へ開く扉のみ。 */
export function doorSector(room: Room, fx: Fixture): Vec2[] | null {
  if (fx.type !== 'door' || !fx.door || fx.door.opensInto !== 'room') return null;
  const pos = fixturePosition(room, fx);
  const hinge = fx.door.hinge === 'start' ? pos.a : pos.b;
  const closedDir = fx.door.hinge === 'start' ? pos.dir : mul(pos.dir, -1);
  const out: Vec2[] = [hinge];
  for (let a = 0; a <= fx.door.maxAngleDeg; a += 5) {
    const t = (a * Math.PI) / 180;
    out.push(add(hinge, add(mul(closedDir, Math.cos(t) * fx.width), mul(pos.inward, Math.sin(t) * fx.width))));
  }
  return out;
}
