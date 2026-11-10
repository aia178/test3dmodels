// 上から見た平面図。x は右、z は下（3D を真上から見たときと同じ向き）。
import { useEffect, useRef } from 'preact/hooks';
import type { Furniture, Issue, Placement, Room, Vec2 } from '../core/types';
import { footprint, frontNormal, pointInPolygon, bbox, add, mul, v, dist, rectPoly, frontZone } from '../core/geom';
import { doorSector, fixturePosition } from '../core/judge';
import { fmtCm } from '../core/units';

export interface PlanProps {
  room: Room;
  cloud?: Float32Array | null; // xyz…（部屋座標, 縮尺適用前）
  items?: { p: Placement; f: Furniture }[];
  selectedId?: string | null;
  selectedEdge?: number | null;
  selectedFixtureId?: string | null;
  selectedObstacleId?: string | null;
  issue?: Issue | null;
  issueTargets?: Map<string, 'problem' | 'needs_check'>;
  pickPoints?: Vec2[];
  showEdgeNumbers?: boolean;
  showLengths?: boolean;
  onSelect?: (placementId: string | null) => void;
  onMove?: (placementId: string, x: number, z: number, done: boolean) => void;
  onTapFloor?: (p: Vec2) => void;
  onTapEdge?: (edge: number, p: Vec2) => void;
  onTapFixture?: (id: string) => void;
  onTapObstacle?: (id: string) => void;
  onMoveObstacle?: (id: string, dx: number, dz: number, done: boolean) => void;
  focusKey?: string;
}

const C = {
  bg: '#fbfaf7', floor: '#f1ece3', wall: '#3d4a45', wallAuto: '#6f7f78', unobs: '#b3541e', grid: '#e7e1d6',
  furn: '#c9d6cf', furnSel: '#2f6f5e', problem: '#b3261e', check: '#a15c00', text: '#2a2f2c', cloud: 'rgba(60,70,66,0.16)',
};

interface View { scale: number; ox: number; oz: number }

export function Plan2D(props: PlanProps) {
  const ref = useRef<HTMLCanvasElement>(null);
  const view = useRef<View | null>(null);
  const propsRef = useRef(props);
  propsRef.current = props;
  const drawRef = useRef<() => void>(() => {});

  useEffect(() => {
    const cv = ref.current!;
    const ctx = cv.getContext('2d')!;
    const fit = () => {
      const r = propsRef.current.room;
      if (!r.outline.length) return;
      const b = bbox(r.outline);
      const w = cv.clientWidth, h = cv.clientHeight;
      const s = Math.min((w - 56) / Math.max(0.5, b.x1 - b.x0), (h - 56) / Math.max(0.5, b.z1 - b.z0));
      view.current = { scale: s, ox: w / 2 - ((b.x0 + b.x1) / 2) * s, oz: h / 2 - ((b.z0 + b.z1) / 2) * s };
    };
    const toScreen = (p: Vec2) => ({ x: p.x * view.current!.scale + view.current!.ox, y: p.z * view.current!.scale + view.current!.oz });
    const toWorld = (x: number, y: number): Vec2 => v((x - view.current!.ox) / view.current!.scale, (y - view.current!.oz) / view.current!.scale);
    const path = (poly: Vec2[], close = true) => {
      ctx.beginPath();
      poly.forEach((p, i) => { const s = toScreen(p); if (i) ctx.lineTo(s.x, s.y); else ctx.moveTo(s.x, s.y); });
      if (close) ctx.closePath();
    };

    const draw = () => {
      const P = propsRef.current;
      const dpr = window.devicePixelRatio || 1;
      const w = cv.clientWidth, h = cv.clientHeight;
      if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.fillStyle = C.bg; ctx.fillRect(0, 0, w, h);
      if (!P.room.outline.length) return;
      if (!view.current) fit();
      const V = view.current!;
      const room = P.room;

      // 床
      path(room.outline); ctx.fillStyle = C.floor; ctx.fill();
      // 50cm の目盛り
      ctx.save(); path(room.outline); ctx.clip();
      ctx.strokeStyle = C.grid; ctx.lineWidth = 1;
      const b = bbox(room.outline);
      ctx.beginPath();
      for (let x = Math.ceil(b.x0 / 0.5) * 0.5; x <= b.x1; x += 0.5) { const a = toScreen(v(x, b.z0)), c = toScreen(v(x, b.z1)); ctx.moveTo(a.x, a.y); ctx.lineTo(c.x, c.y); }
      for (let z = Math.ceil(b.z0 / 0.5) * 0.5; z <= b.z1; z += 0.5) { const a = toScreen(v(b.x0, z)), c = toScreen(v(b.x1, z)); ctx.moveTo(a.x, a.y); ctx.lineTo(c.x, c.y); }
      ctx.stroke(); ctx.restore();

      // 復元した点（床から 30cm〜150cm の高さだけ。壁の位置を見比べるため）
      if (P.cloud) {
        const s = room.scale.factor;
        ctx.fillStyle = C.cloud;
        const n = P.cloud.length / 3, step = Math.max(1, Math.floor(n / 60000));
        for (let i = 0; i < n; i += step) {
          const y = P.cloud[i * 3 + 1] * s;
          if (y < 0.3 || y > 1.5) continue;
          ctx.fillRect(P.cloud[i * 3] * s * V.scale + V.ox, P.cloud[i * 3 + 2] * s * V.scale + V.oz, 1.5, 1.5);
        }
      }

      // 障害物
      for (const ob of room.obstacles) {
        path(rectPoly(ob.rect));
        ctx.fillStyle = ob.yMin > 0.05 ? 'rgba(61,74,69,0.18)' : 'rgba(61,74,69,0.55)'; ctx.fill();
        ctx.lineWidth = P.selectedObstacleId === ob.id ? 3 : 1.5; ctx.strokeStyle = ob.confirm === 'auto' ? C.unobs : C.wall;
        ctx.setLineDash(ob.confirm === 'auto' ? [5, 4] : []); ctx.stroke(); ctx.setLineDash([]);
        const c = toScreen(v((ob.rect[0] + ob.rect[2]) / 2, (ob.rect[1] + ob.rect[3]) / 2));
        ctx.fillStyle = C.text; ctx.font = '11px sans-serif'; ctx.textAlign = 'center';
        ctx.fillText(ob.label ?? ({ pillar: '柱', beam: '梁', protrusion: '出っ張り', unknown_vertical: '？' } as const)[ob.kind], c.x, c.y + 4);
      }

      // 壁。確認済み=太い実線 / 自動復元=細い実線 / 未確認=破線＋「未確認」の文字（色だけに頼らない）
      for (let i = 0; i < room.outline.length; i++) {
        const a = room.outline[i], q = room.outline[(i + 1) % room.outline.length];
        const st = room.edges[i]?.status ?? 'manual';
        const sa = toScreen(a), sb = toScreen(q);
        ctx.beginPath(); ctx.moveTo(sa.x, sa.y); ctx.lineTo(sb.x, sb.y);
        ctx.lineCap = 'round';
        if (st === 'unobserved') { ctx.setLineDash([7, 6]); ctx.strokeStyle = C.unobs; ctx.lineWidth = 2.5; }
        else if (st === 'observed') { ctx.setLineDash([]); ctx.strokeStyle = C.wallAuto; ctx.lineWidth = 3; }
        else { ctx.setLineDash([]); ctx.strokeStyle = C.wall; ctx.lineWidth = 5; }
        if (P.selectedEdge === i) { ctx.lineWidth += 3; ctx.strokeStyle = C.furnSel; }
        ctx.stroke(); ctx.setLineDash([]);
        const mid = toScreen(mul(add(a, q), 0.5));
        const L = dist(a, q);
        const horizontal = Math.abs(sb.x - sa.x) > Math.abs(sb.y - sa.y);
        const pxLen = Math.hypot(sb.x - sa.x, sb.y - sa.y);
        ctx.font = '11px sans-serif'; ctx.textAlign = 'center'; ctx.fillStyle = st === 'unobserved' ? C.unobs : C.text;
        const parts: string[] = [];
        if (P.showEdgeNumbers) parts.push(`壁${i + 1}`);
        if (P.showLengths && pxLen > 44) parts.push(fmtCm(L));
        if (st === 'unobserved' && pxLen > 44) parts.push('未確認');
        if (parts.length) {
          const t = parts.join(' ');
          const tw = ctx.measureText(t).width + 6;
          const tx = mid.x + (horizontal ? 0 : 0), ty = mid.y + (horizontal ? -8 : 4);
          ctx.fillStyle = 'rgba(251,250,247,0.9)'; ctx.fillRect(tx - tw / 2, ty - 11, tw, 14);
          ctx.fillStyle = st === 'unobserved' ? C.unobs : C.text; ctx.fillText(t, tx, ty);
        }
      }

      // 設備
      for (const fx of room.fixtures) {
        const pos = fixturePosition(room, fx);
        const sa = toScreen(pos.a), sb = toScreen(pos.b);
        const sel = P.selectedFixtureId === fx.id;
        if (fx.type === 'door') {
          const sec = doorSector(room, fx);
          if (sec) { path(sec); ctx.fillStyle = 'rgba(47,111,94,0.10)'; ctx.fill(); ctx.strokeStyle = 'rgba(47,111,94,0.6)'; ctx.lineWidth = 1; ctx.setLineDash([3, 3]); ctx.stroke(); ctx.setLineDash([]); }
          ctx.beginPath(); ctx.moveTo(sa.x, sa.y); ctx.lineTo(sb.x, sb.y); ctx.strokeStyle = '#fbfaf7'; ctx.lineWidth = 6; ctx.stroke();
          ctx.strokeStyle = C.furnSel; ctx.lineWidth = sel ? 4 : 2; ctx.stroke();
        } else if (fx.type === 'window') {
          ctx.beginPath(); ctx.moveTo(sa.x, sa.y); ctx.lineTo(sb.x, sb.y); ctx.strokeStyle = '#fbfaf7'; ctx.lineWidth = 6; ctx.stroke();
          ctx.strokeStyle = '#4a7fa8'; ctx.lineWidth = sel ? 4 : 2; ctx.setLineDash([2, 2]); ctx.stroke(); ctx.setLineDash([]);
        }
        const c = toScreen(add(pos.center, mul(pos.inward, fx.type === 'door' || fx.type === 'window' ? 0.12 : 0.08)));
        const glyph = { outlet: '⏚', door: '扉', window: '窓', lan: 'L', tv: 'T', aircon: 'AC' }[fx.type];
        if (fx.type !== 'door' && fx.type !== 'window') {
          ctx.beginPath(); ctx.arc(c.x, c.y, sel ? 10 : 8, 0, Math.PI * 2);
          ctx.fillStyle = '#fff'; ctx.fill(); ctx.strokeStyle = sel ? C.furnSel : C.text; ctx.lineWidth = sel ? 2.5 : 1.2; ctx.stroke();
        }
        ctx.fillStyle = C.text; ctx.font = '10px sans-serif'; ctx.textAlign = 'center';
        ctx.fillText(glyph + (fx.door?.isEntrance ? '(入口)' : ''), c.x, c.y + 3.5);
      }

      // 家具
      for (const it of P.items ?? []) {
        const fp = footprint(it.p, it.f);
        const sel = P.selectedId === it.p.id;
        const flag = P.issueTargets?.get(it.p.id);
        if (sel && it.f.useSpace.depth > 0 && it.f.useSpace.kind !== 'none' && it.f.useSpace.kind !== 'unknown') {
          path(frontZone(it.p, it.f, it.f.useSpace.depth)); ctx.fillStyle = 'rgba(47,111,94,0.10)'; ctx.fill();
          ctx.strokeStyle = 'rgba(47,111,94,0.5)'; ctx.setLineDash([4, 3]); ctx.lineWidth = 1; ctx.stroke(); ctx.setLineDash([]);
        }
        path(fp); ctx.fillStyle = it.f.color + (it.p.elevation > 0.05 ? '77' : 'dd'); ctx.fill();
        ctx.lineWidth = sel ? 3 : 1.5;
        ctx.strokeStyle = sel ? C.furnSel : flag === 'problem' ? C.problem : flag === 'needs_check' ? C.check : '#5c6661';
        if (!it.f.dimConfirmed) ctx.setLineDash([4, 3]);
        ctx.stroke(); ctx.setLineDash([]);
        // 正面の印（三角）
        const fn = frontNormal(it.p);
        const fm = add(v(it.p.x, it.p.z), mul(fn, it.f.d / 2));
        const tip = toScreen(add(fm, mul(fn, -Math.min(0.12, it.f.d / 3))));
        const side = v(-fn.z, fn.x);
        const l = toScreen(add(fm, mul(side, -Math.min(0.1, it.f.w / 4)))), r = toScreen(add(fm, mul(side, Math.min(0.1, it.f.w / 4))));
        ctx.beginPath(); ctx.moveTo(l.x, l.y); ctx.lineTo(r.x, r.y); ctx.lineTo(tip.x, tip.y); ctx.closePath(); ctx.fillStyle = 'rgba(42,47,44,0.55)'; ctx.fill();
        const c = toScreen(v(it.p.x, it.p.z));
        ctx.fillStyle = C.text; ctx.font = '12px sans-serif'; ctx.textAlign = 'center';
        const mark = flag === 'problem' ? '✕ ' : flag === 'needs_check' ? '？ ' : '';
        if (it.f.w * V.scale > 40 || it.f.d * V.scale > 40) ctx.fillText(mark + it.f.name, c.x, c.y + 4);
      }

      // 選んだ問題の場所
      const hl = P.issue?.highlight;
      if (hl) {
        const col = P.issue!.status === 'problem' ? C.problem : C.check;
        if (hl.polygon) { path(hl.polygon); ctx.fillStyle = col + '2e'; ctx.fill(); ctx.strokeStyle = col; ctx.lineWidth = 2.5; ctx.setLineDash([6, 4]); ctx.stroke(); ctx.setLineDash([]); }
        if (hl.path && hl.path.length > 1) { path(hl.path, false); ctx.strokeStyle = col; ctx.lineWidth = 3; ctx.setLineDash([2, 5]); ctx.stroke(); ctx.setLineDash([]); }
        if (hl.point) {
          const s = toScreen(hl.point);
          ctx.beginPath(); ctx.arc(s.x, s.y, 13, 0, Math.PI * 2); ctx.strokeStyle = col; ctx.lineWidth = 3; ctx.stroke();
          ctx.beginPath(); ctx.moveTo(s.x - 6, s.y - 6); ctx.lineTo(s.x + 6, s.y + 6); ctx.moveTo(s.x + 6, s.y - 6); ctx.lineTo(s.x - 6, s.y + 6); ctx.stroke();
        }
      }

      // 測定の線と、選択中の 2 点（1 点目=A, 2 点目=B を区別して表示）
      for (const m of room.measurements) {
        const a = toScreen(m.a), q = toScreen(m.b);
        ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(q.x, q.y);
        ctx.strokeStyle = m.role === 'calibration' ? '#7a4fa3' : '#1f6fb2'; ctx.lineWidth = 1.5; ctx.setLineDash(m.role === 'calibration' ? [] : [6, 3]); ctx.stroke(); ctx.setLineDash([]);
      }
      (P.pickPoints ?? []).forEach((p, i) => {
        const s = toScreen(p);
        ctx.beginPath(); ctx.arc(s.x, s.y, 9, 0, Math.PI * 2); ctx.fillStyle = i === 0 ? '#7a4fa3' : '#1f6fb2'; ctx.fill();
        ctx.fillStyle = '#fff'; ctx.font = 'bold 11px sans-serif'; ctx.textAlign = 'center'; ctx.fillText(i === 0 ? 'A' : 'B', s.x, s.y + 4);
      });
      if ((P.pickPoints ?? []).length === 2) {
        const a = toScreen(P.pickPoints![0]), q = toScreen(P.pickPoints![1]);
        ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(q.x, q.y); ctx.strokeStyle = '#2a2f2c'; ctx.lineWidth = 1.5; ctx.stroke();
      }

      // 縮尺の目安
      const barM = V.scale > 160 ? 0.5 : 1;
      ctx.strokeStyle = C.text; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(14, h - 14); ctx.lineTo(14 + barM * V.scale, h - 14); ctx.stroke();
      ctx.fillStyle = C.text; ctx.font = '11px sans-serif'; ctx.textAlign = 'left'; ctx.fillText(barM === 1 ? '1m' : '50cm', 16, h - 20);
    };
    drawRef.current = draw;

    // ---- 操作
    const pointers = new Map<number, { x: number; y: number }>();
    let drag: null | { kind: 'pan' } | { kind: 'item'; id: string; dx: number; dz: number } | { kind: 'ob'; id: string; last: Vec2 } = null;
    let downAt = { x: 0, y: 0 }, moved = false, pinch = 0;
    const local = (e: PointerEvent) => { const r = cv.getBoundingClientRect(); return { x: e.clientX - r.left, y: e.clientY - r.top }; };
    const hitItem = (wp: Vec2) => {
      const items = propsRef.current.items ?? [];
      for (let i = items.length - 1; i >= 0; i--) if (pointInPolygon(wp, footprint(items[i].p, items[i].f))) return items[i];
      return null;
    };
    const down = (e: PointerEvent) => {
      if (!view.current) return;
      try { cv.setPointerCapture(e.pointerId); } catch { /* 取得できなくても操作は続けられる */ }
      const p = local(e); pointers.set(e.pointerId, p); downAt = p; moved = false;
      if (pointers.size === 2) { const [a, b2] = [...pointers.values()]; pinch = Math.hypot(a.x - b2.x, a.y - b2.y); drag = { kind: 'pan' }; return; }
      const P = propsRef.current, wp = toWorld(p.x, p.y);
      const it = P.onMove ? hitItem(wp) : null;
      if (it) { P.onSelect?.(it.p.id); drag = { kind: 'item', id: it.p.id, dx: it.p.x - wp.x, dz: it.p.z - wp.z }; return; }
      if (P.onMoveObstacle) {
        const ob = P.room.obstacles.find((o) => pointInPolygon(wp, rectPoly(o.rect)));
        if (ob) { P.onTapObstacle?.(ob.id); drag = { kind: 'ob', id: ob.id, last: wp }; return; }
      }
      drag = { kind: 'pan' };
    };
    const move = (e: PointerEvent) => {
      if (!pointers.has(e.pointerId) || !view.current) return;
      const prev = pointers.get(e.pointerId)!, p = local(e);
      pointers.set(e.pointerId, p);
      if (Math.hypot(p.x - downAt.x, p.y - downAt.y) > 5) moved = true;
      const V = view.current;
      if (pointers.size === 2) {
        const [a, b2] = [...pointers.values()];
        const d = Math.hypot(a.x - b2.x, a.y - b2.y), cx = (a.x + b2.x) / 2, cy = (a.y + b2.y) / 2;
        if (pinch > 0) zoomAt(cx, cy, d / pinch);
        pinch = d; draw(); return;
      }
      if (!drag) return;
      if (drag.kind === 'pan') { V.ox += p.x - prev.x; V.oz += p.y - prev.y; }
      else if (drag.kind === 'item' && moved) { const wp = toWorld(p.x, p.y); propsRef.current.onMove?.(drag.id, wp.x + drag.dx, wp.z + drag.dz, false); }
      else if (drag.kind === 'ob' && moved) { const wp = toWorld(p.x, p.y); propsRef.current.onMoveObstacle?.(drag.id, wp.x - drag.last.x, wp.z - drag.last.z, false); drag.last = wp; }
      draw();
    };
    const up = (e: PointerEvent) => {
      if (!pointers.has(e.pointerId) || !view.current) return;
      const p = local(e); pointers.delete(e.pointerId); pinch = 0;
      const P = propsRef.current, wp = toWorld(p.x, p.y);
      if (drag?.kind === 'item' && moved) P.onMove?.(drag.id, wp.x + drag.dx, wp.z + drag.dz, true);
      else if (drag?.kind === 'ob' && moved) P.onMoveObstacle?.(drag.id, 0, 0, true);
      else if (!moved) {
        const fxHit = P.onTapFixture && P.room.fixtures.find((f) => dist(fixturePosition(P.room, f).center, wp) < Math.max(0.18, 14 / view.current!.scale, f.width / 2));
        const obHit = P.onTapObstacle && P.room.obstacles.find((o) => pointInPolygon(wp, rectPoly(o.rect)));
        if (drag?.kind === 'item') { /* 選択済み */ }
        else if (fxHit) P.onTapFixture!(fxHit.id);
        else if (obHit) P.onTapObstacle!(obHit.id);
        else {
          let bestE = -1, bestD = 16 / view.current.scale;
          for (let i = 0; i < P.room.outline.length; i++) {
            const a = P.room.outline[i], q = P.room.outline[(i + 1) % P.room.outline.length];
            const L = dist(a, q); if (L < 1e-6) continue;
            const t = Math.max(0, Math.min(1, ((wp.x - a.x) * (q.x - a.x) + (wp.z - a.z) * (q.z - a.z)) / (L * L)));
            const d = dist(wp, v(a.x + (q.x - a.x) * t, a.z + (q.z - a.z) * t));
            if (d < bestD) { bestD = d; bestE = i; }
          }
          if (bestE >= 0 && P.onTapEdge) P.onTapEdge(bestE, wp);
          else { P.onSelect?.(null); P.onTapFloor?.(wp); }
        }
      }
      if (pointers.size === 0) drag = null;
      draw();
    };
    const zoomAt = (x: number, y: number, k: number) => {
      const V = view.current!;
      const ns = Math.max(20, Math.min(900, V.scale * k));
      V.ox = x - ((x - V.ox) * ns) / V.scale; V.oz = y - ((y - V.oz) * ns) / V.scale; V.scale = ns;
    };
    const wheel = (e: WheelEvent) => { if (!view.current) return; e.preventDefault(); const r = cv.getBoundingClientRect(); zoomAt(e.clientX - r.left, e.clientY - r.top, e.deltaY < 0 ? 1.12 : 1 / 1.12); draw(); };
    cv.addEventListener('pointerdown', down); cv.addEventListener('pointermove', move);
    cv.addEventListener('pointerup', up); cv.addEventListener('pointercancel', up);
    cv.addEventListener('wheel', wheel, { passive: false });
    const ro = new ResizeObserver(() => { if (!view.current) fit(); draw(); });
    ro.observe(cv);
    (cv as unknown as { __fit: () => void }).__fit = () => { fit(); draw(); };
    draw();
    return () => { ro.disconnect(); cv.removeEventListener('pointerdown', down); cv.removeEventListener('pointermove', move); cv.removeEventListener('pointerup', up); cv.removeEventListener('pointercancel', up); cv.removeEventListener('wheel', wheel); };
  }, []);

  useEffect(() => { drawRef.current(); });
  // 部屋が変わった・問題を選んだときは全体が見えるように戻す
  useEffect(() => { (ref.current as unknown as { __fit?: () => void }).__fit?.(); }, [props.room.id, props.focusKey]);

  return <canvas ref={ref} class="plan" aria-label="部屋の平面図" />;
}
