import type { Vec2, Placement, Furniture } from './types';

export const v = (x: number, z: number): Vec2 => ({ x, z });
export const add = (a: Vec2, b: Vec2) => v(a.x + b.x, a.z + b.z);
export const sub = (a: Vec2, b: Vec2) => v(a.x - b.x, a.z - b.z);
export const mul = (a: Vec2, s: number) => v(a.x * s, a.z * s);
export const dot = (a: Vec2, b: Vec2) => a.x * b.x + a.z * b.z;
export const cross = (a: Vec2, b: Vec2) => a.x * b.z - a.z * b.x;
export const len = (a: Vec2) => Math.hypot(a.x, a.z);
export const dist = (a: Vec2, b: Vec2) => Math.hypot(a.x - b.x, a.z - b.z);
export const norm = (a: Vec2) => {
  const l = len(a);
  return l < 1e-12 ? v(0, 0) : v(a.x / l, a.z / l);
};

/** Y 軸まわりの回転。Three.js の Object3D.rotation.y と同じ（+x が -z 側へ回る）。 */
export function rotY(p: Vec2, a: number): Vec2 {
  const c = Math.cos(a), s = Math.sin(a);
  return v(p.x * c + p.z * s, -p.x * s + p.z * c);
}

/** (x,z) 平面での符号付き面積。正なら反時計回り。 */
export function signedArea(poly: Vec2[]): number {
  let s = 0;
  for (let i = 0; i < poly.length; i++) {
    const a = poly[i], b = poly[(i + 1) % poly.length];
    s += a.x * b.z - b.x * a.z;
  }
  return s / 2;
}

/** 辺 a→b の室内側（左）を向く単位法線 */
export function inwardNormal(a: Vec2, b: Vec2): Vec2 {
  const d = norm(sub(b, a));
  return v(-d.z, d.x);
}

export function pointInPolygon(p: Vec2, poly: Vec2[]): boolean {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const a = poly[i], b = poly[j];
    if ((a.z > p.z) !== (b.z > p.z) && p.x < ((b.x - a.x) * (p.z - a.z)) / (b.z - a.z) + a.x) inside = !inside;
  }
  return inside;
}

export function pointSegDist(p: Vec2, a: Vec2, b: Vec2): number {
  const ab = sub(b, a);
  const l2 = dot(ab, ab);
  const t = l2 < 1e-18 ? 0 : Math.max(0, Math.min(1, dot(sub(p, a), ab) / l2));
  return dist(p, add(a, mul(ab, t)));
}

export function segIntersect(a: Vec2, b: Vec2, c: Vec2, d: Vec2): boolean {
  const d1 = cross(sub(b, a), sub(c, a));
  const d2 = cross(sub(b, a), sub(d, a));
  const d3 = cross(sub(d, c), sub(a, c));
  const d4 = cross(sub(d, c), sub(b, c));
  return ((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) && ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0));
}

export function segSegDist(a: Vec2, b: Vec2, c: Vec2, d: Vec2): number {
  if (segIntersect(a, b, c, d)) return 0;
  return Math.min(pointSegDist(a, c, d), pointSegDist(b, c, d), pointSegDist(c, a, b), pointSegDist(d, a, b));
}

export function pointPolyBoundaryDist(p: Vec2, poly: Vec2[]): { d: number; edge: number } {
  let best = Infinity, edge = -1;
  for (let i = 0; i < poly.length; i++) {
    const dd = pointSegDist(p, poly[i], poly[(i + 1) % poly.length]);
    if (dd < best) { best = dd; edge = i; }
  }
  return { d: best, edge };
}

/** 凸多角形どうしの関係。重なっていれば depth>0 と、a を押し出す向き push（単位ベクトル）。離れていれば gap>0。 */
export function convexRelation(a: Vec2[], b: Vec2[]): { overlap: boolean; depth: number; push: Vec2; gap: number } {
  let minOverlap = Infinity;
  let push = v(0, 0);
  for (const poly of [a, b]) {
    for (let i = 0; i < poly.length; i++) {
      const e = sub(poly[(i + 1) % poly.length], poly[i]);
      const axis = norm(v(-e.z, e.x));
      if (axis.x === 0 && axis.z === 0) continue;
      let amin = Infinity, amax = -Infinity, bmin = Infinity, bmax = -Infinity;
      for (const p of a) { const t = dot(p, axis); amin = Math.min(amin, t); amax = Math.max(amax, t); }
      for (const p of b) { const t = dot(p, axis); bmin = Math.min(bmin, t); bmax = Math.max(bmax, t); }
      const o1 = amax - bmin, o2 = bmax - amin;
      const o = Math.min(o1, o2);
      if (o <= 0) {
        let gap = Infinity;
        for (let m = 0; m < a.length; m++)
          for (let n = 0; n < b.length; n++)
            gap = Math.min(gap, segSegDist(a[m], a[(m + 1) % a.length], b[n], b[(n + 1) % b.length]));
        return { overlap: false, depth: 0, push: v(0, 0), gap };
      }
      if (o < minOverlap) {
        minOverlap = o;
        push = o1 < o2 ? mul(axis, -1) : axis;
      }
    }
  }
  return { overlap: true, depth: minOverlap, push, gap: 0 };
}

export function rectPoly(r: [number, number, number, number]): Vec2[] {
  const [x0, z0, x1, z1] = r;
  return [v(x0, z0), v(x1, z0), v(x1, z1), v(x0, z1)];
}

/** 家具の床への投影（4 頂点, 反時計回り）。ローカル: 幅=x, 奥行き=z, 正面=+z。 */
export function footprint(p: Placement, f: Pick<Furniture, 'w' | 'd'>): Vec2[] {
  const hw = f.w / 2, hd = f.d / 2;
  return [v(-hw, -hd), v(hw, -hd), v(hw, hd), v(-hw, hd)].map((c) => add(rotY(c, p.rotY), v(p.x, p.z)));
}

/** 家具の正面側に depth だけ伸びる矩形（使用空間） */
export function frontZone(p: Placement, f: Pick<Furniture, 'w' | 'd'>, depth: number): Vec2[] {
  const hw = f.w / 2, hd = f.d / 2;
  return [v(-hw, hd), v(hw, hd), v(hw, hd + depth), v(-hw, hd + depth)].map((c) => add(rotY(c, p.rotY), v(p.x, p.z)));
}

export function frontNormal(p: Placement): Vec2 {
  return rotY(v(0, 1), p.rotY);
}

export function polyCentroid(poly: Vec2[]): Vec2 {
  let x = 0, z = 0;
  for (const p of poly) { x += p.x; z += p.z; }
  return v(x / poly.length, z / poly.length);
}

export function bbox(poly: Vec2[]) {
  let x0 = Infinity, z0 = Infinity, x1 = -Infinity, z1 = -Infinity;
  for (const p of poly) { x0 = Math.min(x0, p.x); z0 = Math.min(z0, p.z); x1 = Math.max(x1, p.x); z1 = Math.max(z1, p.z); }
  return { x0, z0, x1, z1 };
}

/**
 * 凸多角形 poly と、単純多角形（部屋の輪郭）の関係。
 * outside: poly が輪郭の外へ出ている深さ（0 なら出ていない）。edge は最も関係する辺。
 * gap: 出ていない場合の、輪郭までの最短距離。
 */
export function convexVsOutline(poly: Vec2[], outline: Vec2[]): { outside: number; gap: number; edge: number } {
  let outside = 0, edge = -1;
  for (const p of poly) {
    if (!pointInPolygon(p, outline)) {
      const r = pointPolyBoundaryDist(p, outline);
      if (r.d > outside) { outside = r.d; edge = r.edge; }
    }
  }
  // 輪郭の角（柱の出隅など）が家具の内側へ食い込んでいる場合
  for (let i = 0; i < outline.length; i++) {
    const q = outline[i];
    if (pointInPolygon(q, poly)) {
      const r = pointPolyBoundaryDist(q, poly);
      if (r.d > outside) { outside = r.d; edge = i; }
    }
  }
  if (outside > 1e-9) return { outside, gap: 0, edge };
  let gap = Infinity;
  for (let i = 0; i < outline.length; i++) {
    const a = outline[i], b = outline[(i + 1) % outline.length];
    for (let k = 0; k < poly.length; k++) {
      const dd = segSegDist(poly[k], poly[(k + 1) % poly.length], a, b);
      if (dd < gap) { gap = dd; edge = i; }
    }
  }
  if (gap === 0) return { outside: 1e-6, gap: 0, edge };
  return { outside: 0, gap, edge };
}

export function intervalsOverlap(a0: number, a1: number, b0: number, b1: number): number {
  return Math.min(a1, b1) - Math.max(a0, b0);
}
