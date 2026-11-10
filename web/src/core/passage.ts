// 通路の判定。物どうしの最短距離ではなく、入口から目的地まで連続してたどれる経路で評価する。
// 人を直径 W の円とみなし、経路上で通れる最大の W（＝経路のいちばん狭い所の幅）を求める。
import type { Vec2 } from './types';
import { bbox, pointInPolygon } from './geom';

export interface PassageGrid {
  cell: number;
  x0: number;
  z0: number;
  nx: number;
  nz: number;
  /** 各マスの中心から、壁・家具・障害物までの距離 [m]。通れないマスは 0。 */
  clearance: Float32Array;
}

function fillPolygon(mask: Uint8Array, poly: Vec2[], g: { cell: number; x0: number; z0: number; nx: number; nz: number }, value: number) {
  const b = bbox(poly);
  const i0 = Math.max(0, Math.floor((b.x0 - g.x0) / g.cell)), i1 = Math.min(g.nx - 1, Math.ceil((b.x1 - g.x0) / g.cell));
  const j0 = Math.max(0, Math.floor((b.z0 - g.z0) / g.cell)), j1 = Math.min(g.nz - 1, Math.ceil((b.z1 - g.z0) / g.cell));
  for (let j = j0; j <= j1; j++)
    for (let i = i0; i <= i1; i++) {
      const p = { x: g.x0 + (i + 0.5) * g.cell, z: g.z0 + (j + 0.5) * g.cell };
      if (pointInPolygon(p, poly)) mask[j * g.nx + i] = value;
    }
}

/** 1 次元の二乗距離変換 (Felzenszwalb & Huttenlocher) */
function edt1d(f: Float64Array, n: number, out: Float64Array, vIdx: Int32Array, zb: Float64Array) {
  let k = 0;
  vIdx[0] = 0; zb[0] = -Infinity; zb[1] = Infinity;
  for (let q = 1; q < n; q++) {
    let s: number;
    for (;;) {
      const p = vIdx[k];
      s = (f[q] + q * q - (f[p] + p * p)) / (2 * q - 2 * p);
      if (s <= zb[k] && k > 0) k--; else break;
    }
    k++; vIdx[k] = q; zb[k] = s; zb[k + 1] = Infinity;
  }
  k = 0;
  for (let q = 0; q < n; q++) {
    while (zb[k + 1] < q) k++;
    const p = vIdx[k];
    out[q] = (q - p) * (q - p) + f[p];
  }
}

export function buildPassageGrid(outline: Vec2[], blockers: Vec2[][], cell = 0.025): PassageGrid {
  const b = bbox(outline);
  const x0 = b.x0 - cell, z0 = b.z0 - cell;
  const nx = Math.ceil((b.x1 - b.x0) / cell) + 2, nz = Math.ceil((b.z1 - b.z0) / cell) + 2;
  const g = { cell, x0, z0, nx, nz };
  const free = new Uint8Array(nx * nz);
  fillPolygon(free, outline, g, 1);
  for (const poly of blockers) fillPolygon(free, poly, g, 0);

  const INF = 1e12;
  const d2 = new Float64Array(nx * nz);
  for (let i = 0; i < d2.length; i++) d2[i] = free[i] ? INF : 0;
  const n = Math.max(nx, nz);
  const f = new Float64Array(n), o = new Float64Array(n), vi = new Int32Array(n), zb = new Float64Array(n + 1);
  for (let i = 0; i < nx; i++) {
    for (let j = 0; j < nz; j++) f[j] = d2[j * nx + i];
    edt1d(f, nz, o, vi, zb);
    for (let j = 0; j < nz; j++) d2[j * nx + i] = o[j];
  }
  for (let j = 0; j < nz; j++) {
    for (let i = 0; i < nx; i++) f[i] = d2[j * nx + i];
    edt1d(f, nx, o, vi, zb);
    for (let i = 0; i < nx; i++) d2[j * nx + i] = o[i];
  }
  const clearance = new Float32Array(nx * nz);
  // 通れないマスの中心までの距離から半マス引き、マスの縁までの距離に近づける
  for (let i = 0; i < d2.length; i++) clearance[i] = free[i] ? Math.max(0, Math.sqrt(d2[i]) * cell - cell / 2) : 0;
  return { cell, x0, z0, nx, nz, clearance };
}

function cellOf(g: PassageGrid, p: Vec2): number {
  const i = Math.floor((p.x - g.x0) / g.cell), j = Math.floor((p.z - g.z0) / g.cell);
  if (i < 0 || j < 0 || i >= g.nx || j >= g.nz) return -1;
  return j * g.nx + i;
}

/** p の近く（radius 以内）で、まわりが最も空いているマスを返す。無ければ -1。 */
export function snapToFree(g: PassageGrid, p: Vec2, radius: number): number {
  const r = Math.ceil(radius / g.cell);
  const ci = Math.floor((p.x - g.x0) / g.cell), cj = Math.floor((p.z - g.z0) / g.cell);
  let best = -1, bestC = 0;
  for (let dj = -r; dj <= r; dj++)
    for (let di = -r; di <= r; di++) {
      if (di * di + dj * dj > r * r) continue;
      const i = ci + di, j = cj + dj;
      if (i < 0 || j < 0 || i >= g.nx || j >= g.nz) continue;
      const c = g.clearance[j * g.nx + i];
      if (c > bestC) { bestC = c; best = j * g.nx + i; }
    }
  return best;
}

export interface WidestPath {
  reachable: boolean;
  /** 経路上で確保できる幅の最大値（いちばん狭い所の幅）[m] */
  widthM: number;
  bottleneck: Vec2 | null;
  path: Vec2[];
}

/** from から to へ、いちばん狭い所ができるだけ広くなる経路を探す（最大ボトルネック経路）。 */
export function widestPath(g: PassageGrid, from: Vec2, to: Vec2, snapRadius = 0.3): WidestPath {
  const s = snapToFree(g, from, snapRadius), t = snapToFree(g, to, snapRadius);
  const none: WidestPath = { reachable: false, widthM: 0, bottleneck: null, path: [] };
  if (s < 0 || t < 0) return none;
  const N = g.nx * g.nz;
  const best = new Float32Array(N).fill(-1);
  const prev = new Int32Array(N).fill(-1);
  // 最大ヒープ
  const heap: number[] = [];
  const key: number[] = [];
  const push = (id: number, k: number) => {
    let i = heap.length; heap.push(id); key.push(k);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (key[p] >= key[i]) break;
      [heap[p], heap[i]] = [heap[i], heap[p]]; [key[p], key[i]] = [key[i], key[p]]; i = p;
    }
  };
  const pop = (): [number, number] => {
    const id = heap[0], k = key[0];
    const lid = heap.pop()!, lk = key.pop()!;
    if (heap.length) {
      heap[0] = lid; key[0] = lk;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1, r = l + 1;
        let m = i;
        if (l < heap.length && key[l] > key[m]) m = l;
        if (r < heap.length && key[r] > key[m]) m = r;
        if (m === i) break;
        [heap[m], heap[i]] = [heap[i], heap[m]]; [key[m], key[i]] = [key[i], key[m]]; i = m;
      }
    }
    return [id, k];
  };
  best[s] = g.clearance[s];
  push(s, best[s]);
  while (heap.length) {
    const [id, k] = pop();
    if (k < best[id]) continue;
    if (id === t) break;
    const i = id % g.nx, j = (id - i) / g.nx;
    for (const [di, dj] of [[1, 0], [-1, 0], [0, 1], [0, -1]] as const) {
      const ni = i + di, nj = j + dj;
      if (ni < 0 || nj < 0 || ni >= g.nx || nj >= g.nz) continue;
      const nid = nj * g.nx + ni;
      const c = g.clearance[nid];
      if (c <= 0) continue;
      const nk = Math.min(k, c);
      if (nk > best[nid]) { best[nid] = nk; prev[nid] = id; push(nid, nk); }
    }
  }
  if (best[t] <= 0) return none;
  // 幅 best[t] を保てるマスだけを使い、壁から離れた短い経路を引き直す（表示と、狭い所の位置の特定用）
  const widthLimit = best[t] - 1e-6;
  const cost = new Float64Array(N).fill(Infinity);
  prev.fill(-1);
  heap.length = 0; key.length = 0;
  cost[s] = 0;
  push(s, 0);
  while (heap.length) {
    const [id, nk] = pop();
    if (-nk > cost[id]) continue;
    if (id === t) break;
    const i = id % g.nx, j = (id - i) / g.nx;
    for (const [di, dj] of [[1, 0], [-1, 0], [0, 1], [0, -1]] as const) {
      const ni = i + di, nj = j + dj;
      if (ni < 0 || nj < 0 || ni >= g.nx || nj >= g.nz) continue;
      const nid = nj * g.nx + ni;
      const c = g.clearance[nid];
      if (c < widthLimit) continue;
      const nc = cost[id] + 1 + 0.05 / c;
      if (nc < cost[nid]) { cost[nid] = nc; prev[nid] = id; push(nid, -nc); }
    }
  }
  const path: Vec2[] = [];
  let bottleneck: Vec2 | null = null, minC = Infinity;
  for (let id = t; id >= 0; id = prev[id]) {
    const i = id % g.nx, j = (id - i) / g.nx;
    const p = { x: g.x0 + (i + 0.5) * g.cell, z: g.z0 + (j + 0.5) * g.cell };
    path.push(p);
    if (g.clearance[id] < minC) { minC = g.clearance[id]; bottleneck = p; }
    if (id === s) break;
  }
  path.reverse();
  return { reachable: true, widthM: best[t] * 2, bottleneck, path };
}

export { cellOf };
