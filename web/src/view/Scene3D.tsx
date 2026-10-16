// 3D 表示。座標は部屋データと同じ（右手系・Y 上・床 y=0・単位 m）で、変換を挟まない。
import { useEffect, useRef } from 'preact/hooks';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import type { Furniture, Issue, Placement, Room, Vec2 } from '../core/types';
import { bbox, polyCentroid } from '../core/geom';
import { fixturePosition } from '../core/judge';
import type { SurfaceImages } from '../ui/cloud';

export interface SceneProps {
  room: Room;
  cloud?: Float32Array | null;
  cloudColors?: Uint8Array | null;
  items: { p: Placement; f: Furniture }[];
  selectedId: string | null;
  issue: Issue | null;
  issueTargets: Map<string, 'problem' | 'needs_check'>;
  showCloud: boolean;
  /** 壁・床に写真を貼って表示する（見た目のため。寸法は部屋データのまま）。渡すと点の代わりにこちらを表示する */
  surfaces?: SurfaceImages | null;
  onSelect: (id: string | null) => void;
  onMove: (id: string, x: number, z: number, done: boolean) => void;
}

const WALL_H_FALLBACK = 2.4;

function disposeGroup(g: THREE.Group) {
  g.traverse((o) => {
    const m = o as THREE.Mesh;
    m.geometry?.dispose?.();
    const mat = m.material as THREE.Material | THREE.Material[] | undefined;
    if (Array.isArray(mat)) mat.forEach((x) => x.dispose()); else mat?.dispose?.();
  });
  g.clear();
}

export function Scene3D(props: SceneProps) {
  const host = useRef<HTMLDivElement>(null);
  const api = useRef<{ sync: (p: SceneProps) => void; focus: (i: Issue | null, room: Room) => void } | null>(null);
  const propsRef = useRef(props);
  propsRef.current = props;

  useEffect(() => {
    const el = host.current!;
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true });
    } catch {
      el.innerHTML = '<p class="scene-fallback">この端末・ブラウザでは 3D 表示を使えません。平面図に切り替えてご利用ください。</p>';
      return;
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0xf6f3ee);
    el.appendChild(renderer.domElement);
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(50, 1, 0.05, 100);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.maxPolarAngle = Math.PI / 2 - 0.02;
    controls.enableDamping = true;
    scene.add(new THREE.HemisphereLight(0xffffff, 0xd8d2c6, 1.0));
    const sun = new THREE.DirectionalLight(0xffffff, 1.1); sun.position.set(3, 6, 2); scene.add(sun);

    const roomG = new THREE.Group(), furnG = new THREE.Group(), hlG = new THREE.Group(), cloudG = new THREE.Group();
    scene.add(roomG, furnG, hlG, cloudG);
    let roomKey = '', cloudKey = '', framed = '';
    const meshById = new Map<string, THREE.Mesh>();

    const loader = new THREE.TextureLoader();
    const photoMat = (url: string) => {
      const tex = loader.load(url);
      tex.colorSpace = THREE.SRGBColorSpace;
      tex.anisotropy = Math.min(8, renderer.capabilities.getMaxAnisotropy());
      return new THREE.MeshBasicMaterial({ map: tex, side: THREE.DoubleSide });
    };

    const buildRoom = (room: Room, cloudShown: boolean, surf: SurfaceImages | null | undefined) => {
      disposeGroup(roomG);
      if (room.outline.length < 3) return;
      const H = room.ceiling.height ?? WALL_H_FALLBACK;
      const shape = new THREE.Shape(room.outline.map((p) => new THREE.Vector2(p.x, p.z)));
      const floorGeo = new THREE.ShapeGeometry(shape);
      let floorMat: THREE.Material = new THREE.MeshLambertMaterial({ color: cloudShown ? 0xf1eee8 : 0xe9e2d5, side: THREE.DoubleSide });
      if (surf?.floor) {
        // 床の写真は輪郭を囲む長方形で作ってある（左→右が x、上→下が z）。形の頂点ごとに画像の位置を当てる
        const [x0, z0, x1, z1] = surf.floor.bbox;
        const pos = floorGeo.getAttribute('position') as THREE.BufferAttribute;
        const uv = new Float32Array(pos.count * 2);
        for (let i = 0; i < pos.count; i++) { uv[i * 2] = (pos.getX(i) - x0) / (x1 - x0); uv[i * 2 + 1] = 1 - (pos.getY(i) - z0) / (z1 - z0); }
        floorGeo.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
        floorMat = photoMat(surf.floor.url);
      }
      const floor = new THREE.Mesh(floorGeo, floorMat);
      if (cloudShown) floor.position.y = -0.03; // 復元した床の点を隠さないよう、少し下に置く
      floor.rotation.x = Math.PI / 2; // Shape の (x,y) を (x, 0, z) へ。y→z なので向きは保たれる
      floor.name = 'floor'; roomG.add(floor);
      room.outline.forEach((a, i) => {
        const b = room.outline[(i + 1) % room.outline.length];
        const st = room.edges[i]?.status ?? 'manual';
        const L = Math.hypot(b.x - a.x, b.z - a.z);
        if (L < 1e-4) return;
        const wallUrl = surf?.walls.get(i);
        if (wallUrl) {   // 写真を貼った壁（左→右が辺の始点→終点、上→下が天井→床）
          const wall = new THREE.Mesh(new THREE.PlaneGeometry(L, H), photoMat(wallUrl));
          wall.position.set((a.x + b.x) / 2, H / 2, (a.z + b.z) / 2);
          wall.rotation.y = -Math.atan2(b.z - a.z, b.x - a.x);
          roomG.add(wall);
          return;
        }
        if (cloudShown) {
          const line = new THREE.Line(
            new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(a.x, 0.01, a.z), new THREE.Vector3(b.x, 0.01, b.z)]),
            st === 'unobserved' ? new THREE.LineDashedMaterial({ color: 0xb3541e, dashSize: 0.12, gapSize: 0.1 }) : new THREE.LineBasicMaterial({ color: 0x2f6f5e }));
          line.computeLineDistances(); roomG.add(line);
          return;
        }
        const geo = new THREE.PlaneGeometry(L, H);
        const unobs = st === 'unobserved';
        if (surf && unobs) {   // 写真で表示しているときは、形の分からない壁は点線の枠だけ（色を重ねると写真がくすむ）
          const edges = new THREE.LineSegments(new THREE.EdgesGeometry(geo), new THREE.LineDashedMaterial({ color: 0xb3541e, dashSize: 0.12, gapSize: 0.1 }));
          edges.computeLineDistances();
          edges.position.set((a.x + b.x) / 2, H / 2, (a.z + b.z) / 2); edges.rotation.y = -Math.atan2(b.z - a.z, b.x - a.x);
          roomG.add(edges);
          return;
        }
        // 未確認の壁は面を塗らず、枠と斜線の柄で「形が分からない」ことを示す
        const mat = new THREE.MeshLambertMaterial({ color: unobs ? 0xb3541e : st === 'observed' ? 0xd9d6cc : 0xcfd3cc, transparent: true, opacity: unobs ? 0.12 : 0.55, side: THREE.DoubleSide, depthWrite: false });
        const wall = new THREE.Mesh(geo, mat);
        wall.position.set((a.x + b.x) / 2, H / 2, (a.z + b.z) / 2);
        wall.rotation.y = -Math.atan2(b.z - a.z, b.x - a.x);
        roomG.add(wall);
        const edges = new THREE.LineSegments(new THREE.EdgesGeometry(geo), unobs
          ? new THREE.LineDashedMaterial({ color: 0xb3541e, dashSize: 0.12, gapSize: 0.1 })
          : new THREE.LineBasicMaterial({ color: st === 'observed' ? 0x8a948f : 0x3d4a45 }));
        edges.computeLineDistances(); edges.position.copy(wall.position); edges.rotation.copy(wall.rotation); roomG.add(edges);
      });
      for (const ob of room.obstacles) {
        const [x0, z0, x1, z1] = ob.rect;
        const geo = new THREE.BoxGeometry(x1 - x0, ob.yMax - ob.yMin, z1 - z0);
        const m = new THREE.Mesh(geo, new THREE.MeshLambertMaterial({ color: ob.confirm === 'auto' ? 0xc98a5e : 0x7d8a84, transparent: true, opacity: 0.85 }));
        m.position.set((x0 + x1) / 2, (ob.yMin + ob.yMax) / 2, (z0 + z1) / 2); roomG.add(m);
      }
      for (const fx of room.fixtures) {
        const pos = fixturePosition(room, fx);
        const color = { outlet: 0xffffff, lan: 0xffffff, tv: 0xffffff, door: 0x2f6f5e, window: 0x8fb8d6, aircon: 0xf2f2f2 }[fx.type];
        const depth = fx.type === 'aircon' ? 0.22 : 0.02;
        const m = new THREE.Mesh(new THREE.BoxGeometry(fx.width, fx.height, depth), new THREE.MeshLambertMaterial({ color, transparent: fx.type === 'door' || fx.type === 'window', opacity: fx.type === 'door' ? 0.45 : fx.type === 'window' ? 0.6 : 1 }));
        m.position.set(pos.center.x + pos.inward.x * depth / 2, fx.bottom + fx.height / 2, pos.center.z + pos.inward.z * depth / 2);
        m.rotation.y = -Math.atan2(pos.dir.z, pos.dir.x);
        roomG.add(m);
        const outline = new THREE.LineSegments(new THREE.EdgesGeometry(m.geometry), new THREE.LineBasicMaterial({ color: 0x2a2f2c }));
        outline.position.copy(m.position); outline.rotation.copy(m.rotation); roomG.add(outline);
      }
    };

    const buildCloud = (P: SceneProps) => {
      disposeGroup(cloudG);
      if (!P.cloud || !P.showCloud) return;
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(P.cloud, 3));
      if (P.cloudColors) geo.setAttribute('color', new THREE.BufferAttribute(P.cloudColors, 3, true));
      const pts = new THREE.Points(geo, new THREE.PointsMaterial({ size: 0.024, vertexColors: !!P.cloudColors, color: P.cloudColors ? 0xffffff : 0x6f7f78 }));
      pts.scale.setScalar(P.room.scale.factor); // 点群は縮尺補正前の値で保存しているので、表示のとき同じ縮尺を掛ける
      cloudG.add(pts);
    };

    const buildFurniture = (P: SceneProps) => {
      disposeGroup(furnG); meshById.clear();
      for (const { p, f } of P.items) {
        const flag = P.issueTargets.get(p.id);
        const geo = new THREE.BoxGeometry(f.w, f.h, f.d);
        const mesh = new THREE.Mesh(geo, new THREE.MeshLambertMaterial({ color: new THREE.Color(f.color), transparent: !f.dimConfirmed, opacity: f.dimConfirmed ? 1 : 0.7 }));
        mesh.position.set(p.x, p.elevation + f.h / 2, p.z);
        mesh.rotation.y = p.rotY;
        mesh.userData.pid = p.id;
        furnG.add(mesh); meshById.set(p.id, mesh);
        const sel = P.selectedId === p.id;
        const lineColor = sel ? 0x2f6f5e : flag === 'problem' ? 0xb3261e : flag === 'needs_check' ? 0xa15c00 : 0x4c5651;
        const edge = new THREE.LineSegments(new THREE.EdgesGeometry(geo), flag === 'needs_check' && !sel
          ? new THREE.LineDashedMaterial({ color: lineColor, dashSize: 0.08, gapSize: 0.05 }) : new THREE.LineBasicMaterial({ color: lineColor }));
        edge.computeLineDistances(); mesh.add(edge);
        // 正面の印
        const front = new THREE.Mesh(new THREE.PlaneGeometry(Math.min(0.3, f.w * 0.5), Math.min(0.08, f.h * 0.3)), new THREE.MeshBasicMaterial({ color: 0x2a2f2c, transparent: true, opacity: 0.5 }));
        front.position.set(0, f.h / 2 - Math.min(0.1, f.h * 0.3), f.d / 2 + 0.002); mesh.add(front);
        if (sel) {
          const ring = new THREE.Mesh(new THREE.RingGeometry(Math.hypot(f.w, f.d) / 2 + 0.03, Math.hypot(f.w, f.d) / 2 + 0.06, 48), new THREE.MeshBasicMaterial({ color: 0x2f6f5e, side: THREE.DoubleSide }));
          ring.rotation.x = -Math.PI / 2; ring.position.set(p.x, 0.005, p.z); furnG.add(ring);
        }
      }
    };

    const buildHighlight = (P: SceneProps) => {
      disposeGroup(hlG);
      const hl = P.issue?.highlight; if (!hl) return;
      const color = P.issue!.status === 'problem' ? 0xb3261e : 0xa15c00;
      const line = (pts: Vec2[], closed: boolean) => {
        const arr = pts.map((q) => new THREE.Vector3(q.x, 0.02, q.z));
        if (closed) arr.push(arr[0].clone());
        const l = new THREE.Line(new THREE.BufferGeometry().setFromPoints(arr), new THREE.LineDashedMaterial({ color, dashSize: 0.08, gapSize: 0.05 }));
        l.computeLineDistances(); hlG.add(l);
      };
      if (hl.polygon) {
        line(hl.polygon, true);
        const sh = new THREE.Mesh(new THREE.ShapeGeometry(new THREE.Shape(hl.polygon.map((q) => new THREE.Vector2(q.x, q.z)))), new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.22, side: THREE.DoubleSide, depthWrite: false }));
        sh.rotation.x = Math.PI / 2; sh.position.y = 0.015; hlG.add(sh);
      }
      if (hl.path) line(hl.path, false);
      if (hl.point) {
        const pin = new THREE.Mesh(new THREE.ConeGeometry(0.06, 0.25, 4), new THREE.MeshBasicMaterial({ color }));
        pin.rotation.x = Math.PI; pin.position.set(hl.point.x, 0.4, hl.point.z); hlG.add(pin);
        const ring = new THREE.Mesh(new THREE.RingGeometry(0.1, 0.13, 32), new THREE.MeshBasicMaterial({ color, side: THREE.DoubleSide }));
        ring.rotation.x = -Math.PI / 2; ring.position.set(hl.point.x, 0.02, hl.point.z); hlG.add(ring);
      }
    };

    const frame = (room: Room) => {
      if (!room.outline.length) return;
      const b = bbox(room.outline);
      const cx = (b.x0 + b.x1) / 2, cz = (b.z0 + b.z1) / 2, r = Math.max(b.x1 - b.x0, b.z1 - b.z0);
      // 復元できた壁と、撮っていない側（未確認の辺）の両方がある部屋は、撮っていない側から壁を見る
      const mid = (i: number) => { const a = room.outline[i], q = room.outline[(i + 1) % room.outline.length]; return { x: (a.x + q.x) / 2, z: (a.z + q.z) / 2, w: Math.hypot(q.x - a.x, q.z - a.z) }; };
      const mean = (idx: number[]) => { let x = 0, z = 0, w = 0; for (const i of idx) { const m = mid(i); x += m.x * m.w; z += m.z * m.w; w += m.w; } return { x: x / w, z: z / w }; };
      const un = room.edges.map((e, i) => (e.status === 'unobserved' ? i : -1)).filter((i) => i >= 0);
      const ob = room.edges.map((e, i) => (e.status !== 'unobserved' ? i : -1)).filter((i) => i >= 0);
      if (un.length && ob.length) {
        const u = mean(un), o = mean(ob);
        const dx = u.x - o.x, dz = u.z - o.z, l = Math.hypot(dx, dz) || 1;
        controls.target.set(o.x * 0.6 + cx * 0.4, 1.0, o.z * 0.6 + cz * 0.4);
        camera.position.set(cx + (dx / l) * r * 1.1, r * 0.55 + 1.4, cz + (dz / l) * r * 1.1);
        return;
      }
      controls.target.set(cx, 0.6, cz);
      camera.position.set(cx + r * 0.2, r * 1.15 + 1, cz + r * 1.05);
    };

    const sync = (P: SceneProps) => {
      const cloudShown = P.showCloud && !!P.cloud;
      const sk = P.surfaces ? `${[...P.surfaces.walls.values()].join(',')}:${P.surfaces.floor?.url ?? ''}` : '';
      const rk = `${P.room.id}:${P.room.geometryVersion}:${P.room.updatedAt}:${cloudShown}:${sk}`;
      if (rk !== roomKey) { roomKey = rk; buildRoom(P.room, cloudShown, P.surfaces); }
      const ck = `${P.room.id}:${P.showCloud}:${P.cloud?.length ?? 0}:${P.room.scale.factor}`;
      if (ck !== cloudKey) { cloudKey = ck; buildCloud(P); }
      if (framed !== P.room.id) { framed = P.room.id; frame(P.room); }
      buildFurniture(P); buildHighlight(P);
    };

    const focus = (issue: Issue | null, room: Room) => {
      const hl = issue?.highlight; if (!hl) return;
      const c = hl.point ?? (hl.polygon ? polyCentroid(hl.polygon) : hl.path ? hl.path[Math.floor(hl.path.length / 2)] : null);
      if (!c) return;
      const rc = polyCentroid(room.outline);
      const dir = new THREE.Vector3(rc.x - c.x, 0, rc.z - c.z);
      if (dir.lengthSq() < 0.01) dir.set(0, 0, 1);
      dir.normalize();
      controls.target.set(c.x, 0.5, c.z);
      camera.position.set(c.x + dir.x * 2.2, 2.4, c.z + dir.z * 2.2);
    };
    api.current = { sync, focus };

    // 家具をつまんで床の上を動かす
    const ray = new THREE.Raycaster(), floorPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
    let dragging: { id: string; dx: number; dz: number } | null = null, downXY = { x: 0, y: 0 }, moved = false;
    const ndc = (e: PointerEvent) => { const r = renderer.domElement.getBoundingClientRect(); return new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1); };
    const onFloor = (e: PointerEvent) => { ray.setFromCamera(ndc(e), camera); const o = new THREE.Vector3(); return ray.ray.intersectPlane(floorPlane, o) ? o : null; };
    const down = (e: PointerEvent) => {
      downXY = { x: e.clientX, y: e.clientY }; moved = false;
      ray.setFromCamera(ndc(e), camera);
      const hit = ray.intersectObjects([...meshById.values()], false)[0];
      if (hit) {
        const id = hit.object.userData.pid as string;
        propsRef.current.onSelect(id);
        const f = onFloor(e); const m = meshById.get(id)!;
        if (f) { dragging = { id, dx: m.position.x - f.x, dz: m.position.z - f.z }; controls.enabled = false; try { renderer.domElement.setPointerCapture(e.pointerId); } catch { /* 取得できなくても続ける */ } }
      }
    };
    const move = (e: PointerEvent) => {
      if (Math.hypot(e.clientX - downXY.x, e.clientY - downXY.y) > 4) moved = true;
      if (!dragging || !moved) return;
      const f = onFloor(e); if (f) propsRef.current.onMove(dragging.id, f.x + dragging.dx, f.z + dragging.dz, false);
    };
    const up = (e: PointerEvent) => {
      if (dragging) {
        const f = onFloor(e);
        if (moved && f) propsRef.current.onMove(dragging.id, f.x + dragging.dx, f.z + dragging.dz, true);
        dragging = null; controls.enabled = true;
      } else if (!moved) {
        ray.setFromCamera(ndc(e), camera);
        if (!ray.intersectObjects([...meshById.values()], false).length) propsRef.current.onSelect(null);
      }
    };
    renderer.domElement.addEventListener('pointerdown', down);
    renderer.domElement.addEventListener('pointermove', move);
    renderer.domElement.addEventListener('pointerup', up);
    renderer.domElement.addEventListener('pointercancel', up);

    const resize = () => { const w = el.clientWidth, h = el.clientHeight; if (!w || !h) return; renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix(); };
    const ro = new ResizeObserver(resize); ro.observe(el); resize();
    let raf = 0;
    const loop = () => { raf = requestAnimationFrame(loop); controls.update(); renderer.render(scene, camera); };
    loop();
    sync(propsRef.current);
    return () => { cancelAnimationFrame(raf); ro.disconnect(); controls.dispose(); [roomG, furnG, hlG, cloudG].forEach(disposeGroup); renderer.dispose(); renderer.domElement.remove(); api.current = null; };
  }, []);

  useEffect(() => { api.current?.sync(props); });
  useEffect(() => { api.current?.focus(props.issue, props.room); }, [props.issue?.id, props.issue?.title]);

  return <div ref={host} class="scene" aria-label="部屋の3D表示" />;
}
