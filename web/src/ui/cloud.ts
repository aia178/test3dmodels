import { useEffect, useState } from 'preact/hooks';
import * as db from '../store/db';
import type { Room } from '../core/types';
import { surfacesUsable } from '../core/room';
import { surfaceFiles } from '../store/db';

export interface Cloud { pos: Float32Array; col: Uint8Array }

/** 復元した点群（部屋座標・縮尺補正前）。形式: float32 xyz × N のあとに uint8 rgb × N。 */
export function useCloud(roomId: string, enabled: boolean): Cloud | null {
  const [cloud, setCloud] = useState<Cloud | null>(null);
  useEffect(() => {
    let alive = true;
    setCloud(null);
    if (!enabled) return;
    db.get('blobs', `cloud:${roomId}`).then((b) => {
      if (!alive || !b || !(b.blob instanceof ArrayBuffer)) return;
      const n = Math.floor(b.blob.byteLength / 15);
      setCloud({ pos: new Float32Array(b.blob, 0, n * 3), col: new Uint8Array(b.blob, n * 12, n * 3) });
    });
    return () => { alive = false; };
  }, [roomId, enabled]);
  return cloud;
}

export interface SurfaceImages { walls: Map<number, string>; floor?: { url: string; bbox: [number, number, number, number] } }

/** 壁・床に写真を貼った画像（blob の URL）。部屋の形を手で直した後など、今の形に貼れない場合は null。 */
export function useSurfaces(room: Room | undefined, enabled: boolean): SurfaceImages | null {
  const [imgs, setImgs] = useState<SurfaceImages | null>(null);
  const usable = !!room && surfacesUsable(room);
  const key = room && usable ? `${room.id}:${surfaceFiles(room).join(',')}` : '';
  useEffect(() => {
    let alive = true;
    const urls: string[] = [];
    setImgs(null);
    if (!enabled || !room || !usable) return;
    (async () => {
      const s = room.surfaces!;
      const url = async (f: string) => {
        const b = await db.get('blobs', `surface:${room.id}:${f}`);
        if (!b || !(b.blob instanceof Blob)) return null;
        const u = URL.createObjectURL(b.blob); urls.push(u); return u;
      };
      const walls = new Map<number, string>();
      for (const w of s.walls) { const u = await url(w.file); if (u) walls.set(w.edge, u); }
      const fu = s.floor ? await url(s.floor.file) : null;
      if (alive && (walls.size || fu)) setImgs({ walls, ...(fu && s.floor ? { floor: { url: fu, bbox: s.floor.bbox } } : {}) });
    })();
    return () => { alive = false; urls.forEach((u) => URL.revokeObjectURL(u)); };
  }, [key, enabled]);
  return imgs;
}
