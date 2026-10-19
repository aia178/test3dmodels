// データ定義。内部単位はすべてメートル [m]。角度はラジアン（Fixture.door.maxAngleDeg のみ度）。
// 座標系: 右手系・Y が上・床は y=0。平面図は (x, z)。
// 部屋の輪郭 outline は (x,z) を通常の2D平面とみたとき反時計回り（符号付き面積が正）。
// 辺 i は outline[i] → outline[i+1]。室内は進行方向の左側。

export type Vec2 = { x: number; z: number };

/** その値を誰が決めたか。AI の再実行で confirmed / manual を上書きしないこと。 */
export type Confirm = 'auto' | 'confirmed' | 'manual';

/** observed: 写真から壁として復元 / unobserved: 撮影されておらず形が分からない / confirmed: 人が確認 / manual: 人が入力 */
export type EdgeStatus = 'observed' | 'unobserved' | 'confirmed' | 'manual';

export interface Obstacle {
  id: string;
  kind: 'pillar' | 'beam' | 'protrusion' | 'unknown_vertical';
  label?: string;
  /** [x0, z0, x1, z1] 軸平行の矩形 */
  rect: [number, number, number, number];
  yMin: number;
  yMax: number;
  confirm: Confirm;
}

export type FixtureType = 'outlet' | 'door' | 'window' | 'lan' | 'tv' | 'aircon';

export interface Fixture {
  id: string;
  type: FixtureType;
  edgeIndex: number;
  /** 辺の始点から設備の中心までの距離 */
  offset: number;
  width: number;
  height: number;
  /** 床から設備の下端まで */
  bottom: number;
  /** manual: 人が位置を入力 / detected: 画像から自動検出（未実装のため現状は使われない） */
  source: 'manual' | 'detected';
  confirmed: boolean;
  door?: {
    hinge: 'start' | 'end';
    opensInto: 'room' | 'outside' | 'sliding';
    maxAngleDeg: number;
    isEntrance: boolean;
  };
  note?: string;
}

export interface Measurement {
  id: string;
  label: string;
  a: Vec2;
  b: Vec2;
  /** a, b の床からの高さ。写真の上で選んだ点のとき入る。両方あれば長さは 3D の距離で測る。 */
  ya?: number;
  yb?: number;
  /** 写真の上で選んだ場合の写真番号と画素位置（選び直しや確認用） */
  photo?: { n: number; ua: number; va: number; ub: number; vb: number };
  /** メジャーなどで実際に測った長さ */
  measuredM: number;
  /** calibration: 縮尺合わせに使った / validation: 精度の確認用（縮尺合わせには使っていない） */
  role: 'calibration' | 'validation';
  at: string;
  condition?: string;
}

export interface Room {
  id: string;
  name: string;
  purpose: string;
  kind: 'current' | 'viewing';
  source: 'photo' | 'manual';
  createdAt: string;
  updatedAt: string;
  outline: Vec2[];
  edges: { status: EdgeStatus }[];
  ceiling: { height: number | null; status: EdgeStatus };
  obstacles: Obstacle[];
  fixtures: Fixture[];
  measurements: Measurement[];
  /** model_estimate: 復元モデルの推定のまま / calibrated: 実測1か所で合わせた / marker: 写真に写した印刷マーカーで合わせた / manual: 寸法入力 */
  scale: { factor: number; status: 'model_estimate' | 'calibrated' | 'marker' | 'manual' };
  /** 形・縮尺が変わるたびに増やす。判定結果はこの版に結び付く。 */
  geometryVersion: number;
  job?: { id: string; model: string; nPhotos: number };
  /** 実測なしで出した、復元の信頼しやすさの目安（サーバーが計算）。good でも誤差が小さいという保証ではない。 */
  captureQuality?: { level: 'good' | 'poor'; reasons: string[]; disagreementMm: number | null; overlap: number; nPhotos: number;
    /** 写真が多く、範囲に分けて復元してつないだ場合の範囲の数（2 以上）。つなぎ目をまたぐ寸法は、1 回で復元した場合より当てにならない */
    chunks?: number };
  /** 写真の上で位置を選ぶためのデータがある写真（モデルが見た大きさ） */
  views?: { n: number; w: number; h: number }[];
  hasPointCloud: boolean;
  /**
   * 壁と床に元の写真を貼った画像（見た目のため。寸法には使わない）。画像そのものは blobs の `surface:<部屋>:<file>`。
   * refOutline は画像を作ったときの輪郭（縮尺合わせのたびに一緒に伸縮する）。今の輪郭と食い違ったら（形を手で直したら）表示しない。
   */
  surfaces?: { walls: { edge: number; file: string }[]; floor?: { file: string; bbox: [number, number, number, number] }; refOutline: Vec2[] };
  history: { at: string; action: string }[];
}

export type DimSource = 'measured' | 'product_page' | 'estimated' | 'unknown';

export interface Furniture {
  id: string;
  name: string;
  category: string;
  /** 幅(正面から見た左右)・奥行き・高さ */
  w: number;
  d: number;
  h: number;
  dimSource: DimSource;
  dimConfirmed: boolean;
  dimNote?: string;
  /** 現状は直方体のみ。脚の隙間や空洞は表現しない。 */
  shape: 'box';
  /** 脚の下の空き高さ。分からなければ null */
  legClearance: number | null;
  ownership: 'owned' | 'planned';
  /** 正面(ローカル +z)側に必要な空間 */
  useSpace: { kind: 'none' | 'drawer' | 'swing_door' | 'chair' | 'access' | 'unknown'; depth: number };
  product?: { url: string; shop?: string; title?: string; price?: number; fetchedAt?: string; affiliate: boolean };
  photoIds: string[];
  color: string;
  createdAt: string;
}

export interface Placement {
  id: string;
  furnitureId: string;
  x: number;
  z: number;
  /** Y 軸まわりの回転。Three.js の rotation.y と同じ向き。 */
  rotY: number;
  elevation: number;
}

export interface Layout {
  id: string;
  roomId: string;
  name: string;
  items: Placement[];
  updatedAt: string;
}

export interface JudgeSettings {
  /** 通路として確保したい幅。利用者が決める目安で、法令や安全の基準ではない。 */
  passageWidthM: number;
  /** コンセント類の前に空けたい奥行き */
  outletAccessM: number;
}

export type IssueStatus = 'problem' | 'needs_check' | 'clear';

export interface Issue {
  id: string;
  check: 'wall' | 'obstacle' | 'furniture' | 'ceiling' | 'door_swing' | 'use_space' | 'passage' | 'fixture' | 'input';
  status: IssueStatus;
  /** 関係する配置・設備・障害物の id */
  targets: string[];
  title: string;
  reason: string;
  /** 判断に使った寸法（cm 表記の説明つき） */
  dims: { label: string; valueM: number }[];
  /** 判断できなかった・確認が要る条件 */
  unverified: string[];
  suggestion?: string;
  /** この分だけ動かすと解消する見込みの移動量 */
  fix?: { placementId: string; dx: number; dz: number };
  /** 画面で強調する場所 */
  highlight?: { polygon?: Vec2[]; path?: Vec2[]; point?: Vec2 };
}

export interface JudgeResult {
  issues: Issue[];
  geometryVersion: number;
  toleranceM: number;
  toleranceBasis: string;
  checkedAt: string;
}
