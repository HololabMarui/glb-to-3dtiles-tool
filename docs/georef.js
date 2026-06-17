/**
 * georef.js
 * Python glb_tiles_georef.py からの移植ロジック
 * - ECEF 変換
 * - ENU 軸生成
 * - HPR 回転行列
 * - root.transform 生成
 * - box_from_bounds
 * - tileset.json children 構成生成
 */

const WGS84_A  = 6378137.0;
const WGS84_F  = 1.0 / 298.257223563;
const WGS84_E2 = WGS84_F * (2.0 - WGS84_F);

/** 度→ラジアン */
function deg2rad(d) { return d * Math.PI / 180; }

/** 経緯度高さ → ECEF [x, y, z] */
function ecefFromLonLatHeight(lonDeg, latDeg, height) {
  const lon = deg2rad(lonDeg), lat = deg2rad(latDeg);
  const sinLat = Math.sin(lat), cosLat = Math.cos(lat);
  const sinLon = Math.sin(lon), cosLon = Math.cos(lon);
  const n = WGS84_A / Math.sqrt(1.0 - WGS84_E2 * sinLat * sinLat);
  return [
    (n + height) * cosLat * cosLon,
    (n + height) * cosLat * sinLon,
    (n * (1.0 - WGS84_E2) + height) * sinLat,
  ];
}

/**
 * ENU 座標系の軸を ECEF で表した 3x3 行列 (列が East/North/Up)
 * 返値: [[row0], [row1], [row2]]  row-major, col = [E, N, U]
 */
function enuAxes(lonDeg, latDeg) {
  const lon = deg2rad(lonDeg), lat = deg2rad(latDeg);
  const sinLon = Math.sin(lon), cosLon = Math.cos(lon);
  const sinLat = Math.sin(lat), cosLat = Math.cos(lat);
  // Python: column_stack([east, north, up]) → m[row][col]
  return [
    [-sinLon,            -sinLat * cosLon,   cosLat * cosLon],
    [ cosLon,            -sinLat * sinLon,   cosLat * sinLon],
    [ 0,                  cosLat,             sinLat        ],
  ];
}

/** 3x3 行列積 (row-major) */
function mat3Mul(A, B) {
  const C = [[0,0,0],[0,0,0],[0,0,0]];
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++)
      for (let k = 0; k < 3; k++)
        C[i][j] += A[i][k] * B[k][j];
  return C;
}

function rotX(rad) {
  const c = Math.cos(rad), s = Math.sin(rad);
  return [[1,0,0],[0,c,-s],[0,s,c]];
}
function rotY(rad) {
  const c = Math.cos(rad), s = Math.sin(rad);
  return [[c,0,s],[0,1,0],[-s,0,c]];
}
function rotZ(rad) {
  const c = Math.cos(rad), s = Math.sin(rad);
  return [[c,-s,0],[s,c,0],[0,0,1]];
}

/**
 * HPR 回転行列
 * Heading は Cesium 準拠でコンパス方向（北=0°, 時計回り正）
 * → rotZ に負号を付けて反時計回りの数学正方向と合わせる
 */
function hprRotation(headingDeg, pitchDeg, rollDeg) {
  return mat3Mul(mat3Mul(rotZ(-deg2rad(headingDeg)), rotY(deg2rad(pitchDeg))), rotX(deg2rad(rollDeg)));
}

/**
 * 3D Tiles root.transform を生成する (16要素 column-major 配列)
 *
 * GLB は Y-up 座標系（+Y=上, -Z=前）。
 * 3D Tiles ENU は Z-up（+Z=上, +Y=北）。
 * rotX(+90°) を最内側に適用して Y-up → Z-up を変換する。
 * これにより Heading=0 でモデルが直立・北向きになる。
 */
function makeTransform(lonDeg, latDeg, height, headingDeg, pitchDeg, rollDeg) {
  const t = ecefFromLonLatHeight(lonDeg, latDeg, height);
  const rEnu   = enuAxes(lonDeg, latDeg);
  const rLocal = hprRotation(headingDeg, pitchDeg, rollDeg);
  const rAxisFix = rotX(Math.PI / 2); // GLB Y-up → ENU Z-up 変換
  const r = mat3Mul(mat3Mul(rEnu, rLocal), rAxisFix);

  // 4x4 row-major: m[:3,:3]=r, m[:3,3]=translation
  // m.T.reshape(-1) → column-major 16 values
  // m[i][j] → mT[j][i]
  // reshape row-by-row of mT:
  // col0=[r00,r10,r20,0], col1=[r01,r11,r21,0], col2=[r02,r12,r22,0], col3=[tx,ty,tz,1]
  return [
    r[0][0], r[1][0], r[2][0], 0,
    r[0][1], r[1][1], r[2][1], 0,
    r[0][2], r[1][2], r[2][2], 0,
    t[0],    t[1],    t[2],    1,
  ];
}

/**
 * バウンディングボックス → 3D Tiles box (12要素)
 * bounds: { min:[x,y,z], max:[x,y,z] }
 */
function boxFromBounds(bounds) {
  const cx = (bounds.min[0] + bounds.max[0]) / 2;
  const cy = (bounds.min[1] + bounds.max[1]) / 2;
  const cz = (bounds.min[2] + bounds.max[2]) / 2;
  const hx = (bounds.max[0] - bounds.min[0]) / 2;
  const hy = (bounds.max[1] - bounds.min[1]) / 2;
  const hz = (bounds.max[2] - bounds.min[2]) / 2;
  return [cx, cy, cz, hx, 0, 0, 0, hy, 0, 0, 0, hz];
}

/**
 * tileset.json オブジェクトを生成する
 * @param {Array} chunks - [{index, bounds:{min,max}, uri}]
 * @param {Object} params - {lon,lat,height,heading,pitch,roll,geometricError,refine}
 */
function buildTileset(chunks, params) {
  const { lon, lat, height, heading, pitch, roll, geometricError, refine } = params;

  // root bounds = union of all chunk bounds
  const rootMin = [Infinity, Infinity, Infinity];
  const rootMax = [-Infinity, -Infinity, -Infinity];
  for (const c of chunks) {
    for (let i = 0; i < 3; i++) {
      rootMin[i] = Math.min(rootMin[i], c.bounds.min[i]);
      rootMax[i] = Math.max(rootMax[i], c.bounds.max[i]);
    }
  }

  const children = chunks.map(c => ({
    boundingVolume: { box: boxFromBounds(c.bounds) },
    geometricError: 0,
    content: { uri: c.uri },
  }));

  return {
    asset: { version: '1.1' },
    geometricError: geometricError,
    root: {
      boundingVolume: { box: boxFromBounds({ min: rootMin, max: rootMax }) },
      geometricError: Math.max(1.0, geometricError / 2.0),
      refine: refine,
      transform: makeTransform(lon, lat, height, heading, pitch, roll),
      children,
    },
  };
}

/**
 * build_report.json 相当オブジェクトを生成する
 */
function buildReport(inputName, chunks, params) {
  return {
    input: inputName,
    chunk_count: chunks.length,
    parameters: params,
    chunks: chunks.map(c => ({
      index: c.index,
      mesh_count: c.meshCount,
      uri: c.uri,
      bounds: { min: c.bounds.min, max: c.bounds.max },
    })),
  };
}

export { makeTransform, boxFromBounds, buildTileset, buildReport };
