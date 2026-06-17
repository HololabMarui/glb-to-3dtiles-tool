/**
 * app.js
 * GLB → 3D Tiles 変換ツール メインロジック
 */
import { buildTileset, buildReport } from './georef.js';

import * as THREE from 'three';
import { GLTFLoader }         from 'three/addons/loaders/GLTFLoader.js';
import { GLTFExporter }       from 'three/addons/exporters/GLTFExporter.js';
import { mergeGeometries }    from 'three/addons/utils/BufferGeometryUtils.js';

// ── DOM helpers ───────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);

// ── State ─────────────────────────────────────────────────────────────
let loadedGLB      = null;
let loadedFileName = '';
let marker         = null;
let map            = null;

// ── Tab switching ─────────────────────────────────────────────────────
function setTab(name) {
  ['guide', 'tool'].forEach(t => {
    $('tab-' + t).classList.toggle('active', t === name);
    // CSS の display:none を上書きするため '' ではなく 'block' を明示する
    $('page-' + t).style.display = t === name ? 'block' : 'none';
  });
}
$('tab-guide').addEventListener('click', () => setTab('guide'));
$('tab-tool').addEventListener('click',  () => setTab('tool'));

// ── GLB file input ────────────────────────────────────────────────────
$('glbInput').addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  loadedFileName = file.name;
  loadedGLB = await file.arrayBuffer();
  $('glbStatus').textContent = `✓ ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
  $('glbStatus').className = 'status ok';
  updatePreview();
});

// ── 3D Preview (Three.js) ─────────────────────────────────────────────
let previewRenderer = null;
let previewAnimId   = null;

function initPreview() {
  const canvas = $('previewCanvas');
  previewRenderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  previewRenderer.setPixelRatio(devicePixelRatio);

  const scene  = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 10000);
  scene.add(new THREE.AmbientLight(0xffffff, 0.8));
  const dir = new THREE.DirectionalLight(0xffffff, 1.0);
  dir.position.set(5, 10, 7);
  scene.add(dir);

  const pivot = new THREE.Object3D();
  scene.add(pivot);
  window._previewPivot = pivot;

  function resize() {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    previewRenderer.setSize(w, h, false);
  }
  new ResizeObserver(resize).observe(canvas);
  resize();

  let drag = false, lastX = 0, lastY = 0, rotX = 0, rotY = 0;
  canvas.addEventListener('mousedown', e => { drag = true; lastX = e.clientX; lastY = e.clientY; });
  window.addEventListener('mouseup',   () => { drag = false; });
  canvas.addEventListener('mousemove', e => {
    if (!drag) return;
    rotY += (e.clientX - lastX) * 0.5;
    rotX += (e.clientY - lastY) * 0.5;
    lastX = e.clientX; lastY = e.clientY;
    pivot.rotation.y = THREE.MathUtils.degToRad(rotY);
    pivot.rotation.x = THREE.MathUtils.degToRad(rotX);
  });
  canvas.addEventListener('wheel', e => {
    e.preventDefault(); // ページスクロールを抑制
    camera.position.z *= 1 + e.deltaY * 0.001;
  }, { passive: false });

  window._previewCamera = camera;

  function animate() {
    previewAnimId = requestAnimationFrame(animate);
    previewRenderer.render(scene, camera);
  }
  animate();
}

function updatePreview() {
  if (!loadedGLB) return;
  const pivot = window._previewPivot;
  while (pivot.children.length) pivot.remove(pivot.children[0]);

  const loader = new GLTFLoader();
  loader.parse(loadedGLB.slice(0), '', (gltf) => {
    const box    = new THREE.Box3().setFromObject(gltf.scene);
    const center = box.getCenter(new THREE.Vector3());
    const size   = box.getSize(new THREE.Vector3()).length();
    gltf.scene.position.sub(center);
    pivot.add(gltf.scene);
    const cam = window._previewCamera;
    cam.position.set(0, 0, size * 1.2);
    cam.near = size * 0.001;
    cam.far  = size * 100;
    cam.updateProjectionMatrix();
  }, (err) => console.warn('Preview error:', err));
}

// ── MapLibre GL JS ────────────────────────────────────────────────────
function initMap() {
  map = new maplibregl.Map({
    container: 'map',
    style: 'https://tile.openstreetmap.jp/styles/osm-bright/style.json',
    center: [parseFloat($('lon').value), parseFloat($('lat').value)],
    zoom: 14,
  });
  map.on('load', placeMarker);
  map.on('click', (e) => {
    $('lon').value = e.lngLat.lng.toFixed(7);
    $('lat').value = e.lngLat.lat.toFixed(7);
    placeMarker();
  });
}

function placeMarker() {
  const lon = parseFloat($('lon').value);
  const lat = parseFloat($('lat').value);
  if (isNaN(lon) || isNaN(lat)) return;
  if (marker) {
    marker.setLngLat([lon, lat]);
  } else {
    marker = new maplibregl.Marker({ color: '#6ea8fe', draggable: true })
      .setLngLat([lon, lat]).addTo(map);
    marker.on('dragend', () => {
      const ll = marker.getLngLat();
      $('lon').value = ll.lng.toFixed(7);
      $('lat').value = ll.lat.toFixed(7);
    });
  }
  map.flyTo({ center: [lon, lat], zoom: Math.max(map.getZoom(), 13) });
}
['lon', 'lat'].forEach(id => $(id).addEventListener('change', placeMarker));

// ── GLB 読み込み → ジオメトリ統合 ────────────────────────────────────
/**
 * GLB から全メッシュをフラット化し、マージした結果を返す
 * @returns {{ geometry: THREE.BufferGeometry, material: THREE.Material }}
 */
function loadAndMerge(arrayBuffer) {
  return new Promise((resolve, reject) => {
    const loader = new GLTFLoader();
    loader.parse(arrayBuffer.slice(0), '', (gltf) => {
      gltf.scene.updateWorldMatrix(true, true);

      const geoms = [];
      let material = new THREE.MeshStandardMaterial();

      gltf.scene.traverse(obj => {
        if (!obj.isMesh) return;
        const geom = obj.geometry.clone();
        geom.applyMatrix4(obj.matrixWorld);
        // 必要な attribute のみ残す（mergeGeometries は attribute が一致している必要がある）
        ['position', 'normal', 'uv'].forEach(name => {
          if (!geom.attributes[name]) geom.deleteAttribute(name);
        });
        // position だけは必須
        if (!geom.attributes.position) return;
        geoms.push(geom);
        if (Array.isArray(obj.material)) material = obj.material[0];
        else material = obj.material;
      });

      if (geoms.length === 0) return reject(new Error('メッシュが見つかりません'));

      // attribute セットを最小公倍数に揃える
      const allAttrs = new Set(geoms.flatMap(g => Object.keys(g.attributes)));
      geoms.forEach(g => {
        allAttrs.forEach(name => {
          if (!g.attributes[name]) {
            // ない attribute はダミーで埋める（mergeGeometries エラー回避）
            const ref = geoms.find(gg => gg.attributes[name]);
            if (ref) {
              const is  = ref.attributes[name].itemSize;
              const arr = new Float32Array(g.attributes.position.count * is);
              g.setAttribute(name, new THREE.BufferAttribute(arr, is));
            }
          }
        });
      });

      const merged = mergeGeometries(geoms, false);
      if (!merged) return reject(new Error('ジオメトリのマージに失敗しました'));
      resolve({ geometry: merged, material });
    }, reject);
  });
}

// ── ジオメトリを N 分割（面ベース）────────────────────────────────────
/**
 * BufferGeometry を面数で N 等分して返す
 * インデックス付き・なし両対応
 */
function splitGeometryByFaces(geometry, n) {
  // インデックスがない場合はインデックス付きに変換
  let geom = geometry.index ? geometry : geometry.toNonIndexed();
  if (!geom.index) {
    // toNonIndexed が使えない場合のフォールバック
    return [geometry];
  }

  const idxArray   = Array.from(geom.index.array);
  const totalFaces = idxArray.length / 3;
  n = Math.max(1, Math.min(n, totalFaces)); // 面数より多くは分割できない
  const facesPerChunk = Math.ceil(totalFaces / n);

  const chunks = [];
  for (let ci = 0; ci < n; ci++) {
    const fStart = ci * facesPerChunk;
    const fEnd   = Math.min(fStart + facesPerChunk, totalFaces);
    if (fStart >= totalFaces) break;

    const subIdx = idxArray.slice(fStart * 3, fEnd * 3);

    // 使用頂点の抽出・再マッピング
    const uniqueVerts = [...new Set(subIdx)].sort((a, b) => a - b);
    const vmap        = new Map(uniqueVerts.map((vi, ni) => [vi, ni]));

    const chunkGeom = new THREE.BufferGeometry();

    // 各 attribute をコピー
    for (const [name, attr] of Object.entries(geom.attributes)) {
      const is  = attr.itemSize;
      const src = attr.array;
      const dst = new (src.constructor)(uniqueVerts.length * is);
      uniqueVerts.forEach((vi, ni) => {
        for (let j = 0; j < is; j++) dst[ni * is + j] = src[vi * is + j];
      });
      chunkGeom.setAttribute(name, new THREE.BufferAttribute(dst, is, attr.normalized));
    }

    // 新しいインデックス
    const newIdx = new Uint32Array(subIdx.length);
    subIdx.forEach((vi, i) => { newIdx[i] = vmap.get(vi); });
    chunkGeom.setIndex(new THREE.BufferAttribute(newIdx, 1));

    chunks.push(chunkGeom);
  }
  return chunks;
}

/** BufferGeometry のバウンディングボックス */
function boundsFromGeometry(geom) {
  geom.computeBoundingBox();
  const b = geom.boundingBox;
  return {
    min: [b.min.x, b.min.y, b.min.z],
    max: [b.max.x, b.max.y, b.max.z],
  };
}

/** Geometry + Material → GLB ArrayBuffer */
function exportGeomToGLB(geometry, material) {
  return new Promise((resolve, reject) => {
    const mesh  = new THREE.Mesh(geometry, material);
    const scene = new THREE.Scene();
    scene.add(mesh);
    new GLTFExporter().parse(scene, glb => resolve(glb), reject, { binary: true });
  });
}

// ── 変換実行 ──────────────────────────────────────────────────────────
$('convertBtn').addEventListener('click', async () => {
  if (!loadedGLB) { alert('GLBファイルを選択してください。'); return; }

  const btn = $('convertBtn');
  btn.disabled = true;
  setProgress('GLBを読み込み中…');

  try {
    const params = {
      lon:            parseFloat($('lon').value),
      lat:            parseFloat($('lat').value),
      height:         parseFloat($('height').value)        || 0,
      heading:        parseFloat($('heading').value)       || 0,
      pitch:          parseFloat($('pitch').value)         || 0,
      roll:           parseFloat($('roll').value)          || 0,
      chunks:         Math.max(1, parseInt($('chunks').value) || 3),
      geometricError: parseFloat($('geometricError').value) || 200,
      refine:         $('refine').value,
      contentFormat:  'glb',
    };

    setProgress('ジオメトリをマージ中…');
    const { geometry, material } = await loadAndMerge(loadedGLB);

    setProgress(`${params.chunks} チャンクに分割中…`);
    const geomChunks = splitGeometryByFaces(geometry, params.chunks);

    const zip        = new JSZip();
    const chunkInfos = [];

    for (let i = 0; i < geomChunks.length; i++) {
      setProgress(`チャンク ${i + 1} / ${geomChunks.length} を書き出し中…`);
      const geom   = geomChunks[i];
      const bounds = boundsFromGeometry(geom);
      const glb    = await exportGeomToGLB(geom, material);
      const fname  = `chunk_${String(i).padStart(3, '0')}.glb`;
      zip.folder('tiles_src').file(fname, glb);
      chunkInfos.push({
        index:     i,
        meshCount: 1,
        bounds,
        uri: `tiles_src/${fname}`,
      });
    }

    setProgress('tileset.json を生成中…');
    const tileset = buildTileset(chunkInfos, params);
    zip.file('tileset.json', JSON.stringify(tileset, null, 2));
    zip.file('build_report.json', JSON.stringify(buildReport(loadedFileName, chunkInfos, params), null, 2));

    setProgress('ZIP を生成中…');
    const blob    = await zip.generateAsync({ type: 'blob' });
    const a       = document.createElement('a');
    const base    = loadedFileName.replace(/\.glb$/i, '');
    a.href        = URL.createObjectURL(blob);
    a.download    = `${base}_3dtiles.zip`;
    a.click();

    const totalFaces = geomChunks.reduce((s, g) => s + (g.index ? g.index.count / 3 : g.attributes.position.count / 3), 0);
    setProgress(`完了: ${geomChunks.length} チャンク / 合計 ${Math.round(totalFaces).toLocaleString()} 面`, true);
  } catch (err) {
    console.error(err);
    setProgress(`エラー: ${err.message}`, false, true);
  } finally {
    btn.disabled = false;
  }
});

function setProgress(msg, ok = false, err = false) {
  const el = $('progressMsg');
  el.textContent = msg;
  el.className = 'status' + (ok ? ' ok' : err ? ' err' : '');
}

// ── 初期化 ────────────────────────────────────────────────────────────
initPreview();
initMap();
setTab('tool');
