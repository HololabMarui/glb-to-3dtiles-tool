/**
 * app.js
 * GLB → 3D Tiles 変換ツール メインロジック
 * - GLB 読み込み (Three.js GLTFLoader)
 * - メッシュ単位チャンク分割 & GLB 書き出し (Three.js GLTFExporter)
 * - MapLibre GL JS による位置調整地図
 * - ZIP 出力 (JSZip)
 */
import { buildTileset, buildReport } from './georef.js';

// ── Three.js import (ESM CDN) ──────────────────────────────────────────
import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.165.0/build/three.module.js';
import { GLTFLoader }   from 'https://cdn.jsdelivr.net/npm/three@0.165.0/examples/jsm/loaders/GLTFLoader.js';
import { GLTFExporter } from 'https://cdn.jsdelivr.net/npm/three@0.165.0/examples/jsm/exporters/GLTFExporter.js';

// ── DOM helpers ────────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);

// ── State ──────────────────────────────────────────────────────────────
let loadedGLB = null;     // ArrayBuffer of original GLB
let loadedFileName = '';
let marker = null;        // MapLibre marker
let map = null;

// ── Tab switching ──────────────────────────────────────────────────────
function setTab(name) {
  ['guide', 'tool'].forEach(t => {
    $('tab-' + t).classList.toggle('active', t === name);
    $('page-' + t).style.display = t === name ? '' : 'none';
  });
}
$('tab-guide').addEventListener('click', () => setTab('guide'));
$('tab-tool').addEventListener('click',  () => setTab('tool'));

// ── GLB file input ─────────────────────────────────────────────────────
$('glbInput').addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  loadedFileName = file.name;
  loadedGLB = await file.arrayBuffer();
  $('glbStatus').textContent = `✓ ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
  $('glbStatus').className = 'status ok';
  updatePreview();
});

// ── 3D Preview (Three.js) ──────────────────────────────────────────────
let previewRenderer = null;
let previewScene = null;
let previewCamera = null;
let previewAnimId = null;

function initPreview() {
  const canvas = $('previewCanvas');
  previewRenderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  previewRenderer.setPixelRatio(devicePixelRatio);
  previewScene  = new THREE.Scene();
  previewCamera = new THREE.PerspectiveCamera(45, 1, 0.01, 10000);

  // ライト
  previewScene.add(new THREE.AmbientLight(0xffffff, 0.8));
  const dir = new THREE.DirectionalLight(0xffffff, 1.0);
  dir.position.set(5, 10, 7);
  previewScene.add(dir);

  // リサイズ
  function resize() {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    previewCamera.aspect = w / h;
    previewCamera.updateProjectionMatrix();
    previewRenderer.setSize(w, h, false);
  }
  new ResizeObserver(resize).observe(canvas);
  resize();

  // マウスドラッグ回転
  let drag = false, lastX = 0, lastY = 0;
  let rotX = 0, rotY = 0;
  const pivot = new THREE.Object3D();
  previewScene.add(pivot);

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
    previewCamera.position.z *= 1 + e.deltaY * 0.001;
  }, { passive: true });

  window._previewPivot = pivot;

  function animate() {
    previewAnimId = requestAnimationFrame(animate);
    previewRenderer.render(previewScene, previewCamera);
  }
  animate();
}

function updatePreview() {
  if (!loadedGLB) return;
  const pivot = window._previewPivot;
  // 既存モデル削除
  while (pivot.children.length) pivot.remove(pivot.children[0]);

  const loader = new GLTFLoader();
  loader.parse(loadedGLB.slice(0), '', (gltf) => {
    // バウンディングボックスを計算してカメラ調整
    const box = new THREE.Box3().setFromObject(gltf.scene);
    const center = box.getCenter(new THREE.Vector3());
    const size   = box.getSize(new THREE.Vector3()).length();

    gltf.scene.position.sub(center);
    pivot.add(gltf.scene);

    previewCamera.position.set(0, 0, size * 1.2);
    previewCamera.near = size * 0.001;
    previewCamera.far  = size * 100;
    previewCamera.updateProjectionMatrix();
  }, (err) => {
    console.warn('Preview error:', err);
  });
}

// ── MapLibre GL JS 地図 ────────────────────────────────────────────────
function initMap() {
  map = new maplibregl.Map({
    container: 'map',
    style: 'https://tile.openstreetmap.jp/styles/osm-bright/style.json',
    center: [parseFloat($('lon').value), parseFloat($('lat').value)],
    zoom: 14,
  });

  map.on('load', () => {
    placeMarker();
  });

  // マップクリックで位置をセット
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
      .setLngLat([lon, lat])
      .addTo(map);

    marker.on('dragend', () => {
      const lngLat = marker.getLngLat();
      $('lon').value = lngLat.lng.toFixed(7);
      $('lat').value = lngLat.lat.toFixed(7);
    });
  }

  map.flyTo({ center: [lon, lat], zoom: Math.max(map.getZoom(), 13) });
}

// 入力フォームの変化で地図マーカーを同期
['lon', 'lat'].forEach(id => {
  $(id).addEventListener('change', placeMarker);
});

// ── GLB メッシュ解析 & チャンク分割 ───────────────────────────────────
/**
 * GLB ArrayBuffer を Three.js で読み込み、メッシュをフラット化して返す
 * 各メッシュにワールド変換を適用済み
 */
function loadMeshes(arrayBuffer) {
  return new Promise((resolve, reject) => {
    const loader = new GLTFLoader();
    loader.parse(arrayBuffer.slice(0), '', (gltf) => {
      gltf.scene.updateWorldMatrix(true, true);
      const meshes = [];
      gltf.scene.traverse(obj => {
        if (!obj.isMesh) return;
        // ワールド変換をジオメトリに焼き込む
        const geom = obj.geometry.clone();
        geom.applyMatrix4(obj.matrixWorld);
        const mat  = obj.material.clone ? obj.material.clone() : obj.material;
        const flat = new THREE.Mesh(geom, mat);
        meshes.push(flat);
      });
      if (meshes.length === 0) reject(new Error('メッシュが見つかりません'));
      else resolve(meshes);
    }, reject);
  });
}

/** メッシュ配列をラウンドロビンで N チャンクに分割 */
function splitEvenly(meshes, n) {
  n = Math.max(1, Math.min(n, meshes.length));
  const groups = Array.from({ length: n }, () => []);
  meshes.forEach((m, i) => groups[i % n].push(m));
  return groups.filter(g => g.length > 0);
}

/** Three.js Mesh 配列のバウンディングボックス */
function computeBounds(meshes) {
  const box = new THREE.Box3();
  meshes.forEach(m => box.expandByObject(m));
  return {
    min: [box.min.x, box.min.y, box.min.z],
    max: [box.max.x, box.max.y, box.max.z],
  };
}

/** Mesh 配列 → GLB ArrayBuffer (GLTFExporter) */
function exportGroupToGLB(meshes) {
  return new Promise((resolve, reject) => {
    const scene = new THREE.Scene();
    meshes.forEach((m, i) => {
      const node = m.clone();
      node.name = `mesh_${i}`;
      scene.add(node);
    });
    const exporter = new GLTFExporter();
    exporter.parse(scene, (glb) => resolve(glb), reject, { binary: true });
  });
}

// ── 変換実行 ──────────────────────────────────────────────────────────
$('convertBtn').addEventListener('click', async () => {
  if (!loadedGLB) {
    alert('GLBファイルを選択してください。');
    return;
  }

  const btn = $('convertBtn');
  btn.disabled = true;
  setProgress('GLBを読み込み中…');

  try {
    // パラメータ収集
    const params = {
      lon:           parseFloat($('lon').value),
      lat:           parseFloat($('lat').value),
      height:        parseFloat($('height').value) || 0,
      heading:       parseFloat($('heading').value) || 0,
      pitch:         parseFloat($('pitch').value)   || 0,
      roll:          parseFloat($('roll').value)    || 0,
      chunks:        Math.max(1, parseInt($('chunks').value) || 3),
      geometricError: parseFloat($('geometricError').value) || 200,
      refine:        $('refine').value,
      contentFormat: 'glb',
    };

    setProgress(`メッシュを解析中…`);
    const meshes = await loadMeshes(loadedGLB);
    const groups = splitEvenly(meshes, params.chunks);

    setProgress(`${groups.length} チャンクに分割中…`);

    const zip = new JSZip();
    const chunkInfos = [];

    for (let i = 0; i < groups.length; i++) {
      setProgress(`チャンク ${i + 1} / ${groups.length} を書き出し中…`);
      const group  = groups[i];
      const bounds = computeBounds(group);
      const glb    = await exportGroupToGLB(group);
      const fname  = `chunk_${String(i).padStart(3, '0')}.glb`;
      zip.folder('tiles_src').file(fname, glb);
      chunkInfos.push({
        index:     i,
        meshCount: group.length,
        bounds,
        uri:       `tiles_src/${fname}`,
      });
    }

    setProgress('tileset.json を生成中…');
    const tileset = buildTileset(chunkInfos, params);
    zip.file('tileset.json', JSON.stringify(tileset, null, 2));

    const report = buildReport(loadedFileName, chunkInfos, params);
    zip.file('build_report.json', JSON.stringify(report, null, 2));

    // プレビュー表示
    $('tilesetPreview').textContent = JSON.stringify(tileset, null, 2);

    setProgress('ZIP を生成中…');
    const blob = await zip.generateAsync({ type: 'blob' });

    // ダウンロード
    const a = document.createElement('a');
    const baseName = loadedFileName.replace(/\.glb$/i, '');
    a.href = URL.createObjectURL(blob);
    a.download = `${baseName}_3dtiles.zip`;
    a.click();

    setProgress(`完了: ${groups.length} チャンク / tileset.json + build_report.json`, true);
    $('downloadSection').style.display = '';
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
