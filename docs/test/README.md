# GLB → 3D Tiles 変換ツール

ブラウザ上で GLB ファイルをメッシュ単位に分割し、地理参照付き `tileset.json` と chunk GLB を ZIP で出力する Web ツールです。
**サーバー不要・GitHub Pages でそのまま動作します。**

---

## 機能

| 機能 | 説明 |
|------|------|
| GLB 読み込み | ファイル選択またはドラッグ&ドロップ |
| 3D プレビュー | Three.js による GLB ビューア（ドラッグ回転・ホイールズーム） |
| 地図表示 | MapLibre GL JS + OpenStreetMap による配置位置確認 |
| 位置調整 | 経度・緯度・高さ・Heading/Pitch/Roll の手動入力、地図マーカードラッグ |
| tileset.json 生成 | Python 版と同一アルゴリズムで `root.transform` を計算 |
| ZIP 出力 | `tiles_src/*.glb` + `tileset.json` + `build_report.json` を ZIP 化 |

---

## ファイル構成

```
glb-to-3dtiles/
  index.html       メインページ（タブ UI）
  js/
    georef.js      座標変換・tileset.json 生成ロジック（Python 移植）
    app.js         UI・GLB 処理・地図・ZIP 出力
  README.md
```

---

## ローカルでの確認方法

1. このフォルダをそのまま HTTP サーバーで配信する（`file://` では ESM モジュールが動作しません）

   ```bash
   # Python 3
   cd glb-to-3dtiles
   python3 -m http.server 8765
   ```

2. ブラウザで `http://localhost:8765/` を開く

---

## GitHub Pages への公開手順

1. リポジトリのルートまたは `docs/` フォルダに `glb-to-3dtiles/` の中身を置く
2. Settings → Pages → Source を設定
3. `https://<user>.github.io/<repo>/` でアクセス

---

## 出力 ZIP の構成

```
output.zip
  tiles_src/
    chunk_000.glb
    chunk_001.glb
    ...
  tileset.json
  build_report.json
```

`tileset.json` の `content.uri` は `tiles_src/chunk_NNN.glb` を参照します。

---

## B3DM 化について（後処理）

今回のツールは GLB chunk までを担当します。B3DM が必要な場合は以下を別途実行してください。

```bash
npx 3d-tiles-tools glbToB3dm -i tiles_src/chunk_000.glb -o tiles/chunk_000.b3dm
npx 3d-tiles-tools updateAlignment -i tiles/chunk_000.b3dm -o tiles/chunk_000_aligned.b3dm
```

---

## 注意事項

- **位置・向きはすべて手動入力値です。自動推定は行いません。**
- チャンク分割はラウンドロビン方式（Python 版と同一）です。
- 大きな GLB はブラウザのメモリを消費します。
- `three.module.js` 等のライブラリは CDN から読み込むためインターネット接続が必要です。

---

## 依存ライブラリ（すべて CDN）

| ライブラリ | 用途 |
|-----------|------|
| [Three.js](https://threejs.org/) v0.165 | GLB 読み込み・プレビュー・チャンク書き出し |
| [MapLibre GL JS](https://maplibre.org/) v4.5 | OSM 地図・マーカー操作 |
| [JSZip](https://stuk.github.io/jszip/) v3.10 | クライアント側 ZIP 生成 |
