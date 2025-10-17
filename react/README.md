# EduRun Frontend

TypeScript + Vite + Tailwind を用いたモダン UI です。FastAPI バックエンドと連携し、以下の画面を提供します。

- ダッシュボード（データプレビュー・統計カード・相関行列）
- χ²検定 / t検定
- 重回帰分析
- 因子分析
- 課題チャット
- 骨格推定（生徒アカウントのみ）
- ユーザー管理（教員のみ）

## セットアップ

```bash
npm install
npm run dev
```

`.env` で `VITE_API_URL` を指定するとバックエンドの URL を切り替えられます（デフォルトは `http://localhost:8000`）。

## ビルド

```bash
npm run build
npm run preview
```

ビルド成果物は `dist/` に生成されます。
