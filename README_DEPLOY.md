# デプロイ手順（Cloudflare Pages + Render Free）

## 事前準備
- GitHubにこのリポジトリをプッシュ
- Cloudflareアカウント作成、Renderアカウント作成

## API（Render）
1) Render の Dashboard → New → **Blueprint** → この repo の `fastAPI/render.yaml` を選択
2) 環境変数を設定
   - `GEMINI_API_KEY`：GeminiのAPIキー（必須）
   - `GEMINI_MODEL`：gemini-1.5-flash（既定）
3) Deploy をクリック → 完了後、Render が API の URL を発行
4) フロント用に `react/.env` を作成し `VITE_API_BASE=<RenderのURL>` を設定してコミット

## フロント（Cloudflare Pages）
### 方法A：GitHub Actions（このリポのワークフローを使用）
1) Cloudflare Dashboard → My Profile → API Tokens → **Pages:Edit** 権限のトークンを作成
2) この GitHub リポの Settings → Secrets and variables → Actions に以下を登録
   - `CF_API_TOKEN`：上で作成したトークン
   - `CF_ACCOUNT_ID`：Cloudflare のアカウントID
3) `react/.env` の `VITE_API_BASE` を Render のURLに設定し `main` に push
4) Actions で `Deploy to Cloudflare Pages` が走り、`react/dist` が公開される

### 方法B：Cloudflare Pages のGit連携（UI）
1) Pages → Create Project → GitHub リポを選択
2) Build command: `npm ci && npm run build`  / Output dir: `react/dist`
3) Deploy
※この場合 Actions は不要

## 開発/本番のAPI呼び分け
- 開発（Vite）：`vite.config.ts` の proxy で `/fa` → 8000
- 本番（Pages）：`VITE_API_BASE` を使って `fetch(\`\${import.meta.env.VITE_API_BASE}/fa/... \`)`

## CORS（FastAPI）
FastAPI 側で本番ドメインを許可：
```py
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
  CORSMiddleware,
  allow_origins=["https://<pages-domain>.pages.dev","https://<your-domain>"],
  allow_methods=["*"], allow_headers=["*"],
)
```
