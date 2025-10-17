# EduRun Backend

FastAPI で実装した API サーバーです。SQLite (`users.db`) を利用し、動画解析や統計処理のロジックを提供します。

## セットアップ

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.main:app --reload
```

### 環境変数
| 変数名 | 説明 | 既定値 |
|--------|------|--------|
| `APP_SECRET_KEY` | JWT 署名キー | `CHANGE_ME` |
| `CORS_ALLOW_ORIGINS` | `,` 区切りの許可オリジン | `*` |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | アクセストークン有効期間（分） | `720` |

## 主なモジュール
- `app/routers/auth.py` 認証関連エンドポイント
- `app/routers/data.py` CSV アップロードと統計関数
- `app/routers/analysis.py` 重回帰・因子分析
- `app/routers/chat.py` 課題チャット
- `app/routers/admin.py` 管理機能
- `app/routers/pose.py` 骨格推定（Mediapipe + DTW）

`/docs` または `/redoc` から API を確認できます。
