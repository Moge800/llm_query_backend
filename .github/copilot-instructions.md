# GitHub Copilot Instructions

## プロジェクト概要
RAG (Retrieval-Augmented Generation) システムのバックエンドAPI。FAISSベクトルデータベースとOllama LLMを統合し、効率的なクエリ応答を実現する。

## 技術スタック
- **Python**: 3.8+
- **パッケージマネージャ**: uv
- **Webフレームワーク**: FastAPI 0.120.4
- **データ検証**: Pydantic 2.12.3
- **環境変数**: python-dotenv 1.2.1
- **ASGIサーバー**: Uvicorn 0.38.0
- **HTTPクライアント**: httpx 0.28.1
- **開発ツール**: mypy (型チェック)

## プロジェクト構造
```
llm_query_backend/
├── src/
│   ├── api/              # FastAPI エンドポイント
│   ├── core/             # コアロジック (RAG, LLM統合)
│   ├── models/           # Pydanticデータモデル
│   ├── services/         # ビジネスロジック層
│   └── utils/            # ユーティリティ関数
├── tests/                # テストコード
├── .env                  # 環境変数 (gitignore済み)
├── .env.example          # 環境変数テンプレート
├── requirements.txt      # 依存関係
├── pyproject.toml        # プロジェクト設定
└── LICENSE               # MITライセンス
```

## コーディング規約

### 1. 型ヒントは必須
```python
# Good
async def query_llm(prompt: str) -> str:
    return await llm_client.generate(prompt)

# Bad
async def query_llm(prompt):
    return await llm_client.generate(prompt)
```

### 2. 環境変数の扱い
- `.env`ファイルは必須(`.env.example`をコピー)
- Pydantic Settingsで型安全に管理
- 機密情報(API Keys, DB接続情報)は環境変数化

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    ollama_base_url: str
    faiss_index_path: str
    
    class Config:
        env_file = ".env"
```

### 3. エラーハンドリング
- `Exception`の汎用捕捉は避ける
- FastAPI例外クラスを活用
```python
# Good
from fastapi import HTTPException

if not query:
    raise HTTPException(status_code=400, detail="Query is required")

# Bad
except Exception as e:
    pass
```

### 4. 非同期処理の推奨
- FastAPIエンドポイントは`async def`を使用
- I/O処理(HTTP, DB)は非同期で実装

```python
# Good
@app.post("/query")
async def query_endpoint(request: QueryRequest) -> QueryResponse:
    result = await llm_service.process(request.query)
    return QueryResponse(answer=result)

# Bad
@app.post("/query")
def query_endpoint(request: QueryRequest) -> QueryResponse:
    result = llm_service.process(request.query)  # 同期処理
    return QueryResponse(answer=result)
```

### 5. インポート順序
```python
# 標準ライブラリ
import os
from typing import List, Optional

# サードパーティ
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx

# ローカル
from models.query import QueryRequest, QueryResponse
from services.llm_service import LLMService
```

### 6. Pydanticモデル
- リクエスト/レスポンスは必ずモデル化
- バリデーションルールを明確に定義

```python
from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=20)
```

### 7. APIエンドポイント設計
- RESTful原則に従う
- HTTPステータスコードを適切に使用
- レスポンスモデルを明示

```python
@app.post(
    "/query",
    response_model=QueryResponse,
    status_code=200,
    tags=["Query"]
)
async def query(request: QueryRequest) -> QueryResponse:
    ...
```

### 8. ロギング
- Python標準loggingモジュールを使用
- ログレベル: DEBUG, INFO, WARNING, ERROR, CRITICAL

```python
import logging

logger = logging.getLogger(__name__)
logger.info("Processing query: %s", query)
logger.error("Failed to connect to Ollama: %s", e)
```

### 9. dotenvの読み込み
```python
# Good
from dotenv import load_dotenv
load_dotenv()

# Bad
import dotenv
dotenv.load_dotenv()
```

### 10. type: ignoreは最小限に
- mypyの設定で`ignore_missing_imports = true`を活用
- やむを得ない場合のみ使用し、理由をコメント

### 11. ファイルエンコーディング
- **すべてのファイル**: UTF-8 BOMなし
  - Python (`.py`)
  - Markdown (`.md`)
  - JSON (`.json`)
  - YAML (`.yml`)
  - テキストファイル (`.txt`, `.env`)

## FastAPI特有の考慮事項
- 依存性注入(Dependency Injection)を活用
- `lifespan`イベントでリソース管理
- CORS設定が必要な場合は明示的に設定

```python
from fastapi import Depends
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 起動時処理
    await initialize_resources()
    yield
    # 終了時処理
    await cleanup_resources()

app = FastAPI(lifespan=lifespan)
```

## セキュリティ
- `.env`はGitにコミットしない(`.gitignore`済み)
- APIキーや機密情報は環境変数化
- HTTPS使用を推奨(本番環境)
- レート制限の実装を検討

## テスト

### テスト駆動開発(TDD)の推奨
**新機能追加時は必ずテストも同時作成する**

#### テストの配置
```
tests/
├── api/              # APIエンドポイントのテスト
├── services/         # サービス層のテスト
├── models/           # Pydanticモデルのテスト
└── utils/            # ユーティリティのテスト
```

#### テスト作成ルール
1. **新しいエンドポイント追加** → 対応するテストを`tests/api/`に作成
2. **ビジネスロジック変更** → 既存テストを更新 + 新ケース追加
3. **バグ修正** → 再現テストを追加してから修正

#### FastAPIテストの例
```python
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_query_endpoint():
    response = client.post(
        "/query",
        json={"query": "Test query", "top_k": 5}
    )
    assert response.status_code == 200
    assert "answer" in response.json()
```

#### テスト実行コマンド
```bash
# 全テスト実行
pytest tests/ -v

# 特定のテストファイルのみ
pytest tests/api/test_query.py -v

# カバレッジ計測
pytest --cov=src tests/
```

#### テストの命名規則
- ファイル: `test_*.py`
- 関数: `test_*` (例: `test_query_endpoint`, `test_validation_error`)

## デプロイ
- `uv sync`で依存関係インストール
- `uvicorn main:app --reload`で開発サーバー起動
- 本番環境: `uvicorn main:app --host 0.0.0.0 --port 8000`

## よくある問題と解決策

### インポートエラー
- `PYTHONPATH`を正しく設定
- 絶対インポート(`from src.api import ...`)を使用

### .envが見つからない
- `.env.example`を`.env`にコピー
- `load_dotenv()`が呼ばれているか確認

### 非同期処理のエラー
- `async def`と`await`の使い分けを確認
- `asyncio.run()`は`main.py`のみで使用

## コード品質
- mypy: 型チェック(設定済み)
- 型ヒント必須
- Pylance: VSCode型チェック推奨

## 命名規則
- クラス: `PascalCase` (例: `QueryRequest`, `LLMService`)
- 関数/変数: `snake_case` (例: `process_query`, `top_k`)
- 定数: `UPPER_SNAKE_CASE` (例: `MAX_QUERY_LENGTH`, `DEFAULT_TOP_K`)
- プライベート: `_leading_underscore` (例: `_internal_method`)

## ドキュメント
- Docstring: Google Style
- 型ヒントで大部分は自己文書化
- 複雑なロジックにはインラインコメント
- FastAPIの自動生成ドキュメント(`/docs`)を活用

## 定期メンテナンス手順

### 大きな変更時・仕事終わりのチェックリスト

#### 1. 全スキャンによるテスト項目チェック
大きな機能追加や1日の開発終了時に、テストの過不足をチェック

**チェック対象**:
- [ ] 新規追加したエンドポイントにテストがあるか
- [ ] 修正したビジネスロジックのテストケースが十分か
- [ ] 新しいPydanticモデルにバリデーションテストがあるか
- [ ] エラーハンドリングのテストが網羅されているか

#### 2. コミット前の最終チェック
```bash
# 1. 全テスト実行
pytest tests/ -v

# 2. 型チェック
mypy src/

# 3. 変更差分確認
git status
git diff

# 4. コミット
git add .
git commit -m "feat: <変更内容の要約>"
git push origin main
```

### 推奨頻度
- **テストチェック**: 大きな機能追加時 or 1日の終わり
- **型チェック**: 毎コミット前
- **カバレッジ計測**: 週1回 or リリース前

---

**このプロジェクトはMITライセンスで公開されています。質問や改善提案は歓迎します！**