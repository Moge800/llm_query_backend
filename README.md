# llm_query_backend

RAG (Retrieval-Augmented Generation) システムのバックエンドAPI。FAISSベクトルデータベースとOllama LLMを統合し、効率的なクエリ応答を実現します。

## 技術スタック

- **Python**: 3.8+
- **パッケージマネージャ**: uv
- **Webフレームワーク**: FastAPI 0.120.4
- **データ検証**: Pydantic 2.12.3
- **ASGIサーバー**: Uvicorn 0.38.0
- **HTTPクライアント**: httpx 0.28.1

## プロジェクト構造

```
llm_query_backend/
├── src/
│   └── api/
│       └── query_api.py    # FastAPI エンドポイント
├── tests/
│   └── api/
│       └── test_query_api.py  # APIテスト
├── .env                    # 環境変数 (gitignore済み)
├── .env.example            # 環境変数テンプレート
├── pyproject.toml          # プロジェクト設定
├── requirements.txt        # 依存関係
└── LICENSE                 # MITライセンス
```

## セットアップ

### 1. リポジトリのクローン

```bash
git clone https://github.com/Moge800/llm_query_backend.git
cd llm_query_backend
```

### 2. 仮想環境の作成と有効化

#### uvを使用する場合（推奨）

```bash
# uv仮想環境の作成
uv venv

# 仮想環境の有効化 (Windows PowerShell)
.venv\Scripts\Activate.ps1

# 仮想環境の有効化 (Linux/Mac)
source .venv/bin/activate
```

#### venvを使用する場合

```bash
# Python標準venv環境の作成
python -m venv .venv

# 仮想環境の有効化 (Windows PowerShell)
.venv\Scripts\Activate.ps1

# 仮想環境の有効化 (Linux/Mac)
source .venv/bin/activate
```

### 3. 依存関係のインストール

#### uvを使用する場合

```bash
uv sync
```

#### pipを使用する場合

```bash
pip install -r requirements.txt
```

### 4. 環境変数の設定

```bash
# .env.example を .env にコピー
cp .env.example .env

# .envファイルを編集して適切な値を設定
# 例:
# FAISS_HOST=localhost
# FAISS_PORT=8000
# OLLAMA_HOST=localhost
# OLLAMA_PORT=11434
# OLLAMA_LLM_MODEL=qwen2.5:14b
```

### 5. プロンプトテンプレートの作成

プロジェクトルートに`prompt_template.txt`を作成してください。

```txt
以下の参考情報を使用して質問に答えてください。

【参考情報】
$references

【質問】
$query

【回答】
```

## 実行方法

### 開発サーバーの起動

```bash
# uvicornで直接起動
uvicorn src.api.query_api:app --reload --host localhost --port 8010

# またはスクリプトから起動
python -m src.api.query_api
```

### APIドキュメントの確認

ブラウザで以下のURLにアクセス:

- Swagger UI: http://localhost:8010/docs
- ReDoc: http://localhost:8010/redoc

## API エンドポイント

### POST /query

ユーザークエリを受け取り、FAISS検索+Ollama応答を返します。

**リクエスト:**

```json
{
  "query": "質問文",
  "top_k": 5
}
```

**レスポンス:**

```json
{
  "answer": "LLMからの回答",
  "references": [
    {
      "text": "参考情報",
      "similarity_score": 0.95
    }
  ]
}
```

### GET /health

バックエンドサービス（FAISS、Ollama）のヘルスチェック。

**レスポンス:**

```json
{
  "status": "ok",
  "faiss": true,
  "ollama": true,
  "faiss_base": "http://localhost:8000",
  "ollama_base": "http://localhost:11434",
  "model": "qwen2.5:14b"
}
```

### その他のエンドポイント

- `POST /debug/search` - デバッグ用FAISS検索
- `GET /prompt_template` - プロンプトテンプレート表示
- `POST /prompt_template/reload` - プロンプトテンプレート再読込
- `POST /backend/reload` - バックエンド設定再読込
- `POST /reload_all` - 全設定再読込

## テスト

### テストの実行

```bash
# 全テスト実行
pytest tests/ -v

# カバレッジ計測
pytest --cov=src tests/
```

### 開発パッケージのインストール

```bash
# uvの場合
uv sync --extra dev

# pipの場合
pip install -e ".[dev]"
```

## 環境変数

| 変数名 | 説明 | デフォルト値 |
|--------|------|-------------|
| `FAISS_HOST` | FAISSバックエンドのホスト | `localhost` |
| `FAISS_PORT` | FAISSバックエンドのポート | `8000` |
| `OLLAMA_HOST` | Ollamaのホスト | `localhost` |
| `OLLAMA_PORT` | Ollamaのポート | `11434` |
| `OLLAMA_LLM_MODEL` | 使用するLLMモデル名 | `qwen2.5:14b` |
| `HOST` | APIサーバーのホスト | `localhost` |
| `PORT` | APIサーバーのポート | `8010` |
| `LOG_LEVEL` | ログレベル | `INFO` |
| `PROMPT_TEMPLATE_PATH` | プロンプトテンプレートのパス | `prompt_template.txt` |

## 開発

### コーディング規約

詳細は [`.github/copilot-instructions.md`](.github/copilot-instructions.md) を参照してください。

主な規約:
- 型ヒントは必須
- 非同期処理を推奨
- Pydanticモデルでデータ検証
- 具体的な例外処理（汎用`Exception`を避ける）

### コード品質

```bash
# 型チェック
mypy src/

# テスト実行
pytest tests/ -v
```

## ライセンス

このプロジェクトはMITライセンスで公開されています。詳細は [LICENSE](LICENSE) ファイルを参照してください。

## 貢献

質問や改善提案は歓迎します！Issueまたはプルリクエストをお送りください。
