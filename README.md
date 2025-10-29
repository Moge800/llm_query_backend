# RAG Query Backend - クリーンアーキテクチャ版

このプロジェクトは、FAISS検索エンジンとOllama LLMを使用したRAG（Retrieval-Augmented Generation）システムをクリーンアーキテクチャで実装したものです。

## アーキテクチャ概要

```
src/
├── domain/              # ドメイン層 - ビジネスロジックの中核
│   ├── entities.py      # ドメインエンティティ
│   ├── interfaces.py    # サービスインターフェース
│   ├── exceptions.py    # ドメイン例外
│   └── __init__.py
├── application/         # アプリケーション層 - ユースケース
│   ├── dto.py          # データ転送オブジェクト
│   ├── usecases.py     # ユースケース実装
│   └── __init__.py
├── infrastructure/     # インフラストラクチャ層 - 外部システム連携
│   ├── config.py       # 設定管理
│   ├── faiss_service.py # FAISS検索サービス実装
│   ├── ollama_service.py # Ollama LLMサービス実装
│   ├── template_service.py # プロンプトテンプレートサービス実装
│   └── __init__.py
├── presentation/       # プレゼンテーション層 - API エンドポイント
│   ├── controllers.py  # FastAPIコントローラー
│   ├── routes.py       # ルート定義
│   └── __init__.py
├── api/                # レガシー互換性（旧query_api.py）
│   └── query_api.py
├── di_container.py     # 依存性注入コンテナ
├── main.py            # メインアプリケーション
├── run.py             # 実行エントリーポイント
└── __init__.py
```

## クリーンアーキテクチャの特徴

### 1. 依存性の方向
- **外側から内側へ**: Presentation → Application → Domain
- **Domain層**は他の層に依存しない
- **Infrastructure層**はDomain層のインターフェースを実装

### 2. 層の責務

#### Domain層
- ビジネスルール・エンティティ・ドメインサービスのインターフェース
- 他の層に依存しない、フレームワークに依存しない
- `SearchResult`, `Query`, `RagResult`エンティティ
- `IKnowledgeSearchService`, `ILlmService`等のインターフェース

#### Application層
- ユースケースの実装
- ドメインサービスを組み合わせてビジネスフローを実行
- `RagUseCase`, `HealthCheckUseCase`, `TemplateUseCase`

#### Infrastructure層
- 外部システム（FAISS、Ollama）との連携
- ドメイン層のインターフェースを実装
- 設定管理・環境変数処理

#### Presentation層
- FastAPI エンドポイント・HTTPリクエスト/レスポンス処理
- DTOとドメインエンティティの変換

### 3. 依存性注入
- `DIContainer`でサービス、ユースケース、コントローラーの依存関係を管理
- テスト時にモックオブジェクトの注入が容易

## 主な利点

1. **テスタビリティ**: 各層が独立しており、モックを使ったテストが容易
2. **保守性**: 関心の分離により、変更の影響範囲を局所化
3. **拡張性**: 新機能追加時も既存コードへの影響を最小化
4. **フレームワーク独立性**: FastAPIを他のフレームワークに置き換えることが可能

## 起動方法

### 1. 新しいクリーンアーキテクチャ版（推奨）
```bash
python -m src.run
# または
python -m src.main
```

### 2. レガシー互換性（旧版）
```bash
python src/api/query_api.py
```

## 環境変数

`.env`ファイルで以下の環境変数を設定：

```env
# Server設定
HOST=localhost
PORT=8010
LOG_LEVEL=INFO

# FAISS設定
FAISS_HOST=localhost
FAISS_PORT=8000

# Ollama設定
OLLAMA_HOST=localhost
OLLAMA_PORT=11434
OLLAMA_LLM_MODEL=your-model-name

# Template設定
PROMPT_TEMPLATE_PATH=prompt_template.txt
```

## APIエンドポイント

### メインエンドポイント
- `POST /query` - RAGクエリ実行
- `POST /debug/search` - デバッグ用検索
- `GET /health` - ヘルスチェック

### 管理エンドポイント
- `GET /prompt_template` - プロンプトテンプレート表示
- `POST /prompt_template/reload` - テンプレート再読み込み
- `POST /backend/reload` - バックエンド設定再読み込み
- `POST /reload_all` - 全体再読み込み

## 今後の拡張

1. **認証・認可**: セキュリティ層の追加
2. **キャッシュ**: レスポンスキャッシュの実装
3. **ロードバランサー**: 複数Ollamaインスタンスへの負荷分散
4. **モニタリング**: メトリクス収集・ヘルスチェック拡張
5. **バッチ処理**: 大量データ処理のための非同期処理