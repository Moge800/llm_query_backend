"""メインアプリケーション"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from .di_container import DIContainer
from .infrastructure import Config
from .presentation import create_routes


# グローバルコンテナ
container = DIContainer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """アプリケーションライフサイクル管理"""
    try:
        # 設定読み込み
        config = Config.from_env()

        # ログ設定
        logging.basicConfig(level=getattr(logging, config.log_level))
        logger = logging.getLogger(__name__)

        # DIコンテナ初期化
        container.initialize(config)

        # 全サービスのヘルスチェック
        if not await container.health_check_all_services():
            raise RuntimeError("一部のサービスに接続できませんでした")

        logger.info("アプリケーションが正常に起動しました")
        yield

        # 終了処理（必要に応じて）
        logger.info("アプリケーションを終了します")

    except Exception as e:
        logging.error(f"アプリケーション起動失敗: {e}")
        raise


def create_app() -> FastAPI:
    """FastAPIアプリケーションを作成"""
    app = FastAPI(
        title="RAG Query API",
        description="FAISS + OllamaによるRAG構成（クリーンアーキテクチャ版）",
        version="2.0.0",
        lifespan=lifespan,
    )

    # ルート設定
    create_routes(
        app=app,
        rag_controller=container.get_rag_controller(),
        health_controller=container.get_health_controller(),
        template_controller=container.get_template_controller(),
    )

    # バックエンド再読み込みエンドポイント
    @app.post("/backend/reload")
    async def reload_backends() -> JSONResponse:
        """バックエンド設定を再読み込み"""
        try:
            await container.reload_configuration()

            # ヘルスチェック
            if not await container.health_check_all_services():
                raise RuntimeError("ヘルスチェックに失敗しました")

            # ヘルスステータスを返却
            health_status = await container.get_health_usecase().execute()
            return JSONResponse(content=health_status)

        except Exception as e:
            return JSONResponse(
                status_code=500, content={"status": "error", "message": str(e)}
            )

    @app.post("/reload_all")
    async def reload_all() -> JSONResponse:
        """全体を再読み込み"""
        try:
            # バックエンド再読み込み
            await container.reload_configuration()

            # テンプレート再読み込み
            await container.get_template_usecase().reload_template()

            # ヘルスチェック
            health_status = await container.get_health_usecase().execute()
            return JSONResponse(content=health_status)

        except Exception as e:
            return JSONResponse(
                status_code=500, content={"status": "error", "message": str(e)}
            )

    return app


# アプリケーションインスタンス
app = create_app()
