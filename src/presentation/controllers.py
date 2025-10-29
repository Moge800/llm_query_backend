"""FastAPIコントローラー"""

import logging
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from typing import Any

from ..application import (
    RagUseCase,
    HealthCheckUseCase,
    TemplateUseCase,
    QueryRequestDto,
    QueryResponseDto,
)
from ..domain import (
    ValidationError,
    ServiceUnavailableError,
    SearchError,
    LlmGenerationError,
    TemplateError,
)

logger = logging.getLogger(__name__)


class RagController:
    """RAGコントローラー"""

    def __init__(self, rag_usecase: RagUseCase):
        self._rag_usecase = rag_usecase

    async def query(self, request: QueryRequestDto) -> QueryResponseDto:
        """ユーザークエリを受け取り、RAG応答を返す"""
        try:
            if request.query.strip() == "":
                raise HTTPException(status_code=400, detail="queryが空欄です。")

            return await self._rag_usecase.execute(request)

        except ValidationError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except (ServiceUnavailableError, SearchError, LlmGenerationError) as e:
            raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            logger.error(f"RAGクエリ処理エラー: {e}")
            raise HTTPException(
                status_code=500, detail="内部サーバーエラーが発生しました"
            )

    async def debug_search(self, text: str, top_k: int) -> Any:
        """開発・デバッグ用のFAISS検索エンドポイント"""
        try:
            request = QueryRequestDto(query=text, top_k=top_k)
            result = await self._rag_usecase.execute(request)

            # JSON変換を試行
            try:
                import json

                answer = (
                    json.loads(result.answer)
                    if isinstance(result.answer, str)
                    else result.answer
                )
                return JSONResponse(content=answer)
            except (json.JSONDecodeError, TypeError):
                return result.answer

        except ValidationError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"デバッグ検索エラー: {e}")
            raise HTTPException(
                status_code=500, detail="検索処理でエラーが発生しました"
            )


class HealthController:
    """ヘルスチェックコントローラー"""

    def __init__(self, health_usecase: HealthCheckUseCase):
        self._health_usecase = health_usecase

    async def health_check(self) -> JSONResponse:
        """ヘルスチェック"""
        try:
            health_status = await self._health_usecase.execute()
            status_code = 200 if health_status["status"] == "ok" else 503
            return JSONResponse(status_code=status_code, content=health_status)
        except Exception as e:
            logger.error(f"ヘルスチェックエラー: {e}")
            return JSONResponse(
                status_code=503, content={"status": "error", "message": str(e)}
            )


class TemplateController:
    """テンプレート管理コントローラー"""

    def __init__(self, template_usecase: TemplateUseCase):
        self._template_usecase = template_usecase

    def get_template(self) -> JSONResponse:
        """現在のプロンプトテンプレート表示"""
        try:
            template = self._template_usecase.get_template()
            return JSONResponse(content={"template": template})
        except Exception as e:
            logger.error(f"テンプレート取得エラー: {e}")
            raise HTTPException(
                status_code=500, detail="テンプレート取得に失敗しました"
            )

    async def reload_template(self) -> JSONResponse:
        """プロンプトテンプレートの再読込"""
        try:
            await self._template_usecase.reload_template()
            return JSONResponse(content={"status": "reloaded"})
        except TemplateError as e:
            raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            logger.error(f"テンプレート再読み込みエラー: {e}")
            raise HTTPException(
                status_code=500, detail="テンプレート再読み込みに失敗しました"
            )
