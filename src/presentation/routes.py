"""FastAPIルーター"""

from fastapi import FastAPI, Query as FastAPIQuery
from fastapi.responses import JSONResponse

from ..application import QueryRequestDto
from .controllers import RagController, HealthController, TemplateController


def create_routes(
    app: FastAPI,
    rag_controller: RagController,
    health_controller: HealthController,
    template_controller: TemplateController,
) -> None:
    """FastAPIルートを作成"""

    @app.post("/query", response_model=None)
    async def query_rag(request: QueryRequestDto):
        """ユーザークエリを受け取り、FAISS検索+Ollama応答を返す"""
        return await rag_controller.query(request)

    @app.post("/debug/search")
    async def simple_search(
        text: str = FastAPIQuery(..., description="検索文字列"),
        top_k: int = FastAPIQuery(5, description="faiss検索注入件数"),
    ) -> JSONResponse:
        """開発・デバッグ用のFAISS検索エンドポイント"""
        return await rag_controller.debug_search(text, top_k)

    @app.get("/health")
    async def health_check() -> JSONResponse:
        """ヘルスチェック"""
        return await health_controller.health_check()

    @app.get("/prompt_template")
    async def read_template() -> JSONResponse:
        """現在のプロンプトテンプレート表示"""
        return template_controller.get_template()

    @app.post("/prompt_template/reload")
    async def reload_prompt() -> JSONResponse:
        """プロンプトテンプレートの再読込"""
        return await template_controller.reload_template()
