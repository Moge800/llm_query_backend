"""アプリケーション層のユースケース"""

from typing import List
import json
import string
import re
import logging
from ..domain import (
    Query,
    SearchResult,
    IKnowledgeSearchService,
    ILlmService,
    IPromptTemplateService,
    ValidationError,
)
from .dto import QueryRequestDto, QueryResponseDto, SearchResultDto

logger = logging.getLogger(__name__)


class RagUseCase:
    """RAGユースケース"""

    def __init__(
        self,
        knowledge_service: IKnowledgeSearchService,
        llm_service: ILlmService,
        template_service: IPromptTemplateService,
    ):
        self._knowledge_service = knowledge_service
        self._llm_service = llm_service
        self._template_service = template_service

    async def execute(self, request: QueryRequestDto) -> QueryResponseDto:
        """RAGクエリを実行する"""
        try:
            # ドメインエンティティに変換
            query = Query(text=request.query, top_k=request.top_k)

            # 検索実行
            search_results = await self._knowledge_service.search(query)

            # プロンプト作成
            references_text = self._format_references(search_results)
            template = self._template_service.get_template()
            prompt = self._build_prompt(template, references_text, query.text)

            # LLM実行
            answer = await self._llm_service.generate(prompt)

            # 回答をクリーンアップ
            cleaned_answer = self._clean_answer(answer)

            # DTOに変換して返却
            return QueryResponseDto(
                answer=cleaned_answer,
                references=[self._to_search_result_dto(sr) for sr in search_results],
            )

        except ValueError as e:
            raise ValidationError(str(e))
        except Exception as e:
            logger.error(f"RAGクエリ実行エラー: {e}")
            raise

    def _format_references(self, references: List[SearchResult]) -> str:
        """リファレンスをJSON形式にフォーマット"""
        formatting_references = []
        for ref in references:
            ref_dict = {"content": ref.content, "metadata": ref.metadata}
            if ref.rank is not None:
                ref_dict["rank"] = ref.rank
            if ref.similarity_score is not None:
                ref_dict["similarity_score"] = ref.similarity_score
            formatting_references.append(ref_dict)

        return json.dumps(formatting_references, ensure_ascii=False, indent=2)

    def _build_prompt(self, template: str, references: str, query: str) -> str:
        """プロンプトを構築"""
        tmpl = string.Template(template)
        return tmpl.safe_substitute(references=references, query=query)

    def _clean_answer(self, answer: str) -> str:
        """回答をクリーンアップ"""
        # コメント除去（// と /* */ 両方対応）
        answer = re.sub(r"//.*", "", answer)
        answer = re.sub(r"/\*.*?\*/", "", answer, flags=re.DOTALL)
        # コードブロック除去（```json 以外にも対応）
        answer = re.sub(r"^```[a-z]*\n|\n```$", "", answer.strip())

        # JSON文字列で返ってくることを想定してパース
        try:
            parsed_answer = json.loads(answer)
            if isinstance(parsed_answer, dict):
                return json.dumps(parsed_answer, ensure_ascii=False, indent=2)
            return parsed_answer
        except json.JSONDecodeError:
            logger.warning(
                "Ollamaの応答がJSONではありませんでした。文字列として返します。"
            )
            return answer

    def _to_search_result_dto(self, sr: SearchResult) -> SearchResultDto:
        """SearchResultをSearchResultDtoに変換"""
        return SearchResultDto(
            content=sr.content,
            rank=sr.rank,
            similarity_score=sr.similarity_score,
            metadata=sr.metadata,
        )


class HealthCheckUseCase:
    """ヘルスチェックユースケース"""

    def __init__(
        self, knowledge_service: IKnowledgeSearchService, llm_service: ILlmService
    ):
        self._knowledge_service = knowledge_service
        self._llm_service = llm_service

    async def execute(self) -> dict:
        """ヘルスチェックを実行"""
        faiss_ok = await self._knowledge_service.health_check()
        ollama_ok = await self._llm_service.health_check()

        status = "ok" if faiss_ok and ollama_ok else "degraded"

        return {"status": status, "faiss": faiss_ok, "ollama": ollama_ok}


class TemplateUseCase:
    """テンプレート管理ユースケース"""

    def __init__(self, template_service: IPromptTemplateService):
        self._template_service = template_service

    def get_template(self) -> str:
        """現在のテンプレートを取得"""
        return self._template_service.get_template()

    async def reload_template(self) -> None:
        """テンプレートを再読み込み"""
        await self._template_service.reload_template()
