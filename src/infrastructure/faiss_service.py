"""FAISS検索サービスの実装"""

import logging
from typing import List, Dict, Any
import httpx
from urllib.parse import urljoin

from ..domain import (
    IKnowledgeSearchService,
    Query,
    SearchResult,
    ServiceUnavailableError,
    SearchError,
)
from .config import Config

logger = logging.getLogger(__name__)


class FaissSearchService(IKnowledgeSearchService):
    """FAISS検索サービスの実装"""

    def __init__(self, config: Config):
        self._config = config
        self._base_url = f"http://{config.faiss_host}:{config.faiss_port}"

    def _make_url(self, endpoint: str) -> str:
        """エンドポイントURLを作成"""
        return urljoin(self._base_url + "/", endpoint.lstrip("/"))

    async def search(self, query: Query) -> List[SearchResult]:
        """クエリに基づいてナレッジベースを検索"""
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    self._make_url("/knowledge/search"),
                    params={
                        "text": query.text,
                        "top_k": query.top_k,
                        "threshold": 0.5,
                        "min_k": 3,
                        "fallback": True,
                    },
                )
                if resp.status_code != 200:
                    raise SearchError(f"FAISS検索失敗: {resp.status_code}")

                results_data = resp.json().get("results", [])
                return [
                    self._convert_to_search_result(result) for result in results_data
                ]

        except httpx.RequestError as e:
            logger.error(f"[FAISS] 検索失敗: {e}")
            raise ServiceUnavailableError("FAISS検索サービスが利用できません")
        except Exception as e:
            logger.error(f"[FAISS] 検索失敗: {e}")
            raise SearchError("FAISS検索に失敗しました")

    async def health_check(self) -> bool:
        """サービスの健全性をチェック"""
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(self._make_url("/health"))
                return resp.status_code == 200 and resp.json().get("status") == "HAPPY"
        except Exception as e:
            logger.error(f"[FAISS] ヘルスチェック失敗: {e}")
            return False

    def _convert_to_search_result(self, result_data: Dict[str, Any]) -> SearchResult:
        """APIレスポンスをSearchResultに変換"""
        return SearchResult(
            content=result_data.get("content", ""),
            rank=result_data.get("rank"),
            similarity_score=result_data.get("similarity_score"),
            metadata=result_data.get("metadata"),
        )

    def reload_config(self, config: Config) -> None:
        """設定を再読み込み"""
        self._config = config
        self._base_url = f"http://{config.faiss_host}:{config.faiss_port}"
