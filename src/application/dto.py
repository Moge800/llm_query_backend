"""アプリケーション層のDTO（Data Transfer Object）"""

from pydantic import BaseModel
from typing import List, Dict, Any


class QueryRequestDto(BaseModel):
    """クエリリクエストDTO"""

    query: str = ""
    top_k: int = 5


class SearchResultDto(BaseModel):
    """検索結果DTO"""

    content: str
    rank: int | None = None
    similarity_score: float | None = None
    metadata: Dict[str, Any] | None = None


class QueryResponseDto(BaseModel):
    """クエリレスポンスDTO"""

    answer: str | dict
    references: List[SearchResultDto]


class HealthStatusDto(BaseModel):
    """ヘルスステータスDTO"""

    status: str
    faiss: bool
    ollama: bool
    faiss_base: str
    ollama_base: str
    model: str


class TemplateDto(BaseModel):
    """テンプレートDTO"""

    template: str


class StatusDto(BaseModel):
    """ステータスDTO"""

    status: str
