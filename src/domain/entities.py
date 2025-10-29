"""ドメインエンティティ"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class SearchResult:
    """検索結果を表すドメインエンティティ"""

    content: str
    rank: Optional[int] = None
    similarity_score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class Query:
    """クエリを表すドメインエンティティ"""

    text: str
    top_k: int = 5

    def __post_init__(self):
        if not self.text.strip():
            raise ValueError("クエリテキストは空にできません")
        if self.top_k <= 0:
            raise ValueError("top_kは正の数である必要があります")


@dataclass(frozen=True)
class RagResult:
    """RAG処理結果を表すドメインエンティティ"""

    answer: str
    references: List[SearchResult]
    query: Query
