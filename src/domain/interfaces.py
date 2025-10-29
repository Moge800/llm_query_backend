"""ドメインサービスのインターフェース"""

from abc import ABC, abstractmethod
from typing import List
from .entities import Query, SearchResult, RagResult


class IKnowledgeSearchService(ABC):
    """ナレッジ検索サービスのインターフェース"""

    @abstractmethod
    async def search(self, query: Query) -> List[SearchResult]:
        """クエリに基づいてナレッジベースを検索する"""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """サービスの健全性をチェックする"""
        pass


class ILlmService(ABC):
    """LLMサービスのインターフェース"""

    @abstractmethod
    async def generate(self, prompt: str) -> str:
        """プロンプトから回答を生成する"""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """サービスの健全性をチェックする"""
        pass


class IPromptTemplateService(ABC):
    """プロンプトテンプレートサービスのインターフェース"""

    @abstractmethod
    def get_template(self) -> str:
        """現在のプロンプトテンプレートを取得する"""
        pass

    @abstractmethod
    async def reload_template(self) -> None:
        """プロンプトテンプレートを再読み込みする"""
        pass


class IRagService(ABC):
    """RAGサービスのインターフェース"""

    @abstractmethod
    async def process_query(self, query: Query) -> RagResult:
        """クエリを処理してRAG結果を返す"""
        pass
