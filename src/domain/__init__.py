"""ドメイン層の公開インターフェース"""

from .entities import SearchResult, Query, RagResult
from .interfaces import (
    IKnowledgeSearchService,
    ILlmService,
    IPromptTemplateService,
    IRagService,
)
from .exceptions import (
    DomainException,
    ValidationError,
    ServiceUnavailableError,
    SearchError,
    LlmGenerationError,
    TemplateError,
)

__all__ = [
    "SearchResult",
    "Query",
    "RagResult",
    "IKnowledgeSearchService",
    "ILlmService",
    "IPromptTemplateService",
    "IRagService",
    "DomainException",
    "ValidationError",
    "ServiceUnavailableError",
    "SearchError",
    "LlmGenerationError",
    "TemplateError",
]
