"""ドメイン例外クラス"""


class DomainException(Exception):
    """ドメイン層の基底例外"""

    pass


class ValidationError(DomainException):
    """バリデーションエラー"""

    pass


class ServiceUnavailableError(DomainException):
    """サービス利用不可エラー"""

    pass


class SearchError(DomainException):
    """検索エラー"""

    pass


class LlmGenerationError(DomainException):
    """LLM生成エラー"""

    pass


class TemplateError(DomainException):
    """テンプレートエラー"""

    pass
