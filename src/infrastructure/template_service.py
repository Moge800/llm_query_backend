"""プロンプトテンプレートサービスの実装"""

import logging
from ..domain import IPromptTemplateService, TemplateError
from .config import Config

logger = logging.getLogger(__name__)


class PromptTemplateService(IPromptTemplateService):
    """プロンプトテンプレートサービスの実装"""

    def __init__(self, config: Config):
        self._config = config
        self._template: str = ""
        self._load_template()

    def get_template(self) -> str:
        """現在のプロンプトテンプレートを取得"""
        return self._template

    async def reload_template(self) -> None:
        """プロンプトテンプレートを再読み込み"""
        try:
            self._load_template()
            logger.info("プロンプトテンプレートを再読み込みしました")
        except Exception as e:
            logger.error(f"プロンプトテンプレート再読み込み失敗: {e}")
            raise TemplateError("テンプレート再読み込みに失敗しました")

    def _load_template(self) -> None:
        """テンプレートファイルを読み込み"""
        try:
            with open(self._config.prompt_template_path, encoding="utf-8") as f:
                self._template = f.read()
        except FileNotFoundError:
            raise TemplateError(
                f"テンプレートファイルが見つかりません: {self._config.prompt_template_path}"
            )
        except Exception as e:
            raise TemplateError(f"テンプレートファイルの読み込みに失敗しました: {e}")

    def reload_config(self, config: Config) -> None:
        """設定を再読み込み"""
        self._config = config
        self._load_template()
