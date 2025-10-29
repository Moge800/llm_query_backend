"""依存性注入コンテナ"""

from typing import Dict, Any
import logging

from .domain import IKnowledgeSearchService, ILlmService, IPromptTemplateService
from .application import RagUseCase, HealthCheckUseCase, TemplateUseCase
from .infrastructure import (
    Config,
    FaissSearchService,
    OllamaLlmService,
    PromptTemplateService,
)
from .presentation import RagController, HealthController, TemplateController

logger = logging.getLogger(__name__)


class DIContainer:
    """依存性注入コンテナ"""

    def __init__(self):
        self._config: Config = None
        self._services: Dict[str, Any] = {}
        self._usecases: Dict[str, Any] = {}
        self._controllers: Dict[str, Any] = {}

    def initialize(self, config: Config) -> None:
        """コンテナを初期化"""
        self._config = config

        # サービス層を初期化
        self._initialize_services()

        # ユースケース層を初期化
        self._initialize_usecases()

        # コントローラー層を初期化
        self._initialize_controllers()

        logger.info("DIコンテナが初期化されました")

    def _initialize_services(self) -> None:
        """サービス層を初期化"""
        self._services["knowledge_search"] = FaissSearchService(self._config)
        self._services["llm"] = OllamaLlmService(self._config)
        self._services["template"] = PromptTemplateService(self._config)

    def _initialize_usecases(self) -> None:
        """ユースケース層を初期化"""
        self._usecases["rag"] = RagUseCase(
            knowledge_service=self.get_knowledge_search_service(),
            llm_service=self.get_llm_service(),
            template_service=self.get_template_service(),
        )
        self._usecases["health"] = HealthCheckUseCase(
            knowledge_service=self.get_knowledge_search_service(),
            llm_service=self.get_llm_service(),
        )
        self._usecases["template"] = TemplateUseCase(
            template_service=self.get_template_service()
        )

    def _initialize_controllers(self) -> None:
        """コントローラー層を初期化"""
        self._controllers["rag"] = RagController(self.get_rag_usecase())
        self._controllers["health"] = HealthController(self.get_health_usecase())
        self._controllers["template"] = TemplateController(self.get_template_usecase())

    # サービス取得メソッド
    def get_knowledge_search_service(self) -> IKnowledgeSearchService:
        return self._services["knowledge_search"]

    def get_llm_service(self) -> ILlmService:
        return self._services["llm"]

    def get_template_service(self) -> IPromptTemplateService:
        return self._services["template"]

    # ユースケース取得メソッド
    def get_rag_usecase(self) -> RagUseCase:
        return self._usecases["rag"]

    def get_health_usecase(self) -> HealthCheckUseCase:
        return self._usecases["health"]

    def get_template_usecase(self) -> TemplateUseCase:
        return self._usecases["template"]

    # コントローラー取得メソッド
    def get_rag_controller(self) -> RagController:
        return self._controllers["rag"]

    def get_health_controller(self) -> HealthController:
        return self._controllers["health"]

    def get_template_controller(self) -> TemplateController:
        return self._controllers["template"]

    async def health_check_all_services(self) -> bool:
        """全サービスのヘルスチェック"""
        try:
            knowledge_ok = await self.get_knowledge_search_service().health_check()
            llm_ok = await self.get_llm_service().health_check()
            return knowledge_ok and llm_ok
        except Exception as e:
            logger.error(f"サービスヘルスチェック失敗: {e}")
            return False

    async def reload_configuration(self) -> None:
        """設定を再読み込み（環境変数の変更を反映）"""
        try:
            # 新しい設定を読み込み
            new_config = Config.from_env()

            # 各サービスの設定を更新
            if hasattr(self._services["knowledge_search"], "reload_config"):
                self._services["knowledge_search"].reload_config(new_config)

            if hasattr(self._services["llm"], "reload_config"):
                self._services["llm"].reload_config(new_config)

            if hasattr(self._services["template"], "reload_config"):
                self._services["template"].reload_config(new_config)

            self._config = new_config
            logger.info("設定を再読み込みしました")

        except Exception as e:
            logger.error(f"設定再読み込み失敗: {e}")
            raise
