"""Ollama LLMサービスの実装"""

import logging
import httpx
from urllib.parse import urljoin

from ..domain import ILlmService, ServiceUnavailableError, LlmGenerationError
from .config import Config

logger = logging.getLogger(__name__)


class OllamaLlmService(ILlmService):
    """Ollama LLMサービスの実装"""

    def __init__(self, config: Config):
        self._config = config
        self._base_url = f"http://{config.ollama_host}:{config.ollama_port}"

    def _make_url(self, endpoint: str) -> str:
        """エンドポイントURLを作成"""
        return urljoin(self._base_url + "/", endpoint.lstrip("/"))

    async def generate(self, prompt: str) -> str:
        """プロンプトから回答を生成"""
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    self._make_url("/api/generate"),
                    json={
                        "model": self._config.ollama_model,
                        "prompt": prompt,
                        "stream": False,
                    },
                )
                if resp.status_code != 200:
                    logger.error(f"[Ollama] 応答失敗: {resp.status_code} - {resp.text}")
                    raise LlmGenerationError("LLM応答に失敗しました")

                data = resp.json()
                answer = data.get("response", "")
                if not answer.strip():
                    logger.warning("[Ollama] 応答が空です")
                    raise LlmGenerationError("LLM応答が空でした")

                return answer

        except httpx.RequestError as e:
            logger.error(f"[Ollama] 呼び出し失敗: {e}")
            raise ServiceUnavailableError("Ollamaサービスが利用できません")
        except LlmGenerationError:
            raise
        except Exception as e:
            logger.error(f"[Ollama] 呼び出し失敗: {e}")
            raise LlmGenerationError("LLM呼び出しで例外が発生しました")

    async def health_check(self) -> bool:
        """サービスの健全性をチェック"""
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(self._make_url(""))
                return resp.status_code == 200 and "Ollama is running" in resp.text
        except Exception as e:
            logger.error(f"[Ollama] ヘルスチェック失敗: {e}")
            return False

    def reload_config(self, config: Config) -> None:
        """設定を再読み込み"""
        self._config = config
        self._base_url = f"http://{config.ollama_host}:{config.ollama_port}"
