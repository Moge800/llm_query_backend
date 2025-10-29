"""設定管理"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv


@dataclass
class Config:
    """アプリケーション設定"""

    # Server設定
    host: str
    port: int
    log_level: str

    # FAISS設定
    faiss_host: str
    faiss_port: int

    # Ollama設定
    ollama_host: str
    ollama_port: int
    ollama_model: str

    # Template設定
    prompt_template_path: str

    @classmethod
    def from_env(cls) -> "Config":
        """環境変数から設定を読み込み"""
        load_dotenv(override=True)

        ollama_model = os.getenv("OLLAMA_LLM_MODEL")
        if not ollama_model:
            raise ValueError("ENV: OLLAMA_LLM_MODEL is required")

        prompt_template_path = os.getenv("PROMPT_TEMPLATE_PATH", "prompt_template.txt")
        if not os.path.exists(prompt_template_path):
            raise ValueError(
                f"ENV: PROMPT_TEMPLATE_PATH '{prompt_template_path}' does not exist"
            )

        return cls(
            host=os.getenv("HOST", "localhost"),
            port=int(os.getenv("PORT", 8010)),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            faiss_host=os.getenv("FAISS_HOST", "localhost"),
            faiss_port=int(os.getenv("FAISS_PORT", 8000)),
            ollama_host=os.getenv("OLLAMA_HOST", "localhost"),
            ollama_port=int(os.getenv("OLLAMA_PORT", 11434)),
            ollama_model=ollama_model,
            prompt_template_path=prompt_template_path,
        )
