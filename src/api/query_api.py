import os
import copy
from typing import Union, List, Dict, Any
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
import httpx
import logging
import json
import string
import re
from dotenv import load_dotenv
from urllib.parse import urljoin

load_dotenv()

# ログ設定
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level))
logger = logging.getLogger(__name__)


# 入力モデル
class QueryRequest(BaseModel):
    query: str = ""
    top_k: int = 5


# 出力モデル
class QueryResponse(BaseModel):
    answer: Union[str, Dict[str, Any]]
    references: List[Dict[str, Any]]


class FaissBackend:
    def __init__(self):
        self.set_base()

    def set_base(self):
        self.base = f"http://{os.getenv('FAISS_HOST', 'localhost')}:{os.getenv('FAISS_PORT', 8000)}"

    def make_url(self, endpoint: str) -> str:
        return urljoin(self.base + "/", endpoint.lstrip("/"))

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(self.make_url("/health"))
                return resp.status_code == 200 and resp.json().get("status") == "HAPPY"
        except Exception as e:
            logger.error(f"[FAISS] ヘルスチェック失敗: {e}")
            return False

    async def search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    self.make_url("/knowledge/search"),
                    params={
                        "text": query,
                        "top_k": top_k,
                        "threshold": 0.5,
                        "min_k": 3,
                        "fallback": True,
                    },
                )
                return resp.json().get("results", [])
        except (httpx.HTTPError, httpx.ConnectError, httpx.TimeoutException) as e:
            logger.error(f"[FAISS] 検索失敗: {e}")
            raise HTTPException(status_code=500, detail="FAISS検索に失敗しました")


class OllamaBackend:
    def __init__(self):
        self.set_base()
        self.model = os.getenv("OLLAMA_LLM_MODEL")
        if not self.model:
            raise RuntimeError("ENV: OLLAMA_LLM_MODEL is None.")

    def set_base(self):
        self.base = f"http://{os.getenv('OLLAMA_HOST', 'localhost')}:{os.getenv('OLLAMA_PORT', 11434)}"

    def make_url(self, endpoint: str) -> str:
        return urljoin(self.base + "/", endpoint.lstrip("/"))

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(self.make_url(""))
                return resp.status_code == 200 and "Ollama is running" in resp.text
        except (httpx.HTTPError, httpx.ConnectError, httpx.TimeoutException) as e:
            logger.error(f"[Ollama] ヘルスチェック失敗: {e}")
            return False

    async def generate(self, prompt: str) -> str:
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    self.make_url("/api/generate"),
                    json={"model": self.model, "prompt": prompt, "stream": False},
                )
                if resp.status_code != 200:
                    logger.error(f"[Ollama] 応答失敗: {resp.status_code} - {resp.text}")
                    raise HTTPException(status_code=500, detail="LLM応答に失敗しました")
                data = resp.json()
                answer = data.get("response", "")
                if not answer.strip():
                    logger.warning("[Ollama] 応答が空です")
                    raise HTTPException(status_code=500, detail="LLM応答が空でした")
                return answer
        except HTTPException:
            raise
        except (httpx.HTTPError, httpx.ConnectError, httpx.TimeoutException) as e:
            logger.error(f"[Ollama] 呼び出し失敗: {e}")
            raise HTTPException(
                status_code=500, detail="LLM呼び出しで例外が発生しました"
            )


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup処理
    load_dotenv(override=True)

    # バックエンド初期化
    app.state.FB = FaissBackend()
    app.state.OB = OllamaBackend()

    # FAISSバックエンドヘルスチェック
    if not await app.state.FB.health_check():
        raise RuntimeError("FAISS backend に接続できませんでした.")
    logger.info("FAISS backend is happy!")

    # Ollamaバックエンドヘルスチェック
    if not await app.state.OB.health_check():
        raise RuntimeError("Ollamaに接続できませんでした.")
    logger.info("Ollama is running")

    # promptテンプレート読込
    template_path = os.getenv("PROMPT_TEMPLATE_PATH", "prompt_template.txt")
    if not os.path.exists(template_path):
        raise RuntimeError(
            f"ENV: PROMPT_TEMPLATE_PATH '{template_path}' does not exist."
        )

    with open(template_path, encoding="utf-8") as f:
        app.state.prompt_template = f.read()

    yield
    # 終了処理（必要ならここに記述）


app = FastAPI(
    title="RAG Query API",
    description="FAISS + OllamaによるRAG構成",
    version="1.0.0",
    lifespan=lifespan,
)


def format_references(references: List[Dict[str, Any]]) -> str:
    formatting_references = copy.deepcopy(references)
    for r in formatting_references:
        r.pop("rank", None)
        r.pop("similarity_score", None)
    return json.dumps(formatting_references, ensure_ascii=False, indent=2)


def build_prompt(template: str, references: str, query: str) -> str:
    # example_prompt = f"{context_text}\n\n質問: {request.query}\n回答:"
    tmpl = string.Template(template)
    return tmpl.safe_substitute(references=references, query=query)


async def run_rag_query(app: FastAPI, query: str, top_k: int) -> QueryResponse:
    # references整形
    references = await app.state.FB.search(query, top_k)
    logger.debug(json.dumps(references, ensure_ascii=False, indent=2))
    formatting_references = format_references(references)

    # prompt作成
    prompt = build_prompt(app.state.prompt_template, formatting_references, query)
    logger.debug(f"{prompt=}")

    # ollama呼び出し
    answer = await app.state.OB.generate(prompt)
    # コメント除去（// と /* */ 両方対応）
    answer = re.sub(r"//.*", "", answer)
    answer = re.sub(r"/\*.*?\*/", "", answer, flags=re.DOTALL)
    # コードブロック除去（```json 以外にも対応）
    answer = re.sub(r"^```[a-z]*\n|\n```$", "", answer.strip())

    # JSON文字列で返ってくることを想定してパース
    try:
        parsed_answer = json.loads(answer)
        if isinstance(parsed_answer, dict):
            parsed_answer = json.dumps(parsed_answer, ensure_ascii=False, indent=2)
    except json.JSONDecodeError:
        logger.warning("Ollamaの応答がJSONではありませんでした。文字列として返します。")
        parsed_answer = answer

    return QueryResponse(answer=parsed_answer, references=references)


@app.post("/debug/search")  # , include_in_schema=False)
async def simple_search(
    text: str = Query(..., description="検索文字列"),
    top_k: int = Query(5, description="faiss検索注入件数"),
) -> JSONResponse:
    """開発・デバッグ用のFAISS検索エンドポイント"""
    logger.debug(f"{text=}")
    ret = await run_rag_query(app, text, top_k)
    logger.debug(ret.answer)
    try:
        # answerが文字列の場合のみJSON変換を試みる
        if isinstance(ret.answer, str):
            answer = json.loads(ret.answer)
            return JSONResponse(content=answer)
        else:
            # 既にdictの場合はそのまま返す
            return JSONResponse(content=ret.answer)
    except json.JSONDecodeError as e:
        logger.debug(f"JSON変換失敗: {e}")
        # JSON変換失敗時は文字列をそのまま返す
        return JSONResponse(content={"answer": ret.answer})


@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """ユーザークエリを受け取り、FAISS検索+Ollama応答を返す"""
    if request.query.strip() == "":
        raise HTTPException(status_code=400, detail="queryが空欄です。")
    return await run_rag_query(app, request.query, request.top_k)


@app.get("/prompt_template")
async def read_template() -> JSONResponse:
    """現在のプロンプトテンプレート表示

    Returns:
        JSONResponse: _description_
    """
    return JSONResponse(content={"template": app.state.prompt_template})


@app.post("/prompt_template/reload")
async def reload_prompt() -> JSONResponse:
    """プロンプトテンプレートの再読込,[PROMPT_TEMPLATE_PATH]を読み込む.

    Raises:
        HTTPException: _description_

    Returns:
        JSONResponse: _description_
    """
    try:
        with open(
            os.getenv("PROMPT_TEMPLATE_PATH", "prompt_template.txt"), encoding="utf-8"
        ) as f:
            app.state.prompt_template = f.read()
        return JSONResponse(content={"status": "reloaded"})
    except (OSError, IOError) as e:
        logger.error(f"[reload_prompt] テンプレート再読込失敗: {e}")
        raise HTTPException(
            status_code=500, detail="テンプレート再読込に失敗しました。"
        )


@app.post("/backend/reload")
async def reload_backends() -> JSONResponse:
    try:
        # 現在の設定をバックアップ
        old_faiss_base = app.state.FB.base
        old_ollama_base = app.state.OB.base
        old_ollama_model = app.state.OB.model

        # .env 再読み込み
        load_dotenv(override=True)

        # 新しい設定を適用
        app.state.FB.set_base()
        app.state.OB.set_base()
        app.state.OB.model = os.getenv("OLLAMA_LLM_MODEL")
        if not app.state.OB.model:
            raise RuntimeError("ENV: OLLAMA_LLM_MODEL is None.")

        # ヘルスチェック
        faiss_ok = await app.state.FB.health_check()
        ollama_ok = await app.state.OB.health_check()

        if not (faiss_ok and ollama_ok):
            # ロールバック
            logger.warning("ヘルスチェックに失敗したため、設定をロールバックします")
            app.state.FB.base = old_faiss_base
            app.state.OB.base = old_ollama_base
            app.state.OB.model = old_ollama_model
            raise RuntimeError("ヘルスチェックに失敗したため、設定を元に戻しました。")

        logger.info("バックエンド設定を再読み込みしました")
        logger.info(f"FAISS base: {app.state.FB.base}")
        logger.info(f"Ollama base: {app.state.OB.base}, model: {app.state.OB.model}")

        return await health_check()

    except (RuntimeError, OSError) as e:
        logger.error(f"[reload_backends] バックエンド再設定失敗: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check() -> JSONResponse:
    faiss_ok = await app.state.FB.health_check()
    ollama_ok = await app.state.OB.health_check()
    status = "ok" if faiss_ok and ollama_ok else "degraded"
    http_status = 200 if status == "ok" else 503
    return JSONResponse(
        status_code=http_status,
        content={
            "status": status,
            "faiss": faiss_ok,
            "ollama": ollama_ok,
            "faiss_base": app.state.FB.base,
            "ollama_base": app.state.OB.base,
            "model": app.state.OB.model,
        },
    )


@app.post("/reload_all")
async def reload_all() -> JSONResponse:
    try:
        await reload_backends()
        await reload_prompt()
        return await health_check()
    except HTTPException:
        raise
    except (RuntimeError, OSError) as e:
        logger.error(f"[reload_all] 全体再読み込み失敗: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app, host=os.getenv("HOST", "localhost"), port=int(os.getenv("PORT", 8010))
    )
