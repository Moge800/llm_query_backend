"""FastAPI エンドポイントのテスト"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch
import sys
import os

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.api.query_api import app, QueryRequest, QueryResponse


@pytest.fixture
def client():
    """TestClientのフィクスチャ"""
    return TestClient(app)


@pytest.fixture
def mock_backends():
    """バックエンドのモック"""
    with (
        patch("src.api.query_api.FaissBackend") as mock_faiss,
        patch("src.api.query_api.OllamaBackend") as mock_ollama,
    ):

        # FAISSモック設定
        faiss_instance = MagicMock()
        faiss_instance.health_check = AsyncMock(return_value=True)
        faiss_instance.search = AsyncMock(
            return_value=[
                {"text": "参考情報1", "similarity_score": 0.95},
                {"text": "参考情報2", "similarity_score": 0.85},
            ]
        )
        mock_faiss.return_value = faiss_instance

        # Ollamaモック設定
        ollama_instance = MagicMock()
        ollama_instance.health_check = AsyncMock(return_value=True)
        ollama_instance.generate = AsyncMock(return_value='{"answer": "テスト回答"}')
        ollama_instance.model = "test-model"
        mock_ollama.return_value = ollama_instance

        yield mock_faiss, mock_ollama


class TestQueryEndpoint:
    """クエリエンドポイントのテスト"""

    def test_query_endpoint_success(self, client, mock_backends):
        """正常なクエリのテスト"""
        response = client.post("/query", json={"query": "テストクエリ", "top_k": 5})
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "references" in data

    def test_query_endpoint_empty_query(self, client):
        """空のクエリのテスト"""
        response = client.post("/query", json={"query": "", "top_k": 5})
        assert response.status_code == 400
        assert "queryが空欄です" in response.json()["detail"]

    def test_query_endpoint_default_top_k(self, client, mock_backends):
        """デフォルトtop_kのテスト"""
        response = client.post("/query", json={"query": "テストクエリ"})
        assert response.status_code == 200


class TestHealthCheck:
    """ヘルスチェックエンドポイントのテスト"""

    def test_health_check_all_ok(self, client, mock_backends):
        """全バックエンドが正常な場合"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["faiss"] is True
        assert data["ollama"] is True

    def test_health_check_degraded(self, client):
        """バックエンドに問題がある場合"""
        with patch.object(
            app.state.FB, "health_check", new_callable=AsyncMock
        ) as mock_faiss_health:
            mock_faiss_health.return_value = False
            response = client.get("/health")
            assert response.status_code == 503
            data = response.json()
            assert data["status"] == "degraded"


class TestPydanticModels:
    """Pydanticモデルのバリデーションテスト"""

    def test_query_request_validation(self):
        """QueryRequestのバリデーション"""
        # 正常なデータ
        req = QueryRequest(query="テスト", top_k=5)
        assert req.query == "テスト"
        assert req.top_k == 5

        # デフォルト値
        req = QueryRequest(query="テスト")
        assert req.top_k == 5

    def test_query_response_structure(self):
        """QueryResponseの構造テスト"""
        resp = QueryResponse(answer="回答", references=[{"text": "参考"}])
        assert resp.answer == "回答"
        assert len(resp.references) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
