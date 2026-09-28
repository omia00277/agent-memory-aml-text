import os

from fastapi.testclient import TestClient

os.environ.setdefault("SiliconFlow_api_key", "sk-test")
os.environ.setdefault("siliconflow_base_url", "https://api.siliconflow.cn/v1")
os.environ.setdefault("qdrant_url", "")
os.environ.setdefault("qdrant_path", "./test_qdrant_storage")
os.environ.setdefault("database_url", "sqlite:///./test_aml_memory.db")
os.environ.setdefault("ENABLE_LLM_CONSOLIDATION", "false")

from app.main import app
from app.database import init_db

init_db()


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


def test_add_and_search_mock(mocker):
    import app.memory_service as ms
    import app.qdrant_store as qs

    mocker.patch.object(ms, "consolidate", return_value=["用户喜欢燕麦拿铁。"])
    mocker.patch.object(
        ms.sf_client,
        "embed",
        side_effect=lambda texts: [[0.1] * 1024 for _ in texts],
    )
    mocker.patch.object(ms.sf_client, "rerank", return_value=[])
    mocker.patch.object(
        qs.qdrant_store,
        "upsert",
        return_value="mem-1",
    )
    mocker.patch.object(
        qs.qdrant_store,
        "search",
        return_value=[
            {
                "id": "mem-1",
                "content": "用户喜欢燕麦拿铁。",
                "score": 0.9,
                "created_at": "2026-09-22T00:00:00+00:00",
                "unit_type": "fact",
                "source_ts": 1704067200000,
                "source_request_id": "test-req-001",
            }
        ],
    )

    add_payload = {
        "request_id": "test-req-001",
        "messages": [
            {"role": "user", "content": "我喜欢燕麦拿铁。", "timestamp": 1704067200000}
        ],
        "user_id": "test-user-001",
        "session_id": "test-session-001",
    }
    with TestClient(app) as client:
        r1 = client.post("/add", json=add_payload)
        assert r1.status_code == 200
        assert r1.json()["success"] is True

        search_payload = {
            "query": "我喜欢喝什么？",
            "user_id": "test-user-001",
            "top_k": 100,
        }
        r2 = client.post("/search", json=search_payload)
        assert r2.status_code == 200
        data = r2.json()["data"]
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["content"] == "用户喜欢燕麦拿铁。"
