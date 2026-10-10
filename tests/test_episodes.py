import json
import os

os.environ.setdefault("SiliconFlow_api_key", "sk-test")
os.environ.setdefault("qdrant_url", "")
os.environ.setdefault("qdrant_path", "./test_episode_qdrant")
os.environ.setdefault("database_url", "sqlite:///./test_episode_memory.db")
os.environ.setdefault("ENABLE_LLM_CONSOLIDATION", "false")
os.environ.setdefault("ENABLE_EPISODE_BUILDING", "true")

from app.consolidator import _parse_llm_output
from app.database import Episode, MemoryUnit, init_db


def test_structured_fact_parser_requires_text_and_keeps_evidence():
    parsed = _parse_llm_output(json.dumps([
        {"text": "用户参加了会议", "evidence": "user: 我参加了会议", "role": "user", "source_id": "D1:1"},
        {"evidence": "丢弃，没有 text"},
    ], ensure_ascii=False))
    assert len(parsed) == 1
    assert parsed[0]["text"] == "用户参加了会议"
    assert parsed[0]["evidence"] == "user: 我参加了会议"
    assert parsed[0]["role"] == "user"


def test_episode_schema_is_available_and_migrates():
    init_db()
    episode = Episode(user_id="u", session_id="s", title="会议", summary="用户参加会议")
    assert episode.user_id == "u"
    unit = MemoryUnit(user_id="u", session_id="s", source_request_id="r", content="用户参加会议", evidence="原文")
    assert unit.evidence == "原文"
