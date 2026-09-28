"""Step-by-step diagnostics for the memory pipeline.

Run from the project root:

    venv\\Scripts\\python.exe tests/step_check.py

This script uses isolated storage (separate SQLite DB and Qdrant folder),
so it never touches the running server's data. Sections marked [net] call
the SiliconFlow API; everything else is local.
"""

# 运行命令：venv\Scripts\python.exe tests\step_check.py

import os
import sys
import time

# 隔离存储，因此开发服务器的Qdrant锁/数据库不受影响。
# Isolated storage so the dev server's Qdrant lock / DB are untouched.
os.environ.setdefault("QDRANT_PATH", "./qdrant_storage_stepcheck")    # 设置环境变量的默认值
os.environ.setdefault("DATABASE_URL", "sqlite:///./stepcheck.db")
os.environ.setdefault("ENABLE_LLM_CONSOLIDATION", "true")

# 把当前文件所在目录的「上一级目录」加入到 Python 模块搜索路径的最前面
# Allow running this file directly: venv\Scripts\python.exe tests/step_check.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    sys.stdout.reconfigure(encoding="utf-8")
    print('utf-8')
except Exception:
    pass

from app.config import settings
from app.schemas import AddRequest, Message, SearchRequest
from app.siliconflow_client import sf_client
from app.database import MemoryUnit, RawChunk, SessionLocal, init_db
from app.qdrant_store import qdrant_store
from app.consolidator import _parse_llm_output, _split_sentences, consolidate
from app.memory_service import (
    _build_query_text,
    _extract_source_ts,
    _keyword_score,
    _normalize_dense_scores,
    _recency_score,
    _serialize_messages,
    _tokens,
    _ts_to_iso,
    add_memory,
    search_memory,
)


def banner(title: str, net: bool = False):
    tag = " [net]" if net else ""
    print()
    print("=" * 70)
    print(f"  {title}{tag}")
    print("=" * 70)


MESSAGES = [
    Message(role="user", content="我喜欢燕麦拿铁，不要加糖。", timestamp=1704067200000),
    Message(role="assistant", content="好的，我记住了。", timestamp=1704067210000),
    Message(role="user", content="我对花生过敏。", timestamp=1704067220000),
]

#  f-string（格式化字符串）
USER_ID = f"stepcheck-{int(time.time())}"
SESSION_ID = "stepcheck-session"
REQUEST_ID = f"stepcheck-req-{int(time.time())}"


def section_1_config():
    banner("1. 配置与模块导入检查")
    print(f"embedding_model       = {settings.embedding_model}")
    print(f"reranker_model        = {settings.reranker_model}")
    print(f"llm_model             = {settings.llm_model}")
    print(f"enable_llm_consolidation = {settings.enable_llm_consolidation}")
    print(f"dense/keyword/recency = {settings.dense_weight} / {settings.keyword_weight} / {settings.recency_weight}")
    print(f"qdrant_path           = {settings.qdrant_path}")
    print(f"database_url          = {settings.database_url}")
    print("api_key 存在:", bool(settings.siliconflow_api_key))


def section_2_serialize():
    banner("2. 消息序列化与时间戳提取（纯本地）")
    text = _serialize_messages(MESSAGES)
    ts = _extract_source_ts(MESSAGES)
    print("输入 messages:")
    for m in MESSAGES:
        print(f"  role={m.role}  content={m.content}  ts={m.timestamp}")
    print(f"_serialize_messages 输出:\n{text}")
    print(f"_extract_source_ts 输出 = {ts}  -> ISO = {_ts_to_iso(ts)}")
    assert "user: 我喜欢燕麦拿铁" in text
    assert ts == 1704067200000


def section_3_fallback():
    banner("3. 句子切分兜底（纯本地，无 LLM）")
    text = _serialize_messages(MESSAGES)
    units = _split_sentences(text)
    print("输入:")
    print(text)
    print("输出（每行一条原子记忆）:")
    for u in units:
        print(f"  - {u}")
    assert units


def section_4_consolidate():
    banner("4. LLM 记忆归档 consolidate()")
    text = _serialize_messages(MESSAGES)
    print("输入:")
    print(text)
    units = consolidate(text)
    print(f"输出（{len(units)} 条事实）:")
    for u in units:
        print(f"  - {u}")
    assert units, "consolidate 不能返回空列表"
    return units


def section_4b_parse():
    banner("4b. LLM 输出解析 _parse_llm_output()（纯本地）")
    samples = [
        '["事实一", "事实二"]',
        '```json\n["带围栏的事实"]\n```',
        '前缀噪音 ["从方括号提取的事实"] 后缀噪音',
        '[{"fact": "字典格式的事实"}]',
    ]
    for s in samples:
        parsed = _parse_llm_output(s)
        print(f"输入: {s!r}")
        print(f"输出: {parsed}")


def section_5_embed(texts):
    banner("5. 向量编码 sf_client.embed()")
    print(f"输入 {len(texts)} 条文本:")
    for t in texts:
        print(f"  - {t}")
    vectors = sf_client.embed(texts)
    v0 = vectors[0]
    print(f"输出 {len(vectors)} 个向量，维度 = {len(v0)}")
    print(f"第一个向量前 5 维: {v0[:5]}")
    assert len(vectors) == len(texts) and len(v0) == 1024
    return vectors


def section_6_upsert(units, vectors):
    banner("6. Qdrant 写入 qdrant_store.upsert()")
    ids = []
    for content, vec in zip(units, vectors):
        point_id = qdrant_store.upsert(
            user_id=USER_ID,
            session_id=SESSION_ID,
            content=content,
            vector=vec,
            unit_type="fact",
            source_ts=1704067200000,
            source_request_id=REQUEST_ID,
        )
        ids.append(point_id)
        print(f"写入 point_id={point_id}  content={content}")
    return ids


def section_7_dense_search():
    banner("7. Qdrant 稠密检索 qdrant_store.search()")
    query_text = _build_query_text("我喜欢喝什么？", ["A. 美式咖啡", "B. 燕麦拿铁", "C. 奶茶"])
    print(f"检索 query（已并入 options）:\n{query_text}")
    qvec = sf_client.embed([query_text])[0]
    candidates = qdrant_store.search(
        query_vector=qvec,
        user_id=USER_ID,
        top_k=10,
    )
    print(f"召回 {len(candidates)} 条（含 Qdrant 余弦分数）:")
    for c in candidates:
        print(f"  score={c['score']:.4f}  content={c['content']}  source_ts={c['source_ts']}")
    assert candidates
    return candidates


def section_8_hybrid(candidates):
    banner("8. 混合打分（关键词 + 新鲜度，纯本地）")
    query = "我喜欢喝什么？"
    latest_ts = max((c.get("source_ts") for c in candidates if c.get("source_ts")), default=None)
    candidates = _normalize_dense_scores(candidates)
    for c in candidates:
        kw = _keyword_score(query, c["content"])
        rec = _recency_score(c.get("source_ts"), latest_ts)
        c["hybrid_score"] = (
            settings.dense_weight * c["dense_norm"]
            + settings.keyword_weight * kw
            + settings.recency_weight * rec
        )
        print(
            f"dense={c['dense_norm']:.4f}  kw={kw:.4f}  rec={rec:.4f}  "
            f"hybrid={c['hybrid_score']:.4f}  | {c['content']}"
        )
    print(f"\nquery 分词示例: {_tokens(query)}")
    candidates.sort(key=lambda c: c["hybrid_score"], reverse=True)
    print(f"混合排序第一名: {candidates[0]['content']}")
    return candidates


def section_9_rerank(candidates):
    banner("9. 重排序 sf_client.rerank()")
    query_text = _build_query_text("我喜欢喝什么？", ["A. 美式咖啡", "B. 燕麦拿铁", "C. 奶茶"])
    documents = [c["content"] for c in candidates]
    results = sf_client.rerank(query=query_text, documents=documents, top_n=len(documents))
    print(f"query:\n{query_text}")
    print(f"rerank 返回 {len(results)} 条（index 对应传入文档下标）:")
    for r in results:
        print(f"  index={r['index']}  score={r['relevance_score']:.4f}  doc={documents[r['index']]}")
    assert results, "rerank 返回为空（检查网络或模型名）"
    return results


def section_10_end_to_end():
    banner("10. 端到端 add_memory() + search_memory()")
    add_req = AddRequest(
        request_id=REQUEST_ID,
        messages=[m.model_dump() for m in MESSAGES],
        user_id=USER_ID,
        session_id=SESSION_ID,
    )
    add_resp = add_memory(add_req)
    print(f"Add 响应: {add_resp.model_dump()}")

    search_req = SearchRequest(
        query="用户对什么过敏？",
        user_id=USER_ID,
        top_k=100,
    )
    search_resp = search_memory(search_req)
    print(f"Search 返回 {len(search_resp.data)} 条:")
    for item in search_resp.data:
        print(f"  score={item.score:.4f}  created_at={item.created_at}  content={item.content}")
    assert search_resp.data, "端到端 Search 返回空"


def section_11_db():
    banner("11. SQLite 持久化检查（RawChunk / MemoryUnit）")
    db = SessionLocal()
    try:
        chunks = db.query(RawChunk).filter(RawChunk.user_id == USER_ID).all()
        print(f"raw_chunks 表 {len(chunks)} 行:")
        for r in chunks:
            print(f"  request_id={r.request_id}  source_ts={r.source_ts}")
            print(f"    content={r.content!r}")
        units = db.query(MemoryUnit).filter(MemoryUnit.user_id == USER_ID).all()
        print(f"memory_units 表 {len(units)} 行:")
        for u in units:
            print(f"  id={u.id}  type={u.unit_type}  source_ts={u.source_ts}")
            print(f"    content={u.content}")
        assert chunks and units
    finally:
        db.close()


if __name__ == "__main__":
    init_db()
    section_1_config()
    section_2_serialize()
    section_3_fallback()
    units = section_4_consolidate()
    section_4b_parse()
    vectors = section_5_embed(units)
    section_6_upsert(units, vectors)
    candidates = section_7_dense_search()
    candidates = section_8_hybrid(candidates)
    section_9_rerank(candidates)
    section_10_end_to_end()
    section_11_db()
    print()
    print("全部检查通过。")
