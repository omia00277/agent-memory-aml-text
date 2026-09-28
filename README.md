# Agent Memory Leaderboard - 文本赛道记忆系统

本项目为第二届 Agent Memory Challenge（CSIG）文本赛道开源方法榜的参赛记忆系统。
系统仅实现 AML 规范要求的 `Add`（记忆写入）和 `Search`（记忆检索）接口，答案生成与评分由 AML 平台统一完成。

记忆链路：写入时把原始对话块归档为原子化事实记忆，检索时用稠密向量、关键词与时间新鲜度做混合召回，再经重排序返回排序后的证据。

## 技术栈

- **后端框架**：Python + FastAPI
- **向量数据库**：Qdrant（稠密向量检索）
- **关系数据库**：PostgreSQL（原始 chunk 持久化、审计、异步任务）
- **Embedding 模型**：BAAI/bge-m3（通过 SiliconFlow API）
- **Reranker 模型**：BAAI/bge-reranker-v2-m3（通过 SiliconFlow API）
- **提取/摘要 LLM**：Qwen/Qwen2.5-7B-Instruct（Add 内同步事实抽取，可关闭）
- **记忆归档**：LLM 事实抽取 + 句子切分兜底
- **检索策略**：稠密向量 + 关键词 + 时间新鲜度混合打分，再 rerank

## 项目结构

```
.
├── app/
│   ├── config.py              # 环境变量配置
│   ├── consolidator.py        # 记忆归档与事实抽取
│   ├── database.py            # PostgreSQL 模型与会话
│   ├── main.py                # FastAPI 入口与 /add /search /health
│   ├── memory_service.py      # Add/Search 业务逻辑
│   ├── qdrant_store.py        # Qdrant 向量存储
│   ├── schemas.py             # Pydantic 请求/响应模型
│   └── siliconflow_client.py  # SiliconFlow API 客户端
├── tests/
│   └── test_smoke.py          # 接口结构冒烟测试
├── docker-compose.yml         # Qdrant + PostgreSQL + App
├── Dockerfile                 # 应用容器镜像
├── requirements.txt           # Python 依赖
├── .env.example               # 环境变量示例
├── README.md                  # 本文件
└── log.md                     # 操作日志
```

## 快速开始

### 1. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入你的 SiliconFlow API Key
```

必填变量：
- `SiliconFlow_api_key`
- `SiliconFlow_base_url`
- `EMBEDDING_MODEL`
- `RERANKER_MODEL`
- `LLM_MODEL`
- `QDRANT_URL`
- `POSTGRES_URL`

可选变量：
- `ENABLE_LLM_CONSOLIDATION`：是否启用 LLM 记忆归档（默认 `true`；置 `false` 时仅做句子切分）
- `DENSE_WEIGHT` / `KEYWORD_WEIGHT` / `RECENCY_WEIGHT`：混合检索权重
- `DENSE_RECALL_MULTIPLIER`：稠密召回放大倍数

### 2. 启动依赖服务

确保 Docker Desktop 已启动，然后执行：

```bash
docker compose up -d
```

这会启动 Qdrant（端口 6333）和 PostgreSQL（端口 5432）。

### 3. 本地运行应用（推荐开发调试）

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. 使用 Docker 运行完整栈

```bash
docker compose up -d --build
```

应用将运行在 http://localhost:8000。

## 接口说明

### Health

```http
GET /health
```

返回：
```json
{"status": "ok"}
```

### Add

```http
POST /add
Content-Type: application/json

{
  "request_id": "eval:run_abc123:locomo_refined:conv-0:chunk-0",
  "messages": [
    {"role": "user", "content": "我喜欢燕麦拿铁。", "timestamp": 1704067200000}
  ],
  "user_id": "eval:run_abc123:locomo:conv-0",
  "session_id": "eval:run_abc123:sample:0"
}
```

返回：
```json
{
  "success": true,
  "request_id": "eval:run_abc123:locomo_refined:conv-0:chunk-0",
  "user_id": "eval:run_abc123:locomo:conv-0",
  "session_id": "eval:run_abc123:sample:0"
}
```

### Search

```http
POST /search
Content-Type: application/json

{
  "query": "我喜欢喝什么？",
  "options": ["A. 美式", "B. 燕麦拿铁", "C. 奶茶"],
  "user_id": "eval:run_abc123:locomo:conv-0",
  "top_k": 100
}
```

返回：
```json
{
  "data": [
    {
      "id": "mem-xxx",
      "content": "用户喜欢燕麦拿铁且不加糖",
      "score": 0.95,
      "created_at": "2024-01-01T00:00:00+00:00"
    }
  ]
}
```

`content` 为归档后的自包含事实，`created_at` 优先取消息来源时间戳，便于回答模型与时间类问题使用。

## 测试

```bash
pytest tests/test_smoke.py -v
```

冒烟测试会 Mock 外部模型调用，重点校验接口结构和响应格式。

## 注意事项

1. `Add` 是同步语义：必须在数据持久化且可检索后才能返回 HTTP 200。
2. `Search` 仅使用 `user_id` 作为检索范围，禁止跨 `user_id` 返回记忆。
3. 返回的 `data` 数组条数不得超过 `top_k`。
4. 评测数据不得用于模型训练或外部传播，任务完成后应按赛事要求在 30 天内删除。

## 联系方式

赛事官网：https://agentmemoryleaderboard.ai
赛事仓库：https://github.com/AML-memory/agent-memory-leaderboard
