# Agent Memory Challenge - 操作日志

## 2026-09-22

### 1. 环境检查
- 检查 Python 版本：3.14.4
- 检查 Docker 版本：29.3.1，Docker Compose v5.1.1
- 发现 Docker config 文件访问被拒绝的警告，但版本命令可执行
- 测试 SiliconFlow API Key：Key 存在于用户环境变量，但调用返回 401 `Token is invalid.`
- 测试 Docker daemon：未运行，提示找不到 `dockerDesktopLinuxEngine`
- 结论：需要用户启动 Docker Desktop 并核实 SiliconFlow Key

### 2. 项目初始化
- 创建 `docker-compose.yml`：Qdrant + PostgreSQL + App 服务
- 创建 `Dockerfile`：基于 Python 3.12-slim
- 创建 `.env.example`：SiliconFlow、模型、Qdrant、PostgreSQL 配置示例
- 创建 `requirements.txt`：FastAPI、Qdrant、SQLAlchemy、OpenAI 等依赖
- 创建 `app/` 模块：
  - `config.py` 环境配置
  - `schemas.py` Pydantic 模型
  - `database.py` PostgreSQL 定义
  - `siliconflow_client.py` SiliconFlow API 客户端
  - `qdrant_store.py` Qdrant 操作封装
  - `memory_service.py` Add/Search 业务逻辑
  - `main.py` FastAPI 入口
- 创建 `tests/test_smoke.py`：Mock 外部依赖的接口冒烟测试
- 创建 `README.md`：项目说明、快速开始、接口文档
- 创建 `log.md`：本操作日志

### 3. 下一步待办
- 用户核实 SiliconFlow API Key 并启动 Docker Desktop
- 启动 docker compose 并验证 Qdrant / PostgreSQL 可用
- 安装 Python 依赖并运行本地服务
- 执行冒烟测试
- 接入真实模型调用并验证 embedding / rerank 链路
 ### 4. 代码校验与修正
 - 使用 `python -m py_compile` 校验所有 Python 文件
 - 发现 `memory_service.py` 因补丁格式问题存在缩进错误，已删除重写
 - 校验通过的文件：
   - `app/config.py`
   - `app/schemas.py`
   - `app/database.py`
   - `app/siliconflow_client.py`
   - `app/qdrant_store.py`
   - `app/memory_service.py`
   - `app/main.py`
   - `tests/test_smoke.py`

 ### 5. Python 依赖安装尝试
 - 创建本地虚拟环境 `venv`
 - 尝试升级 pip 失败：网络连接被拒绝（WinError 10013）
 - 结论：pip 安装和 Docker 镜像拉取都需要网络授权

 ### 6. 当前阻塞
 - SiliconFlow API Key 返回 401，需要用户核实
 - Docker Desktop 未启动，需要用户启动
 - 本地 pip 安装需要网络访问授权

### 7. 本地冒烟链路打通
- 结束旧 uvicorn 进程并释放 `./qdrant_storage` 文件锁与 9000/9001 端口
- 修复 `qdrant_client 1.19.1` 中已移除的 `client.search()`，改用 `client.query_points()`
- 授权网络模式下重启服务，完成真实 Add/Search 验证：Add 返回 `success=true`，Search 返回记忆证据

### 8. 记忆归档与混合检索
- 新增 `app/consolidator.py`：用 SiliconFlow `Qwen2.5-7B-Instruct` 把多轮消息压缩为原子化事实记忆，失败时回退到句子切分
- 扩展 `app/database.py`：新增 `memory_units` 表，`raw_chunks` 增加 `source_ts`；加入 SQLite 缺列迁移
- 重写 `app/memory_service.py`：Add 同步持久化原始 chunk 并写入合并后记忆单元；Search 采用 稠密向量 + 关键词 + 时间新鲜度 混合打分，再 rerank
- `app/qdrant_store.py` 增加 `unit_type`、`source_ts`、`source_request_id` payload 与召回放大参数
- `app/config.py` 增加 `enable_llm_consolidation` 与混合检索权重配置
- 更新 `.env` / `.env.example`，补充记忆归档与检索权重变量
- 更新冒烟测试以覆盖新流程，`pytest tests/test_smoke.py -v` 通过
- 真实验证：偏好事实 "用户喜欢燕麦拿铁且不加糖"、过敏事实 "用户对花生过敏" 均正确召回；带选项的选择题将正确候选排第一；跨 user 隔离返回空数组；重复 request_id 幂等返回成功

## 2026-09-23

### 9. LoCoMo-Refined 本地评测（第 1 段对话）
- 接入本地数据集 `LoCoMo_refined-main`：10 段对话 / 1,382 题，带金标证据 dia_id
- 新增 `bench/locomo_loader.py`：会话拍平、session 时间解析为 Unix 毫秒、复刻 AML 分段（20 条消息或 2,000 词）
- 新增 `bench/run_locomo_eval.py`：真实调用 LLM 归档 + embedding + rerank 的检索级评测（hit@k / coverage@k / MRR，按类目分解），存储隔离于 `bench.db` / `qdrant_bench`
- conv-26 全链路实测：21 块写入 → 270 条事实归档，138 题检索
- 首轮结果 hit@100=0.9855；排查发现两处 harness 问题并修正：evidence 空列表需剔除、`'D8:6; D9:17'` 分号打包 ID 需拆分
- 修正后结果（137 有效题）：hit@100=1.0000、coverage=1.0000、MRR=0.5286；类目 1/2/3/4 的 MRR = 0.381/0.574/0.281/0.585
- 顺带修复生产问题：LLM 归档输出偶发回显 `user:` 前缀，`_parse_llm_output` 统一剥离；冒烟测试 2 项通过
- 结论：召回层（hit/coverage）在 top-100 内全命中；排序层（MRR≈0.53，类目 3 最弱 0.28）是下一个优化点
