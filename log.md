# Agent Memory Challenge - 操作日志

## 2026-10-12

### 10. 区分"LLM 判定无事实"与"LLM 内容异常"的兜底策略
- 背景：纯寒暄 chunk（如 LoCoMo 里的问候/感谢）在 LLM 归档开启时仍会把 "Thanks, Caroline"、"Good luck" 等拆成原子事实入库（2026-10-09 conv-26:chunk-001 实证）。
- 修改 `app/consolidator.py::consolidate()`：
  - LLM **正常返回但 facts 为空** → 直接返回 `[]`，**不再**走 `_split_sentences` 句子切分兜底（尊重模型的噪声过滤结果）。
  - LLM **抛异常**（网络/超时/401/JSON 解析失败）→ 仍走 `_split_sentences` 兜底，保证写入不因模型故障而缺失。
- 修改 `app/memory_service.py::add_memory()`：
  - `consolidate()` 返回空且 `ENABLE_LLM_CONSOLIDATION=true` 时，**不再**把整块原文硬塞成一条"事实"，该 chunk 只存 `raw_chunks` 证据，不写任何 `memory_units`；
  - `ENABLE_LLM_CONSOLIDATION=false` 且切分也无结果时保留旧行为（退化整块原文）；
  - raw-sentence 二次切分通道（`enable_raw_sentence_fallback`）增加 `and consolidated` 条件：LLM 已判定无事实时不触发机械切分。
- 验证：离线打桩 `sf_client.chat_structured` 三种分支全部符合预期；`pytest tests/test_smoke.py tests/test_episodes.py` 4 passed。
- 遗留：origin：`_split_sentences` 本身仍无内容过滤（异常兜底路径会把 "Hello" 等拆成事实，属预期保留行为）；测试期发现的旧文件 ACL 只读问题（test_aml_memory.db / .pytest_cache）与本次改动无关。

### 11. 阿里云服务器部署与环境摸底（47.79.224.195）
- 服务器：Alibaba Cloud Linux 3、内存 1.8G、系统 Python 3.6/3.8、无 uv。
- 摸底发现既有环境：`/opt/aml-memory`（10-11 日旧部署，Python 3.8.17 venv，`aml-memory.service` systemd 跑在 8000 端口）；另有多依赖 docker 容器（opencode/feishu-bot/ghost/nginx/mailpit）与 1panel，这些用户的其它服务未动。
- 新代码部署于 `/root/dsh_jige`：uv + Python 3.12 venv + requirements 安装完成；关闭了被取代的旧 `aml-memory.service`（释放 ~100MB 内存与 8000 端口），可用内存 808→899MB。
- 服务器可直连 OpenAI/SiliconFlow/阿里云 embedding（本地直连 OpenAI 超时的替代测试环境，已确认）。

### 12. 修复链路问题（服务器实测发现）
- `chat_structured` 的 `json_object` 降级分支必挂（OpenAI 要求 messages 含 "json" 字样，中文 prompt 无 → 400）；改为纯 chat 降级 + 容错 JSON 提取（`_extract_json_object` 处理围栏/散文包裹）。
- `llm_consolidation_max_tokens` 默认 1024 → 8192：大 chunk（~2000 词）结构化输出被截断导致 JSONDecodeError，进而每块静默降级句子切分。改后最大 chunk（3962 字符/20 消息）8.6s 抽出 9 条结构化事实。
- 修复 `tests/step_check.py`：`_build_query_text` 缺失（memory_service 补上并让 search_memory 复用）+ 第 5/6 段把 consolidate 的 dict 单元当字符串喂 embed 的 bug。服务器上 `全部检查通过`。
- `bench/run_locomo_eval.py` 增加 `--category` 过滤参数。

### 13. Cat 1（Multi-hop）定向评测（conv-26，10 题）
- 命令：`bench/run_locomo_eval.py --limit 1 --category 1 --max-questions 10 --tag cat1 --fresh`。
- 写入：21 块 → 333 条事实、143 个事件（LLM 抽取正常，无降级）。
- 结果：**hit@100=1.0000、coverage@k=1.0000、MRR=0.5573**（cat 1: n=10）。
- 定性（详见会话）：单跳路径题（research/career/support）top1 命中（MRR=1.0）；跨会话组装题（camping 海滩/山/森林、LGBTQ 参与方式）top5 内部分恢复；计数/聚合题（2023 去海滩次数=2）与身份题（transgender woman，rank~14）top5 未命中——所有金标证据都在 top-100 池内（coverage=1.0），瓶颈在排序精度与"完整链条"聚合恢复，答案组装仍依赖下游模型。

### 14. SYSTEM_PROMPT 补充 JSON 输出格式约束（合规修复）
- 核实：原 SYSTEM_PROMPT（全中文）确实不含 "json/JSON" 字样；OpenAI 的 `response_format=json_object` 要求 messages 中必须出现 "json"（此前该降级分支因此 400 必挂）。
- `app/consolidator.py::SYSTEM_PROMPT` 新增 "5、输出格式要求"：唯一输出必须是合法 JSON 对象（顶层 `{"facts": [...]}`、严格符合 JSON Schema、必填字段给 null、禁 Markdown 围栏/前言/额外字段）。
- `app/siliconflow_client.py::chat_structured` 恢复 `json_object` 二级降级（json_schema → json_object → 纯 chat + 容错解析），并保证 messages 无 "json" 时自动附加输出格式提示（兼容其它调用方，如事件归属判定）。
- 服务器复测：`json_object` 模式现被 OpenAI 接受（不再 400）；`consolidate()` 对真实 chunk（3962 字符/20 消息）产出 8-10 条结构化事实，字段齐全（text/evidence/subject/action/fact_type/event_title/outcome），输出合理。

### 15. Query embedding 前缀实验（结论：对最终评测指标零影响）
- 背景：短查询×长文档导致稠密检索区分度差（0.16-0.30 窄带）；单 query 实测 bge 英文前缀把正确答案稠密排名从 4→1。
- 实现：`settings.search_query_prefix`（默认 "Represent this sentence for searching relevant passages: "），仅拼进 `search_memory` 的 embedding 输入，不污染 rerank / 关键词打分。
- 两次 `--fresh` 评测对比：MRR 0.5573 → 0.5708（+0.0135）——但两次记忆库本就不同（333 vs 337 units，LLM 抽取随机），提升无法归因于前缀。
- **A/B 对照（同一存储 337 units，仅切换前缀）**：10 题逐题 MRR 完全相同（delta 全 0.000，MRR=0.5208）。结论：**rerank（交叉编码器，全排序、无截断）主导最终顺序，稠密阶段前缀被完全吸收，对最终指标零影响**；前缀的价值仅在 rerank 降级/小 top_k 截断场景的兜底。
- 另发现：同一存储下评测 0.5708 vs A/B 0.5208（q0030 beach 计数题 1.0↔0.5 摆动）→ rerank API 存在 run-to-run 波动（±0.05），加上重建随机性（±0.02-0.03），单次评测 MRR 不可作为微小改动的证据。
- 方法论教训：本项目 A/B 必须复用同一记忆库（`--fresh` 重建噪声太大）；要真正提升 Cat 1 MRR 应改 rerank 侧/候选质量，而非 query embedding。

### 16. Episode 分配过分割根因（同 chunk 三条收养事实被拆成两个 episode）
- 案例：chunk-1 内 msg[05] research adoption / msg[09] chose LGBTQ agency / msg[11] create family（单亲）三条事实被拆成 adoption research + adoption agency 两个 episode。
- 重放决策（按插入顺序 fact[0..4]，真实 API + 临时副本库）：fact[4]（chose LGBTQ agency）时，qdrant 向量近邻**强烈指向 adoption research（cosine=0.6768，最高）**，但 `_text_similarity`（token 集合 Jaccard）只有 **0.091** → 命中 `best_sim<=0.25 且候选>=2 → 新建` 规则分支，**连模糊带 LLM 判定都没走到**，直接新建 adoption agency。
- 根因：`_text_similarity` 是字面 token Jaccard，对词面差异（researching agencies vs chose agency/LGBTQ）极度敏感；向量语义（0.68）与词面 Jaccard（0.09）严重背离时，规则选择了错误的信号。且 `len(candidates)>=2` 使"候选越多越容易新建"（与直觉相反）。对照 fact[5]（create family）因与 summary 共享 family/kids/home 词面而 Jaccard≥0.6 并入——同一收养线索的合并/拆分完全由词面重合决定。
- 修复方向（待用户确认）：① 决策信号换成向量余弦（fact_vec vs episode 代表向量/anchor），本案例 0.68 足以正确归并；② 扩大模糊带（best_sim≥0.05 即走 LLM 判定）；③ 最小改动：`<=0.25 且候选≥2` 分支加"若向量余弦≥0.5 仍走 LLM 判定"。

### 17. 扩大 episode 模糊带（0.25 → 0.05）修复验证
- 实施：`config.py` 新增 `episode_create_sim_threshold: float = 0.05`（原硬编码 0.25），`memory_service._assign_episode` 的"直接新建"分支改用该阈值：sim≤0.05 且候选≥2 才新建，0.05~0.6 全部进入 LLM 判定。
- 重放同案例（fact[0..5]，真实 API）：fact[4]（chose LGBTQ agency，text_sim=0.091 但向量 cosine=0.6768）不再直接新建，进入模糊带后 **LLM 判定 same_episode_id=adoption research → 合并**；fact[5]（create family，sim=0.238）同样 LLM 判合并。**三条收养/家庭事实最终全部归并到同一个 adoption research episode，不再产生两个事件** ✅。
- 附带验证：fact[1]/[2]/[3]（me-time/camping/adoption research 首条）走 LLM 判定均正确判 None→新建各自独立主题——LLM 能正确区分相关/无关。
- 代价：模糊带扩大使更多 case 触发 LLM 调用（本次 6 条事实 5 次判定；旧逻辑仅候选=1 的 case 走 LLM），批量写入成本/耗时上升。后续可考虑用向量余弦（本案例 0.68 判别力足够）作为规则信号减少 LLM 依赖（即 anchor/centroid 方案）。

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
