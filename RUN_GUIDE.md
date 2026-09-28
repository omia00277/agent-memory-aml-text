# 运行指南（VS Code）

本文档描述如何从零把本项目跑起来：环境激活、VS Code 配置、四类可执行入口（单元测试 / 分步诊断 / 本地评测 / HTTP 接口），每一步的预期输出，以及如何判断"跑对了"。

> 前提说明：Python 虚拟环境 `venv` 已创建且依赖已安装；SiliconFlow API Key 已写入 `.env`。你在自己的终端里运行不需要任何额外网络授权（那是 Codex 沙箱的概念，与你本机无关）。运行单元测试不需要 Docker；只有 `docker-compose.yml` 完整栈部署才需要。

---

## 1. VS Code 初始配置（只需做一次）

1. `File → Open Folder` 打开 `D:\Document\opencode_project\zcode\agent学习\比赛`。
2. 安装微软官方 **Python** 扩展（扩展面板搜索 `Python`）。
3. `Ctrl+Shift+P` → 输入 `Python: Select Interpreter` → 选择：
   `D:\Document\opencode_project\zcode\agent学习\比赛\venv\Scripts\python.exe`
4. `` Ctrl+` `` 打开内置终端。终端提示符前出现 `(venv)` 即激活成功；即使没有 `(venv)` 前缀，下面所有命令都用 `venv\Scripts\python.exe` 全路径调用，不依赖激活状态。

验证解释器选对了，在终端执行：

```powershell
venv\Scripts\python.exe --version
venv\Scripts\python.exe -m pytest --version
```

预期输出：

```text
Python 3.14.4
pytest 9.1.1
```

建议先执行一次，避免中文乱码：

```powershell
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
```

---

## 2. 单元测试（最快验证，约 2 秒，不联网）

```powershell
venv\Scripts\python.exe -m pytest tests\test_smoke.py -v
```

预期输出（关键行）：

```text
tests/test_smoke.py::test_health PASSED                                  [ 50%]
tests/test_smoke.py::test_add_and_search_mock PASSED                     [100%]
======================== 2 passed, * warnings in *.*s ========================
```

**比对标准**：必须是 `2 passed`，退出码 0。warnings（DeprecationWarning / pytest 缓存路径）可以忽略。如果报 `ModuleNotFoundError`，说明解释器没选 venv 里的 python。

VS Code 图形化方式：`Ctrl+Shift+P` → `Python: Configure Tests` → `pytest` → 选 `tests` 目录，之后左侧 Testing 面板里逐个点运行。

---

## 3. 分步诊断脚本（验证每个函数的输入输出，约 1 分钟，联网）

```powershell
venv\Scripts\python.exe tests\step_check.py
```

它使用隔离存储（`stepcheck.db` + `qdrant_storage_stepcheck`），不碰服务数据。共 11 个检测段，每段打印输入和输出。

**逐段预期输出与比对要点**：

| 段 | 内容 | 预期 |
|---|---|---|
| 1 | 配置 | 模型名为 `BAAI/bge-m3` / `bge-reranker-v2-m3` / `Qwen2.5-7B-Instruct`，`api_key 存在: True` |
| 2 | 消息序列化 | 输出含 `user: 我喜欢燕麦拿铁`，时间戳 `1704067200000` → ISO `2024-01-01T00:00:00+00:00` |
| 3 | 句子切分兜底 | 3 条无角色前缀的记忆，如 `我喜欢燕麦拿铁，不要加糖` |
| 4 | LLM 归档 | 1-3 条事实（措辞每次可能不同），必须非空 |
| 4b | LLM 输出解析 | 4 组样例各自正确解析出字符串 |
| 5 | 向量编码 | `2 个向量，维度 = 1024` |
| 6 | Qdrant 写入 | 2 个 point_id（UUID 格式） |
| 7 | 稠密检索 | 召回 2 条，燕麦拿铁分数最高 |
| 8 | 混合打分 | 打印 dense/kw/rec 三个分量，燕麦拿铁 hybrid 第一 |
| 9 | 重排序 | 2 条结果，index 0 分数高于 index 1 |
| 10 | 端到端 | Add `success=True`；Search 返回 **4 条**（含两对重复内容——这是脚本第 6 段手工写入 + 第 10 段端到端写入叠加的正常现象，不是 bug） |
| 11 | SQLite 检查 | `raw_chunks` 1 行、`memory_units` 2 行 |

最后必须出现：`全部检查通过。`

结尾的 `Exception ignored ... QdrantClient.__del__ ... sys.meta_path is None` 是 Python 退出时 qdrant 本地客户端的清理噪音，**可以忽略**。

---

## 4. 启动 HTTP 服务 + 手动调接口

```powershell
venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 9001
```

预期输出：

```text
INFO:     Uvicorn running on http://127.0.0.1:9001 (Press CTRL+C to quit)
```

保持这个终端开着，**另开一个终端**测试。注意：浏览器直接访问 `http://127.0.0.1:9001/` 会看到 `{"detail":"Not Found"}`，这是正常的——根路径没有路由，接口只有 `/health` `/add` `/search` 三个。

新终端里先设编码，然后依次执行：

```powershell
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# 健康检查
curl.exe http://127.0.0.1:9001/health

# 写入记忆
curl.exe -s -X POST http://127.0.0.1:9001/add -H "Content-Type: application/json" -d '{"request_id":"manual-001","messages":[{"role":"user","content":"我喜欢燕麦拿铁，不要加糖。","timestamp":1704067200000}],"user_id":"manual-user","session_id":"manual-session"}'

# 检索记忆
curl.exe -s -X POST http://127.0.0.1:9001/search -H "Content-Type: application/json" -d '{"query":"我喜欢喝什么？","user_id":"manual-user","top_k":100}'
```

**比对标准**：

- health → `{"status":"ok"}`
- add → `{"success":true,"request_id":"manual-001",...}`
- search → `data[0].content` 是归档后的事实（语义等价即可，LLM 措辞每次会变），`score` 数值型，`created_at` 为 `2024-01-01T00:00:00+00:00`（来自时间戳，不是当天日期）
- 把 `user_id` 换成任意其他值再查 → `"data":[]`（隔离验证）
- 重复执行同样的 add（相同 request_id）→ 仍返回 success，且不产生新记忆（幂等验证）

---

## 5. LoCoMo 本地评测（真实链路，第 1 段对话约 10 分钟，联网）

```powershell
venv\Scripts\python.exe bench\run_locomo_eval.py --limit 1
```

存储隔离于 `bench.db` / `qdrant_bench`。第一阶段 21 个块写入（每块 LLM 归档，约 3-8 秒，偶尔有 20-30 秒的慢调用属正常重试），第二阶段 138 题检索（每题约 2.5-4 秒）。

**预期进度输出样例**：

```text
conversations=1  questions_total=1382
=== conversation conv-26 ===
  add 1/21 ok in 6.7s  (messages=20)
  ...
  memory units indexed: 270
  search 10/138  hit@k=1.000  cov=1.000  mrr=0.327  (last 2.7s)
  ...
=== summary ===
questions=137.0  chunks=21  skipped_empty_msgs=0
hit@100      = 1.0000
coverage@k = 1.0000
mrr        = 0.5286
```

**比对标准**（LLM 有随机性，按区间比对，不比对精确值）：

| 项 | 达标范围 | 说明 |
|---|---|---|
| 有效题数 | 137 | 138 题中 1 题无金标证据被剔除 |
| hit@100 | **= 1.0000** | 结构性指标，低于 1 说明有回归 |
| coverage@k | **= 1.0000** | 同上 |
| MRR | 0.45 - 0.60 | 每次因归档措辞不同而浮动 |
| memory units | 约 250-300 | 归档条数随 LLM 输出浮动 |

类目行 `cat 1/2/3/4` 中，cat 3 的 MRR 历史最低（约 0.28），这是已知短板不是异常。

**结果文件比对**：运行结束生成 `bench_results.json`，打开后看顶层 `summary` 块与上表一致；`results` 数组里每道题有 `hit / coverage / mrr / top5` 字段，`top5` 是检索到的前 5 条事实原文，可用于人工抽查排序质量。

全量评测（10 段对话）命令为 `--limit 10`，约 1.5 小时，验证过单段后再跑。

---

## 6. 常见问题对照表

| 现象 | 原因 | 处理 |
|---|---|---|
| `ModuleNotFoundError: No module named 'app'` | 解释器/工作目录不对 | 在项目根目录执行；确认用的是 `venv\Scripts\python.exe` |
| Search 返回 `{"detail":"'QdrantClient' object has no attribute 'search'"}` | 跑的是旧代码进程 | 重启 uvicorn |
| `Storage folder ... is already accessed by another instance` | 本地 Qdrant 文件锁被占用 | 关掉占用该存储目录的其他进程（服务与脚本使用了不同目录，正常不会撞） |
| 中文输出乱码 | PowerShell 默认 GBK | 先执行 `[Console]::OutputEncoding = [System.Text.Encoding]::UTF8` |
| SiliconFlow 401/超时 | Key 失效或网络问题 | 检查 `.env` 的 `SiliconFlow_api_key`；脚本会自动重试 3 次 |
| 端口 9001 被占用 | 上一个服务未关 | `netstat -ano | findstr :9001` 找 PID，换端口或结束进程 |

---

## 7. 每次改代码后的推荐验证顺序

1. `pytest tests\test_smoke.py -v` → 2 passed（秒级，结构回归）
2. `tests\step_check.py` → 全部检查通过（分钟级，各函数行为）
3. `bench\run_locomo_eval.py --limit 1` → hit@100=1.0 且 MRR 不低于上次的 85%（10 分钟，端到端质量）
4. 把结果与过程记入 `log.md`
