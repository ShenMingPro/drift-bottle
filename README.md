# The Player - 移动端漂流瓶应用

## 项目简介

The Player 是一个适配手机浏览器的漂流瓶 Web 应用。用户注册后可以"写纸团"和"拾纸团"，查看历史记录并与其他用户互动交流。核心用户、纸团、聊天和广场数据存储在 MySQL 中；大学数据和广场上传的媒体文件存储在本地文件系统。

## 技术栈

- **后端**: Python 3.10+ + FastAPI + Uvicorn + MySQL（aiomysql 异步驱动）
- **前端**: Vue 3 + Vant 4 (CDN 方式集成)
- **AI/工作流**: LangChain + LangGraph + DeepSeek API
- **样式**: 响应式、移动优先

## 从哪里开始理解 LangGraph

先读 [LangGraph 学习导读](LANGGRAPH_LEARNING.md)，再在项目根目录运行离线演示：

```powershell
.\.venv\Scripts\python.exe .\scripts\learn_pick_workflow.py
```

演示运行项目真实状态图，以固定示例代替模型和数据库，展示状态传递、失败终止及人工审批恢复，无需先启动服务器。

## 安装与运行

### 1. 创建 Python 环境并安装依赖

```powershell
cd "D:\Trae项目\new_piao-liu-ping"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

如果 PowerShell 禁止激活脚本，可以在当前终端临时执行 `Set-ExecutionPolicy -Scope Process Bypass`。

### 2. 配置数据库连接

项目根目录需要有 `.env` 文件，并填写 MySQL 连接信息：

```
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=你的MySQL密码
DB_NAME=paper_bottle
PORT=3000
```

可以先复制 `.env.example` 作为模板；不要把真实的 `.env` 提交到 Git。

### 3. 初始化数据库

```powershell
python init_db.py
```

此脚本会创建 `paper_bottle` 数据库以及 `users`、`bottles`、`pick_records`、`pick_commits`、`conversations`、`messages` 核心表。`pick_commits` 保存匹配确认的业务提交凭据，以用户和请求标识建立联合主键。已有数据库也会在应用启动时通过 `CREATE TABLE IF NOT EXISTS` 补齐该表，保留原有记录；启动账号需要有此表的建表权限。

如果是从旧的“12 位数字账号”版本切换到用户名版本，并且确定可以删除所有现有用户及其业务数据，可执行：

```powershell
python reset_user_database.py --database paper_bottle --yes
```

该命令会删除用户、纸团、拾取记录、聊天、广场动态以及 MySQL 中的旧 LangGraph 运行状态，不可用于需要保留数据的环境。聊天 AI 辅助新增的 SQLite 检查点独立存储，不会被此脚本清除。

广场功能还需要创建动态、点赞和评论表。首次部署时执行：

```powershell
Get-Content .\seed_sql\200_plaza.sql | mysql -u root -p paper_bottle
```

然后输入 MySQL 密码。命令中的 `root` 应替换为 `.env` 中配置的 `DB_USER`，并确保该账号具有建表权限。`200_plaza.sql` 当前固定使用 `paper_bottle` 数据库；如果修改了 `DB_NAME`，需要先将该 SQL 文件中的 `USE paper_bottle` 改为对应数据库名。

`seed_data.sql` 和编号的用户/纸团 SQL 文件用于导入演示数据；`seed_sql/99_verify.sql` 只用于查询验证。`seed_sql/00_cleanup.sql` 会清空核心表并重置部分数据，禁止在生产数据库执行。

### 4. 启动服务器

```powershell
python -m uvicorn app.main:app --host 0.0.0.0 --port 3000 --reload
```

也可以直接运行：

```powershell
python -m app.main
```

### 5. 访问应用

打开浏览器访问 [http://localhost:3000](http://localhost:3000)

API 文档地址为 [http://localhost:3000/docs](http://localhost:3000/docs)。建议使用手机浏览器或 Chrome DevTools 的设备模拟器测试。

## 功能说明

### 注册页
- 2~20 位唯一用户名，可使用中文、字母、数字和下划线
- 密码长度为 8~128 位，不强制字母与数字组合，服务端使用 Argon2id 哈希存储
- 注册后用户名同时作为社交场景的昵称，无需重复填写
- 其他信息包括联系方式、性别、年龄（10~100）和爱好
- 三级联动地址选择器：省份 → 城市 → 大学（当前包含 31 个省级地区、347 个城市和 2919 条高校数据）

### 主页面
- 顶部显示"The Player · 昵称"
- "写纸团盲盒"和"拾纸团盲盒"两个主要操作按钮
- 底部退出登录按钮

### 写纸团
- 弹窗输入内容（最多200字）
- 有取消按钮，提交后 Toast 提示"投放成功"
- 常规写纸团在服务端原子扣减写纸团次数；`useBonus=true` 表示使用当日写纸团奖励，不扣减常规次数，并由服务端校验每日上限
- AI 生成按当前用户保留本页面最近展示的 30 条建议，生成时排除重复及近似重复；刷新网页后历史清空
- 页面通过 POST `/api/content-gen/stream` 发送 `userId` 和 `previousSuggestions` JSON，流不可用时回退 POST `/api/content-gen`；原 GET 接口保留兼容，历史最多 30 条、每条最多 200 字
- 模型输出重复或格式无效时最多尝试三次，仍失败则提示重试，不使用固定文案补齐结果
- AI 生成的每条建议都携带服务端签发的短期内容凭证；用户原样发布时跳过耗时的 LLM 审核
- 用户手写内容，或对 AI 建议做任何手动编辑后，凭证立即失效并恢复 `ContentGuardAgent` 审核
- AI 建议在签发凭证前会经过不调用 LLM 的敏感词、联系方式和垃圾文本规则检查

### 拾纸团
- 加载动画至少 0.5 秒
- 按学校、城市、省份、全国逐级查找候选纸团，再按用户画像评分选择最佳匹配；同一用户的多条纸团会随机选取一条
- 展示纸团内容 + 发布者信息（昵称、性别、年龄、爱好标签、地址）
- 匹配成功后由服务端扣减拾纸团次数；智能完整工作流只在用户确认后才扣减

### 拾纸团流程
页面恢复使用旧版单步接口 `/api/pick`：Supervisor 只调度 MatchAgent，MatchAgent 按学校、城市、省份、全国逐级查找并评分，成功后由路由写入 `pick_records`，再返回扁平的匹配结果。页面不会在拾取时调用画像分析、对话引导、破冰或内容审核 Agent。

`/api/pick/full` 仍作为兼容接口保留，供需要完整多 Agent 协作结果的调用方使用，但不是页面的默认入口。完整流程会依次调度匹配、画像分析、对话引导、破冰和内容审核；`write_smart` 与 `chat_rescue` 还可以按需调度 MCP Research Agent。

旧版单步拾纸团的请求、消息和 JSON 流程记录在 [旧版拾纸团流程说明.md](旧版拾纸团流程说明.md)，当前页面和 `/api/pick` 路由按该流程运行。

### Agent 与工具调用失败

Supervisor 将调用异常、无效结果与真正的“没有候选”分别处理。子 Agent 失败后不会再汇总成 `success`；已知业务必须具有实际执行的必要 Agent 结果，模型自行输出的最终 JSON 不能代替这些结果。旧拾取接口遇到技术故障返回错误，不再显示“没有纸团”；写纸团、发送消息及匹配流程失败时不会进入扣次或业务写入。

`.env` 中的 `AGENT_CALL_TIMEOUT_SECONDS` 默认为 60 秒，用于 Supervisor 每次模型/子 Agent 调用、显式拾取工作流每次工具调用、MCP 调用及各模型客户端。`AGENT_MAX_ATTEMPTS` 默认为 2（含首次，可设为 1～3），用于 Supervisor 模型和已知只读子 Agent 的临时网络故障重试；重试占用工具步骤预算。参数或结果格式错误直接失败。模型 SDK 的额外重试已关闭，内容生成原有的三次输出质量重试仍保留。

MCP 服务的 `isError` 和调用异常会保留为失败状态，AI Chat 随即停止，不把错误文本当作研究证据。MCP 工具最多调用 8 次；未知工具可能具有副作用，因此不会自动重放工具或整个 Research Agent。技术失败通过普通接口的 502/503/504 或 SSE `error` 反馈，不向客户端展示供应商原始异常。

画像、对话引导、破冰和审核的无效模型输出会明确报错，不再用固定文案冒充成功。审核服务异常与正常的内容拒绝分别处理；审核字段必须为真正的布尔值。破冰请求只有在输出校验通过后才记为完成，失败后可用同一 `requestId` 恢复并重新生成。

### 可恢复的完整拾取工作流

新增的 `/api/pick/runs` 使用显式 LangGraph 工作流，不再让 Supervisor 通过提示词决定每一步：

```text
match
  → profile_analysis
  → conversation_guide
  → ice_break
  → content_guard
  → interrupt（等待用户确认）
  → approved / canceled
```

工作流使用独立的 `AsyncSqliteSaver` 保存状态，默认文件为 `data/pick_workflow_checkpoints.sqlite3`，可通过 `PICK_WORKFLOW_CHECKPOINT_PATH` 修改。服务重启后可以使用同一个 `runId` 查询状态并继续审批；审批时只能选择已经通过 ContentGuard 的破冰建议。

MySQL 业务提交由统一的匹配提交模块完成：在短事务内锁定当前用户行，写入提交凭据、扣减次数并保存拾取记录；任一步失败全部回滚。`runId` 用于同一审批请求的幂等提交，新旧匹配接口均经过该模块。不同请求命中同一纸团时复用原记录、不再扣次数；命中已拾取作者的其他纸团时返回 409，要求重新匹配，与 MatchAgent 排除已匹配作者的规则一致。历史拾取记录不会被删除或合并。

`status=approved` 只表示用户已经确认，业务是否完成以 `pickRecorded` 与 `commitStatus` 为准：`not_requested` 为尚未请求提交，`pending` 为已确认但尚无 MySQL 业务记录，`committed` 为业务已完成。升级前的审批没有新提交凭据，查询会按当前用户与该纸团精确读取历史拾取记录，避免误报待提交或重新扣次。如果图确认后写库失败或服务中断，查询仍显示待提交；以同一 `runId` 重试确认只重试业务提交，不重新调用模型。关闭审批窗口不会触发待提交业务的重试。

MySQL 提交中的用户行锁和请求主键可协调不同进程的数据库写入；SQLite 工作流审批仍依赖进程内锁，应用当前按单进程运行。旧版 `/api/pick` 与 `/api/pick/full` 每次调用仍执行一次匹配，未提供基于客户端请求编号的整次 HTTP 请求结果复用；同瓶去重不等同于这些旧接口的完整请求幂等。

创建任务：

```http
POST /api/pick/runs
Content-Type: application/json
Authorization: Bearer <token>

{"userId": 7, "university": "测试大学", "city": "测试市", "province": "测试省"}
```

返回 `status=awaiting_approval` 后，可以调用：

```http
GET  /api/pick/runs/{runId}
POST /api/pick/runs/{runId}/resume

{"userId": 7, "approved": true, "hint": "你也喜欢摄影吗"}
```

旧接口 `/api/pick` 和 `/api/pick/full` 保留兼容，便于逐步迁移前端。

页面主界面新增“智能全流程”入口：启动后展示匹配结果和已通过审核的破冰建议，用户选择一条后调用 resume 完成流程。原有的“拾纸团盲盒”入口继续使用旧接口，方便对比两种实现。

### 轻量级检索与排序基线

MatchAgent 现在除了地理位置、性别、年龄和爱好评分，还会计算纸团文本与用户兴趣的相关性，并计入总分。实现位于 `app/services/retrieval.py`，使用确定性的字符 n-gram 特征哈希和余弦相似度，不依赖外部 embedding 服务。

写纸团的 AI 建议也不再只取最近 20 条纸团，而是先按用户兴趣对近期话题做检索排序，再放入生成 Prompt；这样检索层同时服务于匹配和内容生成。

`app/evals/ranking.py` 提供 NDCG@K、MRR 和 Recall@K 指标，`app/evals/matching.py` 可以直接对一组 query/relevant_ids/candidates 样例计算排序报告。当前检索实现是可替换的基线，后续可以切换到 pgvector、Qdrant 或真实 embedding provider，而不改变匹配业务层的输入输出。

运行内置样例和 CI 质量门槛：

```powershell
python -m app.evals.cli --min-ndcg 0.8 --min-mrr 0.8 --min-recall 0.8
```

样例数据在 `evals/matching_cases.json`。当前样例规模很小，只用于验证评测链路和回归门槛，不应把 1.0 的报告当作真实业务效果。

仓库已配置 GitHub Actions：每次 push/PR 会安装 `requirements-dev.txt`、执行 Python 编译和全量测试、检查前端 JavaScript 语法，并运行匹配质量门槛。

真实 MySQL 集成测试默认跳过。可以启动专用测试数据库：

```powershell
docker compose -f docker-compose.test.yml up -d
$env:RUN_MYSQL_INTEGRATION="1"
$env:DB_HOST="127.0.0.1"
$env:DB_PORT="3307"
$env:DB_USER="root"
$env:DB_PASSWORD="test-root"
$env:DB_NAME="paper_bottle_test_it"
python -m pytest tests/integration -q
```

测试会创建、清空并删除名称包含 `test` 的专用数据库，不能把 `DB_NAME` 指向业务数据库。CI 中也配置了独立的 MySQL service 自动执行这组测试。

### 人与人聊天 AI 辅助与检查点

人与人聊天页面的 AI 辅助（IceBreakAgent）采用 LangGraph `AsyncSqliteSaver` 保存近期建议和工作流状态。默认文件为项目根目录下的 `data/ice_break_checkpoints.sqlite3`；可在 `.env` 中设置 `ICE_BREAK_CHECKPOINT_PATH`，相对路径始终相对于项目根目录解析。应用启动时创建存储并初始化检查点表，关闭时释放 SQLite 连接；初始化失败会使启动失败，不会默默切换到内存存储。

普通 GET 接口 `/api/ice-break` 和 SSE 接口 `/api/ice-break/stream` 都接受可选的 `requestId` 查询参数，格式为 UUID。每次点击“AI 辅助”生成新的 `requestId`，失败后重试同一次生成复用原值；不传时由服务端生成。服务端用经过认证的当前用户 ID 与 `targetUserId` 共同生成稳定的 `thread_id`，按“谁向谁请求建议”的方向隔离状态，不接受客户端直接指定其他用户的检查点。

双方实际发送的聊天记录仍保存在 MySQL，AI 辅助每个新请求都通过 `fetch_context` 获取最新聊天内容；检查点保存的近期建议可供后续生成使用，减少连续请求重复给出同样话题。失败后使用同一 `requestId` 可以继续未完成的流程，已完成请求的重试复用原结果。状态可跨服务重启保存。同一方向会话的请求在一个服务进程内串行处理，避免状态覆盖；当前 SQLite 与进程内锁适用于单进程运行，不提供多 worker 间的会话锁。

独立的热点 AI 助手 `/api/ai-chat` 不使用检查点。检查点保留的是人与人聊天 AI 辅助的会话状态，不是跨所有会话的用户长期记忆，也不会训练模型。近期建议保留最近 9 条不同文本，已完成结果保留最近 10 次供同请求重试；新请求会重新读取最新聊天，不会沿用旧聊天快照。SSE 断流或可重试错误时，前端最多以同一请求编号回退一次普通接口；已成功的请求复用结果，未完成请求恢复执行。未完成的节点在恢复时可能重新执行，不能据此保证外部操作的副作用只发生一次。SQLite 文件及其附属文件已加入 `.gitignore`；备份应用数据时应同时备份此文件。

聊天消息现在始终经过 ContentGuard，即使内容来自 AI 破冰建议；前端不再通过 `isAiRecommended` 布尔字段跳过审核。发送和读取消息都会校验当前用户是否是会话成员。

## 技术亮点

- **混合编排**：开放式的 Supervisor/ReAct 与确定性的 LangGraph Pipeline 并存，固定业务流程不交给模型自由规划。
- **节点协作**：画像分析和对话引导将结论写入共享状态，破冰节点读取兴趣标签、契合度和沟通策略作为生成上下文，让前序分析实际参与后续生成。
- **可恢复工作流**：拾取流程使用 `AsyncSqliteSaver`、`interrupt` 和 `Command(resume=...)` 支持人工审批和跨服务重启恢复。
- **运行时契约**：Match、Profile、ConversationGuide、IceBreak 的结果在进入图状态前通过 Pydantic `WorkflowArtifact` 校验，非法字段不会继续传播。
- **可替换检索层**：匹配包含文本相关性分数，并提供 NDCG@K、MRR、Recall@K 指标；当前使用无外部依赖的确定性基线。
- **安全边界**：会话成员校验、统一 ContentGuard、AI 建议进入审批前逐条审核、模型上下文默认不发送真实联系方式。
- **服务端次数**：写纸团和拾纸团在服务端原子扣减，签到、邀请和奖励由服务端增加；客户端只读取最新状态，`/api/update-chances` 已废弃。
- **事务与并发控制**：匹配确认的扣次、记录和请求凭据共同提交；发纸团的额度／奖励与纸团写入共同提交；发送消息与会话更新时间共同提交。签到、邀请和写奖励通过数据库用户行锁控制每日次数，日期统一使用同一次数据库读取的业务日期。
- **短连接作用域**：匹配、发纸团和消息审核期间不持有请求级 MySQL 连接；调用模型前的查询结束即释放连接，模型完成后再借连接执行短事务。
- **流式体验**：SSE 推送节点进度和文本，AI Chat 只执行一次图，避免重复调用模型或 MCP 工具。
- **可验证**：覆盖工作流暂停/恢复、SQLite 重启恢复、越权聊天、审核绕过、SSE 执行次数等场景。

## API 接口

除注册、登录和大学数据接口外，大部分接口需要在请求头中携带 JWT：

```http
Authorization: Bearer <token>
```

| 模块 | 方法 | 接口 | 说明 |
|------|------|------|------|
| 认证 | POST | `/api/register` | 注册并返回 JWT |
| 认证 | POST | `/api/login` | 登录并返回 JWT |
| 纸团 | GET | `/api/universities` | 获取大学数据 |
| 纸团 | POST | `/api/write` | 写纸团 |
| 纸团 | GET | `/api/pick` | 旧版单步匹配并返回扁平拾取结果 |
| 纸团 | GET | `/api/pick/full` | 兼容接口：Supervisor 协调匹配、画像分析、对话引导、破冰和审核 |
| 纸团 | POST | `/api/pick/runs` | 创建可恢复的完整拾取工作流 |
| 纸团 | GET | `/api/pick/runs/{runId}` | 查询工作流状态和待审批内容 |
| 纸团 | POST | `/api/pick/runs/{runId}/resume` | 确认或取消工作流并继续执行 |
| 历史 | GET | `/api/history` | 查询写入和拾取历史 |
| 聊天 | POST | `/api/conversations` | 创建会话 |
| 聊天 | GET | `/api/conversations` | 查询会话列表 |
| 聊天 | POST | `/api/messages` | 发送消息 |
| 聊天 | GET | `/api/messages` | 查询消息 |
| 任务 | GET | `/api/user-chances` | 查询剩余次数 |
| 任务 | GET | `/api/checkin-status` | 查询签到、邀请和奖励状态 |
| 任务 | POST | `/api/checkin` | 每日签到 |
| 任务 | POST | `/api/invite` | 邀请好友并领取次数 |
| 任务 | POST | `/api/write-bonus` | 领取写纸团奖励 |
| 任务 | POST | `/api/update-chances` | 已废弃：次数由服务端管理，返回 410 |
| AI | GET | `/api/content-gen`、`/api/content-gen/stream` | 生成纸团内容建议（普通/SSE） |
| AI | GET | `/api/ice-break`、`/api/ice-break/stream` | 生成破冰建议（普通/SSE） |
| AI | POST/GET | `/api/ai-chat`、`/api/ai-chat/stream` | AI 聊天（普通/SSE） |
| AI | GET | `/api/ai-chat/tools` | 查询可用 MCP 工具 |
| 广场 | GET/POST | `/api/plaza/posts` | 动态列表和发布 |
| 广场 | POST | `/api/plaza/upload` | 上传图片或视频 |
| 广场 | GET/POST | `/api/plaza/posts/{post_id}/comments` | 查询或发表评论 |
| 广场 | POST | `/api/plaza/posts/{post_id}/like` | 点赞/取消点赞 |

完整的请求参数和响应结构以 `/docs` 中的 OpenAPI 文档为准。

## 数据库结构

- 数据库名：`paper_bottle`，字符集：`utf8mb4`
- `users` 表：唯一用户名、Argon2id 密码哈希、用户基本信息和可用次数（JWT 本身不存储在表中）
- `bottles` 表：纸团内容（关联用户 ID）
- `pick_records` 表：拾取记录
- `conversations`、`messages` 表：聊天会话和消息
- `posts`、`post_likes`、`post_comments` 表：广场动态、点赞和评论（由 `seed_sql/200_plaza.sql` 创建）

## 项目目录结构

```
.
├── app/
│   ├── main.py              # FastAPI 应用入口
│   ├── config.py            # 环境变量配置
│   ├── database.py          # MySQL 连接池
│   ├── ice_break_checkpoint.py # 人与人聊天 AI 辅助 SQLite 检查点与会话锁
│   ├── pick_checkpoint.py   # 可恢复拾取工作流检查点
│   ├── workflows/           # 显式 LangGraph 工作流
│   │   └── pick_workflow.py # 匹配→画像→引导→破冰→审核→审批
│   ├── services/            # 次数、检索等业务服务
│   ├── evals/               # 排序评测指标
│   ├── auth_jwt.py          # JWT 认证
│   ├── agents/              # AI/工作流 Agent
│   ├── models/              # Pydantic 数据模型
│   └── routers/             # API 路由
├── data/universities.json   # 大学数据
├── evals/matching_cases.json # 匹配排序评测样例
├── public/index.html        # Vue 3 + Vant 4 前端页面
├── public/uploads/          # 广场上传文件
├── seed_sql/200_plaza.sql   # 广场表结构
├── init_db.py               # 核心数据库初始化脚本
├── requirements.txt         # Python 依赖
├── requirements-dev.txt     # 测试和 CI 依赖
├── .env                     # 本地环境变量（自行配置）
└── README.md
```

## 可选配置

写纸团内容审核、拾纸团匹配、内容建议、破冰建议和 AI 聊天等智能流程都依赖 `.env` 中的 `DEEPSEEK_API_KEY`；未配置时这些流程可能无法正常执行。MCP 服务需要 Node.js/npm 和网络访问；本地不使用联网工具时，可以将 `MCP_TRENDS_HUB_ENABLED`、`MCP_SEARCH1API_ENABLED` 都设置为 `false`。拾取工作流检查点可配置 `PICK_WORKFLOW_CHECKPOINT_PATH`。不要将包含真实密钥和密码的 `.env` 提交到仓库。

## 常见问题

### MySQL 连接失败

确认 MySQL 服务已启动，并检查 `.env` 中的主机、端口、用户名、密码和数据库名。

### 端口 3000 被占用

启动时指定其他端口，例如：

```powershell
python -m uvicorn app.main:app --host 0.0.0.0 --port 3001 --reload
```
