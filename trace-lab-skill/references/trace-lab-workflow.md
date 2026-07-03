# TRACE Lab 工作流参考

## 核心逻辑

TRACE Lab 的 skill 操作逻辑不是单次 API 调用，而是实验协作闭环：

1. 实验人员给出 CSV 输入。
2. agent 根据 CSV 和项目状态调用 TRACE Lab。
3. TRACE Lab 生成候选推荐。
4. agent 用 `save_recommendations.py` 把推荐整理成 CSV，保存到 `trace-collab-skill/output/`。
5. agent 在回复中展示推荐表格。
6. 用户要求第二轮及以上推荐时，agent 先引导实验人员逐条补充上一轮实验结果。
7. agent 把用户自然语言回复解析为 outcome JSON，并用 `save_experiment_results.py` 自动填写结果 CSV。
8. agent 用结果 CSV 提交 `tell`，再生成下一轮推荐。

每轮推荐都必须产生两类输出：

- 文件输出：`trace-collab-skill/output/{project_id}_{round_id}_recommendations.csv`
- 回复输出：Markdown 推荐表格，表头为 `rank`、candidate 变量列和 `rationale`

用户可见 Markdown 表格不展示 `recommendation_id` 和 `batch_role`；这两个字段仍保存在推荐 CSV 中，用于后续结果回填、追踪和 `tell` 提交。

## CSV 输入

### 设计空间 CSV

用于创建新项目，常见列：

```text
variable,type,value,low,high,step,unit,stage,active,role,fixed_value
```

支持变量类型：

- `categorical`
- `discrete_numeric`
- `continuous`
- `fixed`

descriptor 列可写成 `descriptor__<name>`。

### 历史实验 CSV

用于第一次推荐前导入已有实验记录。至少应包含实验变量和目标值，例如：

```text
Catalyst,Solvent,Temperature,yield,status
cat_a,MeOH,60,55.2,completed
```

### 实验结果 CSV

用于实验人员完成推荐实验后回填结果。至少包含：

```text
recommendation_id,status,yield,failure_reason,notes
```

有效状态：

- `completed`：需要目标值，例如 `yield`。
- `failed`：记录失败原因，不需要目标值。
- `skipped`：记录跳过原因，不需要目标值。
- `pending`：仍未完成，不是终态。

## 项目目录结构

TRACE Lab 的每个项目都位于 `runs/lab_projects/{project_id}`。

核心文件：

- `project.yaml`：项目目标、planner、controller mode、批次大小、证据文件名等配置。
- `design_space.csv`：实验变量、候选值、连续范围、固定条件等设计空间。
- `observations.csv`：历史实验结果和推荐实验完成后的真实观测结果。
- `evidence_cards.jsonl`：文献或领域知识证据卡。
- `recommendations_round_*.json/csv`：TRACE Lab 原始推荐文件。
- `trace_round_*.jsonl`：推荐决策、证据引用、代理动作、实验结果反思等审计轨迹。

skill 自己生成的推荐表格 CSV 固定保存到：

```text
trace-collab-skill/output/
```

## 运行环境

项目根目录下的 `.env` 通常包含：

```env
OPENAI_API_KEY=
OPENAI_BASE_URL=
TRACE_LAB_PROJECTS_ROOT=runs/lab_projects
TRACE_LAB_HOST=127.0.0.1
TRACE_LAB_PORT=8788
```

不要打印密钥值。检查环境时只报告这些变量是否已设置。

`agentic` 模式需要 `OPENAI_API_KEY`。如果使用 OpenAI 兼容中转接口，还需要可用的 `OPENAI_BASE_URL`。模型配置主要在 `configs/agent_bo.yaml` 的 `runtime.model_name` 和 `runtime.llm_fallback_model_name`。

## 启动和检查

Docker 构建：

```powershell
docker build -t trace-collab-lab .
```

使用 `.env` 启动服务：

```powershell
docker rm -f trace-collab-lab-run
docker run -d --name trace-collab-lab-run -p 8788:8788 --env-file .env trace-collab-lab
```

健康检查：

```powershell
python trace-collab-skill\scripts\trace_lab_api.py health --pretty
```

项目摘要：

```powershell
python trace-collab-skill\scripts\trace_lab_api.py summary oxidative_esterification_demo --pretty
```

## 第一轮推荐

如果用户提供了新的设计空间 CSV，先创建项目：

```powershell
python trace-collab-skill\scripts\trace_lab_api.py create my_project `
  --design-csv design_space.csv `
  --objective-name yield `
  --planner-name atlas `
  --controller-mode agentic `
  --batch-size 6 `
  --pretty
```

如果用户提供了历史实验 CSV，先导入：

```powershell
python trace-collab-skill\scripts\trace_lab_api.py import-observations my_project `
  --rows-csv historical.csv `
  --source historical `
  --pretty
```

生成候选推荐：

```powershell
python trace-collab-skill\scripts\trace_lab_api.py ask my_project `
  --batch-size 6 `
  --planner-name atlas `
  --controller-mode agentic `
  --format json `
  --pretty > ask_response.json
```

整理推荐并输出表格：

```powershell
python trace-collab-skill\scripts\save_recommendations.py ask_response.json
```

agent 回复必须展示该脚本打印的 Markdown 表格，并说明 CSV 保存路径。Markdown 表格展示 `rank`、candidate 变量列和 `rationale`；不要在用户可见表格中展示 `recommendation_id` 或 `batch_role`。

## 实验人员回填结果

实验人员根据推荐表格完成实验后，可以直接给自然语言回复；agent 负责自动整理成结果 CSV。用户可以按 rank 或 `recommendation_id` 回答，例如：

```text
1号 completed，yield 72.5，备注 clean run；
2号 failed，原因 precipitation；
3号 skipped，原因 not enough material。
```

agent 应先把用户回复解析为 outcome JSON：

```json
[
  {"rank": 1, "status": "completed", "yield": 72.5, "notes": "clean run"},
  {"rank": 2, "status": "failed", "failure_reason": "precipitation"},
  {"rank": 3, "status": "skipped", "failure_reason": "not enough material"}
]
```

保存为：

```text
trace-collab-skill/output/{project_id}_{round_id}_outcomes.json
```

再运行：

```powershell
python trace-collab-skill\scripts\save_experiment_results.py `
  --recommendations-csv trace-collab-skill\output\{project_id}_{round_id}_recommendations.csv `
  --outcomes-json trace-collab-skill\output\{project_id}_{round_id}_outcomes.json
```

脚本会生成：

```text
trace-collab-skill/output/{project_id}_{round_id}_results.csv
```

结果 CSV 结构为：

```text
recommendation_id,status,yield,failure_reason,notes
round_001_rec_001,completed,72.5,,clean run
round_001_rec_002,failed,,precipitation,
round_001_rec_003,skipped,,not enough material,
```

如果用户回复缺少某条推荐、`completed` 缺少 `yield`、或状态不明确，agent 必须继续追问，不能提交 `tell`。

提交结果：

```powershell
python trace-collab-skill\scripts\trace_lab_api.py tell my_project `
  --results-csv trace-collab-skill\output\{project_id}_{round_id}_results.csv `
  --defer-reflection `
  --format both `
  --pretty
```

提交后检查：

- `appended`
- `observation_count`
- `completed_observation_count`
- `best_so_far`
- `reflection_status`
- 是否还有 pending recommendation

## 下一轮推荐

当最新批次没有 pending，且上一轮结果 CSV 已经通过 `tell` 提交后，才能生成下一轮：

如果用户只提出“继续下一轮”“第二轮推荐”但没有给出上一轮实验结果，agent 必须先回到“实验人员回填结果”步骤，引导用户逐条补充状态、yield、失败原因或备注。不能跳过 `save_experiment_results.py` 和 `tell`。

```powershell
python trace-collab-skill\scripts\trace_lab_api.py next my_project `
  --batch-size 6 `
  --planner-name atlas `
  --controller-mode agentic `
  --format json `
  --pretty > next_response.json
```

整理并展示下一轮：

```powershell
python trace-collab-skill\scripts\save_recommendations.py next_response.json
```

agent 回复同样必须展示 Markdown 推荐表格，并说明新的 CSV 保存路径。Markdown 表格展示 `rank`、candidate 变量列和 `rationale`；不要在用户可见表格中展示 `recommendation_id` 或 `batch_role`。

## API 总览

基础地址：`http://127.0.0.1:8788`

- `GET /api/health`：服务健康检查。
- `GET /api/projects`：列出项目摘要。
- `POST /api/projects`：根据配置和 long-format design records 创建项目。
- `GET /api/projects/{project_id}`：读取项目摘要。
- `POST /api/projects/{project_id}/ask`：生成候选推荐。
- `POST /api/projects/{project_id}/tell`：提交实验结果。
- `GET /api/projects/{project_id}/recommendations`：读取推荐批次。
- `GET /api/projects/{project_id}/evidence`：读取 UI 可展示的证据卡。
- `GET /api/projects/{project_id}/observations`：读取观测结果。
- `POST /api/projects/{project_id}/observations/import`：导入历史观测。
- `GET /api/projects/{project_id}/trace/{round_id}`：读取某轮 trace。
- `POST /api/projects/{project_id}/reset`：重置运行状态，同时保留项目配置文件。

## Trace 查看

当用户问“为什么推荐这个候选”时，读取 trace：

```powershell
python trace-collab-skill\scripts\trace_lab_api.py trace my_project round_001 --pretty
```

解释时对应推荐表格中的 `recommendation_id`，说明候选条件、rationale、证据引用和决策路径。

## 常见失败原因

`LLM agent unavailable for stagnation_diagnosis`

- 原因：运行中的进程或容器没有 `OPENAI_API_KEY`，或者 DecisionEngine 无法构建 LLM agents。
- 处理：用 `--env-file .env` 重启 Docker；如果不需要 LLM，使用 `controller_mode=bo_only`。

`Non-retryable LLM failure`

- 原因：模型名错误、结构化输出不被支持、代理/base URL 错误、认证失败、额度问题或 endpoint 不兼容。
- 处理：检查 `configs/agent_bo.yaml` 中的模型名，以及 `.env` 中的 `OPENAI_BASE_URL`；很多兼容接口要求 URL 以 `/v1` 结尾。

Windows 上 `matter-golem` 安装失败

- 原因：C 扩展编译需要 Windows SDK 头文件。
- 处理：使用 Docker/Linux。Windows 本地安装可以通过 platform marker 跳过。

`ModuleNotFoundError: olympus`

- 原因：本地第三方源码路径没有放进 `PYTHONPATH`。
- PowerShell 处理方式：

```powershell
$env:PYTHONPATH = "$PWD\third_party\atlas\src;$PWD\third_party\olympus\src;$env:PYTHONPATH"
```
