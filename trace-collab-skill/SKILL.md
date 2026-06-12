---
name: trace-collab-skill
description: Use when Codex needs to operate TRACE-COLLAB / TRACE Lab chemical experiment recommendation workflows from CSV inputs, generate candidate recommendations, save recommendation CSVs under trace-collab-skill/output, show recommendation tables, submit completed experiment results, continue next-round recommendations, or troubleshoot agentic/bo_only, .env, model/API key, project.yaml, design_space.csv, observations.csv, recommendations, and trace files.
---

# TRACE-COLLAB Skill

本 skill 是 TRACE Lab 的操作技能，用于协助实验人员完成闭环候选推荐：

1. 实验人员提供 CSV 输入。
2. agent 调用 TRACE Lab 生成候选推荐。
3. agent 运行整理脚本，把推荐保存为 CSV 到 `trace-collab-skill/output/`。
4. agent 在回复中展示推荐表格。
5. 用户要求第二轮及以上推荐时，agent 先引导实验人员补充上一轮实验结果。
6. agent 根据用户回复自动填写结果 CSV。
7. agent 提交结果并继续下一轮推荐。

不要把“推荐已生成”作为最终回复；每轮推荐必须展示表格，并说明保存的 CSV 路径。

## 输入 CSV 类型

根据用户给出的 CSV 判断当前阶段：

- `design_space.csv`：用于创建或检查项目设计空间。
- 历史实验 CSV：用于第一次推荐前导入已有实验记录。
- 实验结果 CSV：用于 `tell`，提交实验人员完成后的结果。
- TRACE Lab 推荐 JSON：用于整理推荐输出；如果只有 API 返回结果，先保存为 JSON，再交给 `save_recommendations.py`。
- 用户自然语言实验结果：用于第二轮及以上推荐前，由 agent 解析为 outcome JSON，再交给 `save_experiment_results.py` 自动填写结果 CSV。

实验结果 CSV 至少包含：

```text
recommendation_id,status,yield,failure_reason,notes
```

其中 `completed` 行必须包含目标值，例如 `yield`；`failed`/`skipped` 可以只填写原因或备注。

## 主工作流

### 1. 检查服务和项目

先确认 API 可用：

```powershell
python trace-collab-skill\scripts\trace_lab_api.py health --pretty
```

查看项目摘要：

```powershell
python trace-collab-skill\scripts\trace_lab_api.py summary oxidative_esterification_demo --pretty
```

检查内容：

- 项目是否存在。
- active variables 是否正确。
- 是否已有 observations。
- 是否存在 pending recommendations。
- `controller_mode`、`planner_name`、`batch_size` 是否符合本轮需求。

### 2. 根据 CSV 建立或补充项目

如果用户提供的是新的设计空间 CSV，创建项目：

```powershell
python trace-collab-skill\scripts\trace_lab_api.py create my_project `
  --design-csv design_space.csv `
  --objective-name yield `
  --planner-name atlas `
  --controller-mode agentic `
  --batch-size 6 `
  --pretty
```

如果用户提供的是历史实验 CSV，在第一次推荐前导入：

```powershell
python trace-collab-skill\scripts\trace_lab_api.py import-observations my_project `
  --rows-csv historical.csv `
  --source historical `
  --pretty
```

### 3. 生成候选推荐

没有未处理 pending 推荐时，调用 `ask`：

```powershell
python trace-collab-skill\scripts\trace_lab_api.py ask my_project `
  --batch-size 6 `
  --planner-name atlas `
  --controller-mode agentic `
  --format json `
  --pretty > ask_response.json
```

如果已经完成上一轮实验并提交了结果，调用 `next`：

```powershell
python trace-collab-skill\scripts\trace_lab_api.py next my_project `
  --batch-size 6 `
  --planner-name atlas `
  --controller-mode agentic `
  --format json `
  --pretty > next_response.json
```

`agentic` 模式需要运行环境中有 `OPENAI_API_KEY` 和可用模型配置；无 LLM 或调试时使用 `--controller-mode bo_only`。

如果用户只是说“继续下一轮”“做第二轮推荐”或类似表达，但还没有提供上一轮实验结果，不能直接运行 `next`；必须先执行第 5 步，引导用户补充结果并生成 results CSV。

### 4. 保存推荐 CSV 并展示表格

每次 `ask` 或 `next` 得到推荐 JSON 后，必须运行：

```powershell
python trace-collab-skill\scripts\save_recommendations.py ask_response.json
```

或：

```powershell
python trace-collab-skill\scripts\save_recommendations.py next_response.json
```

整理脚本会输出：

```text
trace-collab-skill/output/{project_id}_{round_id}_recommendations.csv
```

agent 必须把脚本打印出的 Markdown 表格展示给用户。用户可见表格应简洁展示：

- `rank`
- candidate 关键变量和值（每个变量单独成列，例如 `Additive`、`Solvent`、`Temperature`）
- rationale 摘要

用户可见 Markdown 表格不要展示 `recommendation_id` 和 `batch_role`；这些字段仍保存在推荐 CSV 中，用于后续结果回填、追踪和 `tell` 提交。

同时在回复中说明：

- `round_id`
- 推荐 CSV 保存路径
- TRACE Lab 原始 recommendations CSV/JSON 路径，如果 API 返回中包含
- trace 查看方式

### 5. 第二轮及以上推荐前引导用户补充结果

当用户要求第二轮及以上推荐时，不能直接运行 `next`。先找到上一轮推荐 CSV：

```text
trace-collab-skill/output/{project_id}_{round_id}_recommendations.csv
```

然后把上一轮推荐表展示给用户，并要求用户逐条回复实验结果。引导问题使用这个格式：

```text
请补充上一轮每条推荐的实验结果：
- rank 1 / round_001_rec_001：状态 completed/failed/skipped？如果 completed，请给 yield；如果 failed/skipped，请给原因。
- rank 2 / round_001_rec_002：...
- rank 3 / round_001_rec_003：...
```

有效状态：

- `completed`：需要数值型目标值，例如 `yield`。
- `failed`：记录失败原因，不需要目标值。
- `skipped`：记录跳过原因，不需要目标值。
- `pending`：允许保留，但不是终态；仍有 pending 时默认不能进入下一轮。

收到用户自然语言回复后，agent 必须自动解析为 outcome JSON。允许用户用 rank 或 `recommendation_id` 表达，例如“1号 72.5，2号失败沉淀，3号跳过原料不足”。解析后的 JSON 示例：

```json
[
  {"rank": 1, "status": "completed", "yield": 72.5, "notes": "clean run"},
  {"rank": 2, "status": "failed", "failure_reason": "precipitation"},
  {"rank": 3, "status": "skipped", "failure_reason": "not enough material"}
]
```

把 JSON 保存为临时文件，例如：

```text
trace-collab-skill/output/{project_id}_{round_id}_outcomes.json
```

然后运行：

```powershell
python trace-collab-skill\scripts\save_experiment_results.py `
  --recommendations-csv trace-collab-skill\output\{project_id}_{round_id}_recommendations.csv `
  --outcomes-json trace-collab-skill\output\{project_id}_{round_id}_outcomes.json
```

该脚本会自动生成：

```text
trace-collab-skill/output/{project_id}_{round_id}_results.csv
```

并打印结果表。agent 必须向用户展示生成的结果表；如果缺少某条推荐结果、`completed` 缺少 `yield`、或状态不明确，先追问用户，不要进入 `tell`。

### 6. 提交结果并继续下一轮

提交实验结果：

```powershell
python trace-collab-skill\scripts\trace_lab_api.py tell my_project `
  --results-csv trace-collab-skill\output\{project_id}_{round_id}_results.csv `
  --defer-reflection `
  --format both `
  --pretty
```

汇报结果时说明：

- `appended` 数量。
- 各状态数量。
- `observation_count`。
- `completed_observation_count`。
- `best_so_far`。
- `reflection_status`。
- 是否还有 pending recommendation。

如果没有 pending，继续执行第 3 步的 `next`，再执行第 4 步保存 CSV 并展示新一轮推荐表格。

## 脚本命令速查

- `health`：检查 API。
- `projects`：列出项目。
- `summary <project_id>`：项目摘要。
- `create <project_id> --design-csv design.csv`：根据设计空间 CSV 创建项目。
- `import-observations <project_id> --rows-csv historical.csv`：导入历史实验。
- `ask <project_id>`：生成第一轮或当前轮候选推荐。
- `next <project_id>`：在上一轮结果完成后生成下一轮候选。
- `save_recommendations.py <json_path>`：把 ask/next 推荐 JSON 整理成 CSV，保存到 `trace-collab-skill/output/`，并打印 Markdown 表格。
- `save_experiment_results.py --recommendations-csv <csv> --outcomes-json <json>`：根据上一轮推荐 CSV 和用户回复解析出的 outcome JSON 自动填写结果 CSV。
- `tell <project_id> --results-csv trace-collab-skill/output/{project_id}_{round_id}_results.csv`：提交由 `save_experiment_results.py` 自动填写的实验结果 CSV。
- `recommendations <project_id>`：读取推荐批次。
- `observations <project_id>`：读取观测结果。
- `evidence <project_id>`：读取可展示证据。
- `trace <project_id> <round_id>`：读取决策 trace。
- `reset <project_id>`：重置运行状态。

## Trace 查看

当用户问“为什么推荐这个候选”或需要解释推荐依据时，读取 trace：

```powershell
python trace-collab-skill\scripts\trace_lab_api.py trace my_project round_001 --pretty
```

解释时优先对应到推荐表格中的 `recommendation_id`，说明候选条件、rationale、证据引用和决策路径。

## 常见问题处理

- `LLM agent unavailable for stagnation_diagnosis`：运行中的进程或容器没有可用 `OPENAI_API_KEY`，或 agent 构建失败。用 `--env-file .env` 重启 Docker；不使用 LLM 时改用 `controller_mode=bo_only`。
- 兼容 OpenAI 的中转接口失败：检查 `OPENAI_BASE_URL` 是否需要 `/v1` 后缀，并确认 `configs/agent_bo.yaml` 中模型名可被该网关识别。
- Windows 安装 `matter-golem` 失败：使用 Docker/Linux；本机 Windows 可跳过该依赖。
- 本机提示 `ModuleNotFoundError: olympus`：设置 `PYTHONPATH` 包含 `third_party/atlas/src` 和 `third_party/olympus/src`，或直接使用 Docker。
- 改依赖或工作流后运行 smoke test：
  - 本机：设置 `PYTHONPATH` 后运行 `python -m pytest tests/test_lab_mode.py -q`。
  - Docker：`docker run --rm trace-collab-lab python -m pytest tests/test_lab_mode.py -q`。
