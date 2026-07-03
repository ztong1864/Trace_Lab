# TRACE Lab API 接口文档

**版本**: 0.1  
**基础 URL**: `http://127.0.0.1:8788`  
**协议**: HTTP/1.1  
**Content-Type**: `application/json`

---

## 接口总览

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/health` | 健康检查 |
| GET | `/api/projects` | 列出所有项目 |
| POST | `/api/projects` | 创建新项目 |
| GET | `/api/projects/{project_id}` | 获取项目摘要 |
| POST | `/api/projects/{project_id}/ask` | 生成实验批次推荐 |
| POST | `/api/projects/{project_id}/tell` | 提交实验结果 |
| POST | `/api/projects/{project_id}/reset` | 重置项目运行状态 |
| GET | `/api/projects/{project_id}/recommendations` | 获取所有推荐批次 |
| GET | `/api/projects/{project_id}/evidence` | 获取证据卡 |
| GET | `/api/projects/{project_id}/observations` | 获取所有观测记录 |
| POST | `/api/projects/{project_id}/observations/import` | 导入历史观测数据 |
| GET | `/api/projects/{project_id}/trace/{round_id}` | 获取某轮决策追踪 |

---

## 1. 健康检查

### `GET /api/health`

检查服务是否正常运行。

**请求参数**: 无

**响应示例**:
```json
{
  "ok": true,
  "projects_root": "runs/lab_projects"
}
```

**响应字段说明**:

| 字段 | 类型 | 说明 |
|------|------|------|
| `ok` | `boolean` | 服务是否正常 |
| `projects_root` | `string` | 项目存储根目录路径 |

---

## 2. 列出所有项目

### `GET /api/projects`

列出当前所有已创建的实验项目。

**请求参数**: 无

**响应示例**:
```json
{
  "projects": [
    {
      "project_id": "oxidative_esterification_demo",
      "project_dir": "runs/lab_projects/oxidative_esterification_demo",
      "reaction_name": "oxidative_esterification",
      "objective_name": "yield",
      "goal": "maximize",
      "batch_size": 6,
      "planner_name": "atlas",
      "controller_mode": "agentic",
      "observation_count": 15,
      "completed_observation_count": 12,
      "batch_count": 3,
      "best_so_far": 87.5,
      "created_at": "2026-06-01T10:00:00Z",
      "updated_at": "2026-06-10T14:30:00Z"
    }
  ]
}
```

**响应字段说明**:

| 字段 | 类型 | 说明 |
|------|------|------|
| `projects` | `array` | 项目摘要列表 |
| `projects[].project_id` | `string` | 项目唯一标识 |
| `projects[].project_dir` | `string` | 项目目录完整路径 |
| `projects[].reaction_name` | `string` | 反应名称 |
| `projects[].objective_name` | `string` | 优化目标名称（如 yield） |
| `projects[].goal` | `string` | 优化方向：maximize / minimize |
| `projects[].batch_size` | `integer` | 每批推荐实验数 |
| `projects[].planner_name` | `string` | 优化器名称（atlas / random） |
| `projects[].controller_mode` | `string` | 控制模式：agentic / bo_only |
| `projects[].observation_count` | `integer` | 总观测数 |
| `projects[].completed_observation_count` | `integer` | 已完成观测数 |
| `projects[].batch_count` | `integer` | 已生成的推荐批次数 |
| `projects[].best_so_far` | `float` | 当前最优目标值 |
| `projects[].created_at` | `string` | 项目创建时间 (ISO 8601) |
| `projects[].updated_at` | `string` | 项目最后更新时间 (ISO 8601) |

---

## 3. 创建新项目

### `POST /api/projects`

创建一个新的实验优化项目。

**请求体**:
```json
{
  "project_id": "my_reaction_01",
  "overwrite": false,
  "config": {
    "reaction_name": "oxidative_esterification",
    "objective_name": "yield",
    "goal": "maximize",
    "batch_size": 6,
    "planner_name": "atlas",
    "seed": 7,
    "reaction_scope": "Homogeneous metal-catalyzed oxidative esterification of aldehydes with alcohols",
    "controller_mode": "agentic",
    "agent_config_path": "configs/agent_bo.yaml",
    "planner_use_descriptors": false
  },
  "design_records": [
    {
      "variable": "catalyst",
      "type": "categorical",
      "value": "Fe(NO3)3·9H2O",
      "role": "variable",
      "active": true
    },
    {
      "variable": "catalyst",
      "type": "categorical",
      "value": "CuCl",
      "role": "variable",
      "active": true
    },
    {
      "variable": "temperature",
      "type": "continuous",
      "low": 25.0,
      "high": 120.0,
      "unit": "°C",
      "role": "variable",
      "active": true
    }
  ]
}
```

**请求字段说明**:

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `project_id` | `string` | 是 | 项目唯一标识，将作为目录名 |
| `overwrite` | `boolean` | 否 | 是否覆盖已有项目（默认 false） |
| `config` | `object` | 是 | 项目配置 |
| `config.reaction_name` | `string` | 否 | 反应名称（默认 "untitled_reaction"） |
| `config.objective_name` | `string` | 否 | 优化目标名称（默认 "yield"） |
| `config.goal` | `string` | 否 | "maximize" 或 "minimize"（默认 "maximize"） |
| `config.batch_size` | `integer` | 否 | 每批推荐数量（默认 6） |
| `config.planner_name` | `string` | 否 | 优化器：atlas / random（默认 "atlas"） |
| `config.seed` | `integer` | 否 | 随机种子（默认 7） |
| `config.reaction_scope` | `string` | 否 | 反应范围描述，用于证据匹配 |
| `config.controller_mode` | `string` | 否 | "agentic" / "bo_only"（默认 "agentic"） |
| `config.agent_config_path` | `string` | 否 | Agent 配置文件路径（默认 "configs/agent_bo.yaml"） |
| `config.planner_use_descriptors` | `boolean` | 否 | 是否使用描述符（默认 false） |
| `design_records` | `array` | 是 | 设计空间定义列表 |
| `design_records[].variable` | `string` | 是 | 变量/条件名称 |
| `design_records[].type` | `string` | 是 | categorical / continuous / discrete / fixed |
| `design_records[].value` | `string` | 见说明 | categorical 类型的选项值 |
| `design_records[].low` | `float` | 见说明 | continuous 类型的下界 |
| `design_records[].high` | `float` | 见说明 | continuous 类型的上界 |
| `design_records[].unit` | `string` | 否 | 单位 |
| `design_records[].role` | `string` | 否 | variable / controlled / fixed |
| `design_records[].active` | `boolean` | 否 | 是否为活跃优化变量 |
| `design_records[].stage` | `string` | 否 | 阶段标识（如 A、B） |

**响应**: 同项目摘要格式，参见 [获取项目摘要](#4-获取项目摘要)。

---

## 4. 获取项目摘要

### `GET /api/projects/{project_id}`

获取指定项目的完整状态摘要。

**路径参数**:

| 参数 | 类型 | 说明 |
|------|------|------|
| `project_id` | `string` | 项目标识 |

**响应示例**:
```json
{
  "project_id": "oxidative_esterification_demo",
  "project_dir": "runs/lab_projects/oxidative_esterification_demo",
  "reaction_name": "oxidative_esterification",
  "objective_name": "yield",
  "goal": "maximize",
  "batch_size": 6,
  "planner_name": "atlas",
  "controller_mode": "agentic",
  "observation_count": 15,
  "completed_observation_count": 12,
  "batch_count": 3,
  "best_so_far": 87.5,
  "pending_recommendation_count": 6,
  "variable_count": 5,
  "active_variables": ["catalyst", "solvent", "additive", "temperature", "time"],
  "created_at": "2026-06-01T10:00:00Z",
  "updated_at": "2026-06-10T14:30:00Z"
}
```

---

## 5. 生成实验批次推荐 (Ask)

### `POST /api/projects/{project_id}/ask`

核心接口：请求优化器为下一轮实验生成一个批次推荐。在 agentic 模式下会经过 LLM 决策层的审查和组合。

**路径参数**:

| 参数 | 类型 | 说明 |
|------|------|------|
| `project_id` | `string` | 项目标识 |

**请求体**:
```json
{
  "batch_size": 6,
  "planner_name": "atlas",
  "controller_mode": "agentic",
  "agent_config_path": "configs/agent_bo.yaml",
  "planner_use_descriptors": false
}
```

**请求字段说明**:

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `batch_size` | `integer` | 否 | 本批推荐数量（不填则使用项目默认值） |
| `planner_name` | `string` | 否 | atlas / random（不填则使用项目默认值） |
| `controller_mode` | `string` | 否 | agentic / bo_only（不填则使用项目默认值） |
| `agent_config_path` | `string` | 否 | agent 配置文件路径 |
| `planner_use_descriptors` | `boolean` | 否 | 是否启用描述符增强 |

**两种控制模式说明**:
- **agentic**: BO 优化器生成候选池后，经 LLM 智能体决策层审查、组合，生成带策略说明的批次推荐。
- **bo_only**: 纯贝叶斯优化器直接输出推荐，无 LLM 介入。

**响应示例**:
```json
{
  "project": { "...项目配置..." },
  "round_id": "round_003",
  "controller_mode": "agentic",
  "planner_use_descriptors": false,
  "recommendations": [
    {
      "recommendation_id": "rec_001",
      "round_id": "round_003",
      "role": "exploit",
      "rationale": "在当前最优区域附近进行高置信度开发",
      "candidate": {
        "catalyst": "Fe(NO3)3·9H2O",
        "solvent": "MeCN",
        "additive": "TEMPO",
        "temperature": 50.0,
        "time": 12.0
      },
      "predicted_value": 85.3,
      "uncertainty": 4.2,
      "status": "pending",
      "created_at": "2026-06-10T15:00:00Z"
    }
  ],
  "trace_path": "runs/lab_projects/.../trace_round_003.jsonl",
  "recommendations_json_path": "runs/lab_projects/.../recommendations_round_003.json",
  "recommendations_csv_path": "runs/lab_projects/.../recommendations_round_003.csv",
  "oracle_evaluation_disabled": true
}
```

**响应字段说明**:

| 字段 | 类型 | 说明 |
|------|------|------|
| `project` | `object` | 项目配置快照 |
| `round_id` | `string` | 本轮标识（如 round_003） |
| `controller_mode` | `string` | 实际使用的控制模式 |
| `planner_use_descriptors` | `boolean` | 是否启用了描述符 |
| `recommendations` | `array` | 推荐实验列表 |
| `recommendations[].recommendation_id` | `string` | 推荐唯一标识 |
| `recommendations[].round_id` | `string` | 所属轮次标识 |
| `recommendations[].role` | `string` | 策略角色：exploit / explore / diversity / scaffold |
| `recommendations[].rationale` | `string` | 推荐理由说明（agentic 模式下由 LLM 生成） |
| `recommendations[].candidate` | `object` | 实验条件键值对 |
| `recommendations[].predicted_value` | `float` | 预测的目标值 |
| `recommendations[].uncertainty` | `float` | 预测的不确定性 |
| `recommendations[].status` | `string` | pending / completed / failed / skipped |
| `recommendations[].created_at` | `string` | 推荐生成时间 |
| `trace_path` | `string` | 决策追踪文件路径 (.jsonl) |
| `recommendations_json_path` | `string` | 推荐 JSON 文件路径 |
| `recommendations_csv_path` | `string` | 推荐 CSV 文件路径 |
| `oracle_evaluation_disabled` | `boolean` | oracle 评估已禁用（生产环境固定为 true） |

---

## 6. 提交实验结果 (Tell)

### `POST /api/projects/{project_id}/tell`

将真实实验结果提交回系统。提交后优化器会更新模型，agentic 模式下会自动触发反思分析。

**路径参数**:

| 参数 | 类型 | 说明 |
|------|------|------|
| `project_id` | `string` | 项目标识 |

**请求体**:
```json
{
  "defer_reflection": false,
  "results": [
    {
      "recommendation_id": "rec_001",
      "yield": 82.4,
      "status": "completed",
      "failure_reason": "",
      "notes": "反应正常，后处理无明显损失",
      "observed_at": "2026-06-10T16:30:00Z",
      "stage": "lab_recommendation",
      "chemist_override": ""
    },
    {
      "recommendation_id": "rec_002",
      "yield": 0,
      "status": "failed",
      "failure_reason": "催化剂失活，无产物生成",
      "notes": "催化剂存放时间过长",
      "observed_at": "2026-06-10T16:35:00Z"
    }
  ]
}
```

**请求字段说明**:

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `defer_reflection` | `boolean` | 否 | 是否延迟反思（默认 false，同步完成） |
| `results` | `array` | 是 | 实验结果列表 |
| `results[].recommendation_id` | `string` | 是 | 对应的推荐标识 |
| `results[].{objective_name}` | `float` | 条件 | 目标值（如 yield），completed 时必填 |
| `results[].status` | `string` | 是 | completed / failed / skipped |
| `results[].failure_reason` | `string` | 否 | 失败原因（failed 时建议填写） |
| `results[].notes` | `string` | 否 | 实验备注 |
| `results[].observed_at` | `string` | 否 | 实验观测时间（ISO 8601） |
| `results[].stage` | `string` | 否 | 实验阶段标识 |
| `results[].chemist_override` | `string` | 否 | 化学家人工调整记录 |

**响应示例**:
```json
{
  "project_id": "oxidative_esterification_demo",
  "appended": [
    {
      "observation_id": "obs_00016",
      "round_id": "round_003",
      "recommendation_id": "rec_001",
      "source": "recommendation",
      "catalyst": "Fe(NO3)3·9H2O",
      "solvent": "MeCN",
      "yield": "82.4",
      "status": "completed",
      "failure_reason": "",
      "notes": "反应正常",
      "observed_at": "2026-06-10T16:30:00Z"
    }
  ],
  "observation_count": 16,
  "completed_observation_count": 13,
  "best_so_far": 87.5,
  "reflections": [
    {
      "recommendation_id": "rec_001",
      "observed_value": 82.4,
      "predicted_value": 85.3,
      "delta": -2.9,
      "assessment": "within_range",
      "insight": "预测准确，模型保持稳定"
    }
  ],
  "reflection_status": "completed",
  "reflection_recommendation_ids": []
}
```

**响应字段说明**:

| 字段 | 类型 | 说明 |
|------|------|------|
| `project_id` | `string` | 项目标识 |
| `appended` | `array` | 已追加的观测记录列表 |
| `observation_count` | `integer` | 更新后总观测数 |
| `completed_observation_count` | `integer` | 更新后已完成观测数 |
| `best_so_far` | `float` | 更新后当前最优值 |
| `reflections` | `array` | 反思分析结果（仅 agentic 模式 + completed 结果） |
| `reflections[].recommendation_id` | `string` | 对应的推荐标识 |
| `reflections[].observed_value` | `float` | 实际观测值 |
| `reflections[].predicted_value` | `float` | 原始预测值 |
| `reflections[].delta` | `float` | 预测偏差 |
| `reflections[].assessment` | `string` | 评估结论 |
| `reflections[].insight` | `string` | 反思洞察 |
| `reflection_status` | `string` | completed / deferred / none |
| `reflection_recommendation_ids` | `array` | 待后台完成的反思推荐 ID 列表 |

---

## 7. 重置项目运行状态

### `POST /api/projects/{project_id}/reset`

清除项目的运行数据（推荐、追踪、日志），保留项目配置和证据卡。用于调试或重新开始优化。

**路径参数**:

| 参数 | 类型 | 说明 |
|------|------|------|
| `project_id` | `string` | 项目标识 |

**请求体**:
```json
{
  "backup": true,
  "keep_historical": false
}
```

**请求字段说明**:

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `backup` | `boolean` | 否 | 是否备份当前状态（默认 true） |
| `keep_historical` | `boolean` | 否 | 是否保留 source=historical 的观测记录（默认 false） |

**响应示例**:
```json
{
  "project_id": "oxidative_esterification_demo",
  "backup_dir": "runs/lab_projects/_backups/oxidative_esterification_demo_reset_20260610_150000",
  "removed_files": ["recommendations_round_001.csv", "trace_round_001.jsonl"],
  "cleared_observations": true,
  "keep_historical": false,
  "preserved_historical_observation_count": 0,
  "removed_observation_count": 15,
  "summary": { "...项目摘要..." },
  "preserved_files": ["project.yaml", "design_space.csv", "evidence_cards.jsonl"]
}
```

---

## 8. 获取所有推荐批次

### `GET /api/projects/{project_id}/recommendations`

获取该项目所有历史推荐批次及其详情。

**路径参数**:

| 参数 | 类型 | 说明 |
|------|------|------|
| `project_id` | `string` | 项目标识 |

**响应示例**:
```json
{
  "batches": [
    {
      "round_id": "round_001",
      "created_at": "2026-06-01T10:05:00Z",
      "recommendations": [
        {
          "recommendation_id": "rec_001",
          "role": "explore",
          "rationale": "初始探索阶段，覆盖设计空间多样性",
          "candidate": { "catalyst": "Fe(NO3)3·9H2O", "...": "..." },
          "predicted_value": 72.1,
          "uncertainty": 12.5,
          "status": "completed",
          "result": "68.3",
          "observed_at": "2026-06-01T14:00:00Z"
        }
      ],
      "trace_records": [ "...决策追踪记录..." ]
    }
  ]
}
```

---

## 9. 获取证据卡

### `GET /api/projects/{project_id}/evidence`

获取项目中经过审核、用于辅助决策的文献证据卡。仅返回 UI 可见范围和非封禁状态的卡。

**路径参数**:

| 参数 | 类型 | 说明 |
|------|------|------|
| `project_id` | `string` | 项目标识 |

**响应示例**:
```json
{
  "evidence_path": "runs/lab_projects/.../evidence_cards.jsonl",
  "count": 5,
  "cards": [
    {
      "card_id": "evid_001",
      "source": "Fe(NO3)3+TEMPO+KCl-rt.pdf",
      "summary": "Fe(NO3)3/TEMPO/KCl 体系在室温下可实现醛的氧化酯化，收率 85-92%",
      "reaction_scope": "Homogeneous metal-catalyzed oxidative esterification",
      "variable_scope": ["catalyst", "additive"],
      "target_nodes": ["stagnation_diagnosis", "hypothesis_action"],
      "mapping_status": "direct",
      "confidence": "high",
      "allowed_use": "advisory",
      "source_type": "literature",
      "doi": "10.xxxx/xxxx",
      "supporting_excerpt": "Under optimized conditions, the reaction afforded...",
      "transferability_note": "底物范围相似，预计可迁移",
      "leakage_risk": "clean_literature_prior",
      "notes": ""
    }
  ]
}
```

**证据卡字段说明**:

| 字段 | 类型 | 说明 |
|------|------|------|
| `card_id` | `string` | 证据卡唯一标识 |
| `source` | `string` | 来源名称（文件名或引用） |
| `summary` | `string` | 证据摘要 |
| `reaction_scope` | `string` | 反应范围，用于匹配当前项目 |
| `variable_scope` | `array` | 涉及的设计变量 |
| `target_nodes` | `array` | 适用的决策节点 |
| `mapping_status` | `string` | 映射状态：direct / same_start_end / same_reaction_family / variable_level / background |
| `confidence` | `string` | 置信度：high / medium / low |
| `allowed_use` | `string` | 允许的使用方式：advisory / decision_active |
| `source_type` | `string` | 来源类型：literature / expert / curated |
| `doi` | `string` | 数字对象标识符 |
| `supporting_excerpt` | `string` | 支撑性原文摘录 |
| `transferability_note` | `string` | 可迁移性说明 |
| `leakage_risk` | `string` | 数据泄露风险级别 |
| `notes` | `string` | 额外备注 |

---

## 10. 获取所有观测记录

### `GET /api/projects/{project_id}/observations`

获取项目中的所有实验观测记录（含已完成、失败、跳过的实验）。

**路径参数**:

| 参数 | 类型 | 说明 |
|------|------|------|
| `project_id` | `string` | 项目标识 |

**响应示例**:
```json
{
  "observations": [
    {
      "observation_id": "obs_00001",
      "round_id": "round_001",
      "recommendation_id": "rec_001",
      "source": "recommendation",
      "stage": "lab_recommendation",
      "catalyst": "Fe(NO3)3·9H2O",
      "solvent": "MeCN",
      "additive": "TEMPO",
      "temperature": "50.0",
      "time": "12.0",
      "yield": "68.3",
      "status": "completed",
      "failure_reason": "",
      "notes": "",
      "observed_at": "2026-06-01T14:00:00Z",
      "chemist_override": ""
    }
  ]
}
```

---

## 11. 导入历史观测数据

### `POST /api/projects/{project_id}/observations/import`

将非 TRACE 推荐产生的历史实验数据导入系统，作为模型的先验知识。

**路径参数**:

| 参数 | 类型 | 说明 |
|------|------|------|
| `project_id` | `string` | 项目标识 |

**请求体**:
```json
{
  "source": "historical",
  "allow_duplicates": false,
  "rows": [
    {
      "catalyst": "CuCl",
      "solvent": "EtOH",
      "additive": "NHPI",
      "temperature": 80.0,
      "time": 24.0,
      "yield": 55.2,
      "status": "completed",
      "notes": "文献条件复现",
      "observed_at": "2026-01-15T09:00:00Z"
    }
  ]
}
```

**请求字段说明**:

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `source` | `string` | 否 | 数据来源标识（默认 "historical"） |
| `allow_duplicates` | `boolean` | 否 | 是否允许重复条件导入（默认 false） |
| `rows` | `array` | 是 | 历史数据行列表 |
| `rows[].{variable}` | `string` | 是 | 各设计变量/条件的值 |
| `rows[].{objective}` | `float` | 条件 | completed 状态时目标值 |
| `rows[].status` | `string` | 否 | completed / failed / skipped（默认 completed） |
| `rows[].observed_at` | `string` | 否 | 观测时间 |
| `rows[].notes` | `string` | 否 | 备注 |

**响应示例**:
```json
{
  "project_id": "oxidative_esterification_demo",
  "imported": [ "...导入的记录..." ],
  "skipped": [],
  "imported_count": 3,
  "skipped_count": 0,
  "source": "historical",
  "observation_count": 18,
  "previous_observation_count": 15,
  "completed_observation_count": 15,
  "best_so_far": 87.5,
  "summary": { "...项目摘要..." }
}
```

---

## 12. 获取决策追踪

### `GET /api/projects/{project_id}/trace/{round_id}`

获取指定轮次的完整决策追踪记录，用于审查智能体在每个决策节点的推理过程。

**路径参数**:

| 参数 | 类型 | 说明 |
|------|------|------|
| `project_id` | `string` | 项目标识 |
| `round_id` | `string` | 轮次标识（如 round_003） |

**响应示例**:
```json
{
  "round_id": "round_003",
  "trace": [
    {
      "node": "stagnation_diagnosis",
      "timestamp": "2026-06-10T15:00:00Z",
      "input": { "...节点输入..." },
      "output": {
        "is_stagnating": false,
        "diagnosis": "优化仍在进展中，上轮 best_so_far 提升 3.2%",
        "recommended_action": "continue_exploit"
      }
    },
    {
      "node": "hypothesis_action",
      "timestamp": "2026-06-10T15:00:01Z",
      "input": { "...节点输入..." },
      "output": {
        "hypothesis": "铁盐催化剂在中等温度下活性最佳，应在 40-60°C 范围精细搜索",
        "confidence": "medium",
        "evidence_support": ["evid_001"]
      }
    }
  ]
}
```

**决策追踪结构说明**: 追踪记录为 `.jsonl` 格式的按行 JSON 数组，每行为一个决策节点记录。节点类型包括：
- `design_init_experiments` - 实验初始化设计
- `stagnation_diagnosis` - 停滞诊断
- `hypothesis_action` - 假设生成
- `semantic_assessment` - 语义评估
- `verification_pass` - 验证环节
- `reflection_action` - 反思分析
- `lab_batch_composition` - 实验室批次组装

---

## 典型工作流

