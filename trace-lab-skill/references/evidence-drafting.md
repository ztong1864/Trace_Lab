# 从论文起草证据卡

证据卡（evidence card）是给 TRACE 控制器看的、带出处的文献线索。起草由 agent 完成，检查由 TRACE 的代码完成，是否采用由化学研究者决定。**agent 只负责“逐字引用 + 如实概括”，不负责判断这篇论文和项目是否“同一个反应”。**

## 流程

```text
evidence-prepare   PDF -> 分页文本、packets/、context.json          （TRACE）
起草               读 packets/ 和 context.json，写 drafts/*.jsonl      （agent，本指南）
evidence-verify    逐条核对引文、数字、变量名                        （TRACE）
evidence-sheet     生成 review_sheet.csv 给研究者                    （TRACE）
研究者审阅         在表里填 accept / reject，可改状态和措辞
evidence-accept    只导入 accept 的行（严格校验，旧文件自动备份）    （TRACE）
evidence-preview   看控制器实际会看到哪几张卡                        （TRACE）
```

所有中间文件在 `<project>/evidence_work/`。只有 `evidence-accept` 会改项目自己的 `evidence_cards.jsonl`。

## 起草前先读

1. `evidence_work/context.json`：项目的反应范围（`reaction_scope`）、目标、每个设计变量的**准确名称**和选项、已有证据卡。`variable_scope` 只能用这里的变量名（区分大小写，注意 `·` 等符号）。
2. `evidence_work/packets/<source_id>_pXX-YY.txt`：每页有两段文字：
   - `(reading order)`：正文，段落是连贯的，**引正文从这里引**。
   - `(table layout ...)`：只在有表格的页出现，行列对齐，**引表格行从这里引**。阅读顺序里的表格会被拆成一格一行，不要用来判断哪个数字属于哪一行。
3. 如果 `evidence-prepare` 报告某篇 `needs_ocr`，它是扫描件，没有文字，不能起草；告诉用户先做 OCR。

## 每个发现写成一行 JSON

放进 `evidence_work/drafts/<任意名>.jsonl`，一行一个发现：

```json
{"source_id": "fe_no3_3_tempo_kcl_rt_8c3580f9",
 "finding_id": "solvent_screen",
 "page": 2,
 "locator": "Table 1, entries 1-7",
 "quote": "1 10 10 DCE 59 [...] 3 10 10 CH3CN 65 [...] 7 10 10 Toluene 87 (81b)",
 "summary": "Solvent screen for Fe(NO3)3·9H2O/TEMPO/KCl (10 mol% each) aerobic oxidation of butane-1,4-diol (1 mmol) with an O2 balloon at 25 °C. NMR yields: DCE 59%, CH3CN 65%, toluene 87% (81% isolated).",
 "variable_scope": ["Solvent"],
 "proposed_mapping_status": "same_reaction_family",
 "confidence": 0.95,
 "transferability_note": "Simple diol at 25 °C with KCl; this project runs substrate 1b at 60 °C with a varied additive, so use as a starting anchor for solvent choice only.",
 "target_nodes": ["design_init_experiments", "hypothesis_action"]}
```

| 字段 | 要求 |
| --- | --- |
| `source_id` | 用 packet 文件头里的 `SOURCE` 值 |
| `finding_id` | 同一篇论文内唯一，如 `solvent_screen`；卡片编号 = `<source_id>_<finding_id>` |
| `page` | 引文所在页（PDF 页码）；写错会被自动改正并提示 |
| `quote` | **逐字**摘自论文，不超过 900 字符；空白、连字符、上下标差异不影响匹配。要跳过中间内容用 `[...]`，每一段都必须在同一页 |
| `summary` | 做了什么、测到什么：底物、催化体系、溶剂、温度、产率，按论文原样写。至少 40 字符 |
| `variable_scope` | 这条发现涉及的设计变量，名称必须与 `context.json` 一致 |
| `proposed_mapping_status` | 只能是 `same_reaction_family`、`variable_level`、`background`、`out_of_scope` |
| `confidence` | 0–1 的数，或 `high` / `medium` / `low` |
| `transferability_note` | **必填**。这篇论文的底物、规模、温度、条件与本项目有何不同，为什么不能直接照搬 |
| `target_nodes` | 可省略；默认 `design_init_experiments`、`hypothesis_action` |
| `reaction_scope` | 可省略。论文自己研究的反应，如 `iron/nitroxyl aerobic oxidative lactonization of 1,4-diols`（不要照抄项目的 `reaction_scope`）。研究者若把状态升为 `direct` / `same_start_end`，这个字段必须能和项目的 `reaction_scope` 互相包含，否则卡片永远不会被展示，`evidence-preview` 会指出 |
| `source` | 可省略。完整引用（作者、期刊、年份、DOI）；省略时用文件名和 DOI |

## 规则

1. **一条发现一张卡。** 溶剂筛选、对照实验、催化剂用量、放大实验分开写，每条 3–8 个数字为宜。
2. **summary 里的每个百分数必须出现在同一条的 quote 里。** 这是最常见的被拒原因：例如 summary 写了“对照 80%（对比标准条件 87%）”，但 quote 里没有 87，就会被拒。要么把含 87 的那行用 `[...]` 加进 quote，要么不写 87。其他数量（mol%、°C、h、mmol）要求出现在同一页，否则给警告。
3. **不推断。** 不写论文没说的结论（“最佳溶剂”“普遍适用”），除非引文里原话就是这样。
4. **说清差异。** `transferability_note` 用 `context.json` 里的固定条件（温度、底物量、时间）对比论文条件。
5. **不要自己声明“直接匹配”。** `direct`、`same_start_end` 需要化学研究者判断论文底物是否就是本项目底物，起草时写了也会被降为 `same_reaction_family` 并提示。
6. **变量不在优化范围内（`optimized: false`）的发现用 `background`。** 否则控制器永远看不到这张卡（会给警告 `not_retrievable`）。
7. **和项目无关的论文，不写卡片**，在回复里说明“这篇不相关，原因是……”。不要为了凑数强写。
8. **不要把论文里的结果表逐行抄成查表。** 一条引文里同时给出本项目多个选项的产率时会出现警告 `lookup_like`：可以保留，但 summary 应说明趋势，而不是让控制器按表挑候选。
9. **不重复。** 项目里已有卡引用同一篇论文时会提示 `paper_already_cited`，只保留确实补充了新信息的发现。
10. **表格的数字对不上行时，不要按位置推断。** 有的 PDF 表格在文字层里错位（例如某一列的数值整体排在表格底部，或夹在相邻行之间）。代码只能证明数字在引文里，不能证明它属于哪一行；按位置猜出来的“某某产率”一旦猜错，检查也发现不了。这种表格改用正文里明写的句子起草（例如“MeCN gave the best result”），或在 summary 里注明“具体数值见原文 Table 1，此处未引用”，并让研究者对着 PDF 原图核对。

## 写完之后：卡片会不会真被展示？

控制器每次 `ask` 只取得分最高的前几张卡（`knowledge_top_k`，默认 5），整次 ask 的所有节点共用这一组。得分大致是 `状态分 + 10×置信度 + 4×涉及的设计变量数 + 3×命中的节点数`，所以**只涉及 1–2 个变量的精确卡片，很难排进前 5**，项目里已有涉及 6–7 个变量的宽泛卡片时尤其如此。导入后请运行 `evidence-preview` 看新卡排第几；这是检索排序的局限，不要为了排名靠前把 `variable_scope` 写宽——那会让卡片说得比论文更多。

## 检查结果怎么读

运行 `evidence-verify` 后：

| 状态 | 含义 |
| --- | --- |
| `OK` | 引文和数字都核对过，可以进入审阅表（可能带警告） |
| `REJECTED` | 有错误，不会进入审阅表；按提示改 drafts 再运行一次 |
| `DUPLICATE` | 项目里已有同一 card_id 或同一段引文 |

常见错误：`quote_not_found`（引文与论文不符或改过数字）、`number_not_in_quote`、`unknown_variable`（提示最接近的变量名）、`no_transferability_note`、`summary_too_short`。

**检查通过不等于内容正确。** 代码能证明“这句话在论文里、这些数字在引文里”，不能证明“这个产率属于 summary 说的那个条件”。这一步由化学研究者对着引文来判断，所以审阅表把引文和 summary 并排放。
