# TRACE Lab Data Format

Each lab project is a folder under `runs/lab_projects`.

```text
project.yaml
design_space.csv
observations.csv
evidence_cards.jsonl
recommendations_round_*.json
recommendations_round_*.csv
trace_round_*.jsonl
```

## `project.yaml`

Defines project metadata, objective, planner, controller mode, batch size, and
the evidence file.

## `design_space.csv`

Normalized design-space table.  Important columns include:

```text
variable,type,value,low,high,step,unit,stage,active,role,fixed_value
```

Supported variable types:

- `categorical`
- `discrete_numeric`
- `continuous`
- `fixed`

Descriptor columns can be provided as `descriptor__<name>`.

## `observations.csv`

Measured results.  Recommendation-derived observations are written by `tell`.
Historical rows can be imported before recommendation generation.

## Recommendations and trace files

`recommendations_round_*.csv/json` contain pending or completed
recommendations.  `trace_round_*.jsonl` stores controller decisions, evidence
access, action/contract fields, and outcome reflections.

