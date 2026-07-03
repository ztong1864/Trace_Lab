const $ = (selector) => document.querySelector(selector);

let currentProject = null;
let latestBatch = null;
let evidenceById = {};
let uiState = {
  busy: false,
  operation: "",
};

const actionButtons = [
  "#loadProject",
  "#refreshProjects",
  "#askButton",
  "#reloadRecommendations",
  "#reloadObservations",
  "#exportRecommendations",
  "#submitResults",
  "#createProject",
  "#importHistorical",
  "#resetProject",
];

function setStatus(message, isError = false) {
  const node = $("#statusText");
  node.textContent = message;
  node.classList.toggle("error", isError);
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.detail || `Request failed: ${response.status}`);
  }
  return data;
}

function projectId() {
  return $("#projectId").value.trim();
}

function hasLoadedProject() {
  return Boolean(currentProject?.project?.project_id);
}

function pendingRecommendations() {
  return (latestBatch?.recommendations || []).filter(
    (item) => !isTerminalStatus(item.status)
  );
}

function hasPendingRecommendations() {
  return pendingRecommendations().length > 0;
}

function inputForRecommendation(recommendationId, kind) {
  const selectorByKind = {
    result: ".result-input",
    note: ".notes-input",
    status: ".status-select",
    failure: ".failure-input",
  };
  const keyByKind = {
    result: "rec",
    note: "note",
    status: "statusRec",
    failure: "failure",
  };
  const selector = selectorByKind[kind] || ".result-input";
  const key = keyByKind[kind] || "rec";
  return Array.from(document.querySelectorAll(selector)).find(
    (node) => node.dataset[key] === recommendationId
  );
}

function collectResultRows() {
  if (!latestBatch?.recommendations?.length) return [];
  const objective = currentProject?.project?.objective_name || "yield";
  const rows = [];
  for (const item of latestBatch.recommendations) {
    if (isTerminalStatus(item.status)) continue;
    const resultInput = inputForRecommendation(item.recommendation_id, "result");
    const noteInput = inputForRecommendation(item.recommendation_id, "note");
    const statusInput = inputForRecommendation(item.recommendation_id, "status");
    const failureInput = inputForRecommendation(item.recommendation_id, "failure");
    const value = resultInput?.value?.trim() || "";
    const notes = noteInput?.value?.trim() || "";
    const failureReason = failureInput?.value?.trim() || "";
    let status = statusInput?.value || "pending";
    if (status === "pending" && value) status = "completed";
    if (status === "completed" && !value) continue;
    if (status === "pending") continue;
    rows.push({
      recommendation_id: item.recommendation_id,
      [objective]: value,
      status,
      failure_reason: failureReason,
      notes,
    });
  }
  return rows;
}

function hasSubmittableResults() {
  return collectResultRows().length > 0;
}

function setDisabled(selector, disabled, title = "") {
  const node = $(selector);
  if (!node) return;
  node.disabled = Boolean(disabled);
  node.title = title;
}

function updateControls() {
  const busy = uiState.busy;
  const id = projectId();
  const loaded = hasLoadedProject();
  const pending = hasPendingRecommendations();
  const canSubmit = loaded && hasSubmittableResults();
  const canExport = loaded && Boolean(latestBatch?.recommendations?.length);
  const canCreate = Boolean($("#newProjectId").value.trim() && $("#designCsv").files?.[0]);
  const canImportHistorical = loaded && Boolean($("#historicalCsv").files?.[0]);

  setDisabled("#loadProject", busy || !id);
  setDisabled("#refreshProjects", busy);
  setDisabled(
    "#askButton",
    busy || !loaded || pending,
    pending ? "Submit, fail, or skip the current pending recommendations before generating a new batch." : ""
  );
  setDisabled("#reloadRecommendations", busy || !loaded);
  setDisabled("#reloadObservations", busy || !loaded);
  setDisabled("#exportRecommendations", busy || !canExport);
  setDisabled("#submitResults", busy || !canSubmit);
  setDisabled("#createProject", busy || !canCreate);
  setDisabled("#importHistorical", busy || !canImportHistorical);
  setDisabled("#resetProject", busy || !id);

  for (const selector of actionButtons) {
    const node = $(selector);
    if (node) node.classList.toggle("is-busy", busy);
  }
  document.querySelectorAll(".result-input, .notes-input, .status-select, .failure-input").forEach((node) => {
    node.disabled = busy || node.dataset.terminal === "true";
  });
}

async function withUiLock(message, task) {
  if (uiState.busy) return null;
  uiState.busy = true;
  uiState.operation = message;
  setStatus(message);
  updateControls();
  try {
    return await task();
  } finally {
    uiState.busy = false;
    uiState.operation = "";
    updateControls();
  }
}

async function refreshProjectView() {
  const id = projectId();
  currentProject = await api(`/api/projects/${encodeURIComponent(id)}`);
  await loadEvidenceIndex();
  renderProject(currentProject);
  await Promise.all([loadRecommendations(), loadObservations()]);
}

async function loadEvidenceIndex() {
  const id = projectId();
  evidenceById = {};
  if (!id) return;
  try {
    const data = await api(`/api/projects/${encodeURIComponent(id)}/evidence`);
    evidenceById = Object.fromEntries(
      (data.cards || [])
        .filter((card) => card.card_id)
        .map((card) => [card.card_id, card])
    );
  } catch (_error) {
    evidenceById = {};
  }
}

async function loadProject() {
  const id = projectId();
  if (!id) return setStatus("Project ID is required.", true);
  return withUiLock("Loading project...", async () => {
    try {
      await refreshProjectView();
      setStatus("Project loaded.");
    } catch (error) {
      setStatus(error.message, true);
    }
  });
}

function renderProject(summary) {
  const project = summary.project || {};
  $("#projectTitle").textContent = `${project.project_id || "project"} · ${project.reaction_name || ""}`;
  $("#bestChip").textContent = `best: ${summary.best_so_far ?? "n/a"}`;
  renderProjectMeta(summary);
  renderDesignSpace(summary.design_space || {});
  updateControls();
}

function renderProjectMeta(summary) {
  const designSpace = summary.design_space || {};
  const project = summary.project || {};
  const active = (designSpace.active_variables || []).join(", ") || "none";
  const fixed = (designSpace.fixed_conditions || [])
    .map((item) => `${item.name}=${item.fixed_value || item.value_summary || ""}`)
    .join("; ") || "none";
  const stages = (designSpace.stages || []).join(", ") || "n/a";
  const historical = summary.historical_observation_count ?? 0;
  const completed = summary.completed_observation_count ?? 0;
  $("#projectMeta").textContent =
    `stage: ${stages} · objective: ${project.objective_name || "yield"} · ` +
    `active: ${active} · fixed: ${fixed} · completed: ${completed} (${historical} historical)`;
}

function renderDesignSpace(designSpace) {
  const variables = designSpace.variables || [];
  const active = (designSpace.active_variables || [])
    .map((name) => variables.find((item) => item.name === name))
    .filter(Boolean);
  const fixed = designSpace.fixed_conditions || [];
  const controlled = designSpace.controlled_conditions || [];
  const inactive = designSpace.inactive_variables || [];
  const activeCount = designSpace.active_variable_count ?? active.length;
  const fixedCount = designSpace.fixed_condition_count ?? fixed.length;
  const controlledCount = designSpace.controlled_condition_count ?? controlled.length;
  const metaBits = [`${activeCount} active`, `${fixedCount} fixed`];
  if (controlledCount) metaBits.push(`${controlledCount} controlled`);
  $("#spaceMeta").textContent = metaBits.join(" · ");
  const node = $("#designSpace");
  node.classList.remove("empty");
  const sections = [];
  sections.push(designSection("Active Variables", active, "No active optimization variables."));
  sections.push(designSection("Fixed Conditions", fixed, "No fixed conditions."));
  if (controlled.length) {
    sections.push(
      designSection(
        "Controlled Conditions",
        controlled,
        "No controlled conditions."
      )
    );
  }
  if (inactive.length) {
    sections.push(designSection("Inactive / Future Variables", inactive, "No inactive variables."));
  }
  node.innerHTML = sections.join("");
}

function designSection(title, rows, emptyText) {
  const body = rows?.length
    ? table(
        ["Variable", "Type", "Values / Range", "Unit", "Stage", "Role", "Descriptors"],
        rows.map((item) => [
          escapeHtml(item.name),
          escapeHtml(item.type),
          escapeHtml(designValueLabel(item)),
          escapeHtml(item.unit || ""),
          escapeHtml(item.stage || ""),
          escapeHtml(item.role || ""),
          descriptorCell(item),
        ]),
        true
      )
    : `<div class="empty compact">${escapeHtml(emptyText)}</div>`;
  return `<section class="subsection"><h4>${escapeHtml(title)}</h4>${body}</section>`;
}

function descriptorCell(item) {
  const count = item.descriptor_count || 0;
  const names = item.descriptor_names || [];
  if (!count) return "";
  if (!names.length) return escapeHtml(String(count));
  return `
    <details class="cell-details">
      <summary>${escapeHtml(String(count))} descriptors</summary>
      <div>${escapeHtml(names.join(", "))}${count > names.length ? " ..." : ""}</div>
    </details>
  `;
}

function designValueLabel(item) {
  if (item.is_fixed) return item.fixed_value || item.value_summary || "";
  if (item.type === "continuous") return item.value_summary || `${item.low ?? ""}-${item.high ?? ""}`;
  if (item.type === "discrete_numeric") {
    return item.value_summary || compactValues(item.values, item.values_truncated);
  }
  if (item.is_controlled) return item.value_summary || item.fixed_value || "";
  if (item.values?.length && (item.option_count || 0) <= 8) {
    return compactValues(item.values, item.values_truncated);
  }
  if (item.option_count) return `${item.option_count} options`;
  return item.value_summary || "";
}

function compactValues(values = [], truncated = false) {
  const visible = values.slice(0, 8).join(", ");
  if (!visible) return "";
  return `${visible}${truncated || values.length > 8 ? ", ..." : ""}`;
}

async function askBatch() {
  const id = projectId();
  if (!id) return setStatus("Project ID is required.", true);
  if (hasPendingRecommendations()) {
    return setStatus("Submit, fail, or skip the current pending recommendations before generating a new batch.", true);
  }
  return withUiLock("Generating recommendations...", async () => {
    try {
      const payload = {
        batch_size: Number.parseInt($("#batchSize").value || "6", 10),
        planner_name: $("#plannerName").value || null,
        controller_mode: $("#controllerMode").value || null,
      };
      const data = await api(`/api/projects/${encodeURIComponent(id)}/ask`, {
        method: "POST",
        body: JSON.stringify(payload),
      });
      latestBatch = { round_id: data.round_id, recommendations: data.recommendations };
      await loadEvidenceIndex();
      renderRecommendations(latestBatch);
      await refreshProjectView();
      setStatus(`Generated ${data.recommendations.length} recommendations.`);
    } catch (error) {
      setStatus(error.message, true);
    }
  });
}

async function loadRecommendations() {
  const id = projectId();
  if (!id) return;
  try {
    const data = await api(`/api/projects/${encodeURIComponent(id)}/recommendations`);
    const batches = data.batches || [];
    latestBatch = batches[batches.length - 1] || null;
    renderRecommendations(latestBatch);
    updateControls();
  } catch (error) {
    $("#recommendations").textContent = error.message;
    $("#recommendations").classList.add("empty");
    updateControls();
  }
}

async function reloadRecommendations() {
  if (!hasLoadedProject()) return setStatus("Load a project first.", true);
  return withUiLock("Reloading recommendations...", async () => {
    await loadRecommendations();
    setStatus("Recommendations reloaded.");
  });
}

function exportRecommendationsCsv() {
  if (!latestBatch?.recommendations?.length) return setStatus("No recommendations to export.", true);
  const candidateKeys = Array.from(
    new Set(
      latestBatch.recommendations.flatMap((item) => Object.keys(item.candidate || {}))
    )
  );
  const headers = [
    "round_id",
    "recommendation_id",
    "rank",
    "status",
    ...candidateKeys,
    "batch_role",
    "controller_action",
    "rationale",
    "evidence_refs",
  ];
  const rows = latestBatch.recommendations.map((item) => {
    const candidate = item.candidate || {};
    return headers.map((header) => {
      if (candidateKeys.includes(header)) return candidate[header] ?? "";
      if (header === "evidence_refs") return recommendationEvidenceRefs(item).join("; ");
      return item[header] ?? "";
    });
  });
  const csv = [headers, ...rows]
    .map((row) => row.map(csvCell).join(","))
    .join("\n");
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `${projectId() || "trace_lab"}_${latestBatch.round_id || "recommendations"}.csv`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
  setStatus("Recommendation CSV exported.");
}

function renderRecommendations(batch) {
  const node = $("#recommendations");
  if (!batch || !batch.recommendations?.length) {
    node.classList.add("empty");
    node.textContent = "No recommendation batch yet.";
    renderBatchSummary(null);
    updateControls();
    return;
  }
  node.classList.remove("empty");
  renderBatchSummary(batch);
  const rows = batch.recommendations.map((item) => {
    const candidate = item.candidate || {};
    return [
      escapeHtml(item.recommendation_id || ""),
      statusCellHtml(item),
      candidateHtml(candidate),
      escapeHtml(item.controller_action || ""),
      escapeHtml(item.batch_role || ""),
      descriptorHtml(item.descriptor_contrast_to_anchor || {}, item.descriptor_signal || {}),
      evidenceHtml(item),
      rationaleHtml(item.rationale || ""),
      traceLinkHtml(batch.round_id, item),
      resultInputHtml(item),
      failureInputHtml(item),
      notesInputHtml(item),
    ];
  });
  node.innerHTML = table(
    ["ID", "Status", "Candidate", "Action", "Role", "Descriptor", "Evidence", "Rationale", "Trace", "Result", "Failure", "Notes"],
    rows,
    true
  );
  updateControls();
}

function renderBatchSummary(batch) {
  const node = $("#batchSummary");
  if (!node) return;
  const first = batch?.recommendations?.[0] || {};
  const contract = first.batch_contract || {};
  const strategy = contract.batch_strategy || "";
  const rationale = contract.batch_rationale || "";
  const constraints = contract.global_constraints || [];
  const evidenceSummary = batchEvidenceSummaryHtml(batch);
  const staleWarning = batchUsesStaleControlledConditions(batch)
    ? `<div class="warning"><strong>Schema note</strong>: this batch uses controlled-condition values that differ from the current project controls. Reset and regenerate if it was created before the current design-space schema.</div>`
    : "";
  if (!strategy && !rationale && !constraints.length && !evidenceSummary && !staleWarning) {
    node.classList.add("empty");
    node.textContent = "Generate a batch to see the batch strategy.";
    return;
  }
  node.classList.remove("empty");
  const constraintText = constraints.length
    ? `<div><strong>Constraints</strong>: ${escapeHtml(constraints.join("; "))}</div>`
    : "";
  node.innerHTML = `
    <div><strong>Strategy</strong>: ${escapeHtml(strategy || "batch composition")}</div>
    ${rationale ? `<div><strong>Rationale</strong>: ${escapeHtml(rationale)}</div>` : ""}
    ${constraintText}
    ${evidenceSummary}
    ${staleWarning}
  `;
}

function candidateHtml(candidate) {
  const entries = Object.entries(candidate || {});
  if (!entries.length) return "";
  const lead = entries
    .slice(0, 3)
    .map(([key, value]) => `${key}: ${value}`)
    .join("; ");
  const suffix = entries.length > 3 ? ` +${entries.length - 3}` : "";
  const detailRows = entries
    .map(
      ([key, value]) =>
        `<dt>${escapeHtml(key)}</dt><dd>${escapeHtml(value)}</dd>`
    )
    .join("");
  return `
    <details class="cell-details">
      <summary>${escapeHtml(lead)}${escapeHtml(suffix)}</summary>
      <dl class="kv-list">${detailRows}</dl>
    </details>
  `;
}

function recommendationEvidenceRefs(item) {
  const slotRefs = item?.batch_slot?.batch_slot_evidence_refs || item?.batch_slot_evidence_refs || [];
  const values = slotRefs.length ? slotRefs : item?.evidence_refs || [];
  return Array.from(new Set((values || []).filter(Boolean)));
}

function batchEvidenceRefs(batch) {
  const refs = [];
  for (const item of batch?.recommendations || []) {
    refs.push(...recommendationEvidenceRefs(item));
  }
  return Array.from(new Set(refs));
}

function evidenceHtml(item) {
  const refs = recommendationEvidenceRefs(item);
  if (!refs.length) return `<span class="muted">No scoped evidence</span>`;
  const cards = refs.map((ref) => evidenceById[ref]).filter(Boolean);
  if (!cards.length) {
    return `
      <details class="cell-details evidence-details">
        <summary>${escapeHtml(`${refs.length} refs`)}</summary>
        <div class="evidence-list">${refs.map((ref) => `<code>${escapeHtml(ref)}</code>`).join("")}</div>
      </details>
    `;
  }
  const statuses = Array.from(new Set(cards.map((card) => card.mapping_status).filter(Boolean)));
  const uses = Array.from(new Set(cards.map((card) => card.allowed_use).filter(Boolean)));
  const summaryBits = [`${cards.length} refs`];
  if (statuses.length) summaryBits.push(statuses.slice(0, 2).join(", "));
  if (uses.length) summaryBits.push(uses.slice(0, 1).join(", "));
  return `
    <details class="cell-details evidence-details">
      <summary>${escapeHtml(summaryBits.join(" · "))}</summary>
      <div class="evidence-boundary">scoped advisory evidence · oracle disabled</div>
      <div class="evidence-list">${cards.map(evidenceCardHtml).join("")}</div>
    </details>
  `;
}

function batchEvidenceSummaryHtml(batch) {
  const refs = batchEvidenceRefs(batch);
  if (!refs.length) return "";
  const cards = refs.map((ref) => evidenceById[ref]).filter(Boolean);
  if (!cards.length) {
    return `<div><strong>Evidence Used</strong>: ${refs.length} scoped refs · oracle disabled</div>`;
  }
  const statuses = Array.from(new Set(cards.map((card) => card.mapping_status).filter(Boolean)));
  const targetNodes = Array.from(
    new Set(cards.flatMap((card) => card.target_nodes || []).filter(Boolean))
  );
  const variables = Array.from(
    new Set(cards.flatMap((card) => card.variable_scope || []).filter(Boolean))
  );
  return `
    <details class="cell-details batch-evidence-summary">
      <summary><strong>Evidence Used</strong>: ${escapeHtml(String(cards.length))} scoped cards · ${escapeHtml(statuses.join(", ") || "advisory")} · oracle disabled</summary>
      ${targetNodes.length ? `<div><strong>Supports</strong>: ${escapeHtml(targetNodes.slice(0, 6).join(", "))}${targetNodes.length > 6 ? " ..." : ""}</div>` : ""}
      ${variables.length ? `<div><strong>Variables</strong>: ${escapeHtml(variables.slice(0, 6).join(", "))}${variables.length > 6 ? " ..." : ""}</div>` : ""}
      <div class="evidence-list compact-list">${cards.map(evidenceCardHtml).join("")}</div>
    </details>
  `;
}

function evidenceCardHtml(card) {
  const variables = (card.variable_scope || []).join(", ");
  const targets = (card.target_nodes || []).join(", ");
  const source = card.source || card.doi || card.card_id || "evidence";
  const summary = card.summary || "";
  const excerpt = card.supporting_excerpt || "";
  const transfer = card.transferability_note || "";
  return `
    <article class="evidence-card">
      <div class="evidence-card-title">${escapeHtml(source)}</div>
      <div class="evidence-tags">
        ${evidenceTag(card.mapping_status)}
        ${evidenceTag(card.allowed_use)}
        ${evidenceTag(card.leakage_risk)}
      </div>
      ${variables ? `<div class="mini-note"><strong>Scope</strong>: ${escapeHtml(variables)}</div>` : ""}
      ${targets ? `<div class="mini-note"><strong>Supports</strong>: ${escapeHtml(targets)}</div>` : ""}
      ${summary ? `<div>${escapeHtml(summary)}</div>` : ""}
      ${transfer ? `<div class="mini-note"><strong>Transferability</strong>: ${escapeHtml(transfer)}</div>` : ""}
      ${excerpt ? `<details class="cell-details excerpt-details"><summary>Excerpt</summary><div>${escapeHtml(truncateText(excerpt, 520))}</div></details>` : ""}
      <code class="evidence-id">${escapeHtml(card.card_id || "")}</code>
    </article>
  `;
}

function evidenceTag(value) {
  const clean = String(value || "").trim();
  if (!clean) return "";
  return `<span class="evidence-tag">${escapeHtml(clean)}</span>`;
}

function descriptorHtml(contrast, signal) {
  if (!signal?.has_descriptor_contrast) {
    return "";
  }
  const entries = Object.entries(contrast || {}).filter(
    ([, payload]) => payload?.top_descriptor_deltas
  );
  if (!entries.length) return "";
  const summary = entries
    .slice(0, 1)
    .map(([variable, payload]) => {
      const keys = Object.keys(payload.top_descriptor_deltas || {}).slice(0, 2);
      return `${variable}: ${keys.join(", ")}`;
    })
    .join("; ");
  const detail = entries
    .map(([variable, payload]) => {
      const deltas = Object.entries(payload.top_descriptor_deltas || {})
        .slice(0, 5)
        .map(([key, value]) => `${key}: ${value}`)
        .join("; ");
      return `<dt>${escapeHtml(variable)}</dt><dd>${escapeHtml(payload.from || "")} -> ${escapeHtml(payload.to || "")}<br>${escapeHtml(deltas)}</dd>`;
    })
    .join("");
  return `
    <details class="cell-details">
      <summary>${escapeHtml(summary)}</summary>
      <dl class="kv-list">${detail}</dl>
    </details>
  `;
}

function rationaleHtml(text) {
  const clean = String(text || "").trim();
  if (!clean) return "";
  const limit = 220;
  if (clean.length <= limit) return escapeHtml(clean);
  const short = clean.slice(0, limit).trimEnd();
  return `
    <div class="short-text">${escapeHtml(short)}...</div>
    <details class="cell-details">
      <summary>Full rationale</summary>
      <div>${escapeHtml(clean)}</div>
    </details>
  `;
}

function batchUsesStaleControlledConditions(batch) {
  const controlled = currentProject?.design_space?.controlled_conditions || [];
  if (!controlled.length || !batch?.recommendations?.length) return false;
  const expected = new Map(
    controlled.map((item) => [item.name, normalizeConditionValue(item.fixed_value)])
  );
  for (const recommendation of batch.recommendations || []) {
    const candidate = recommendation.candidate || {};
    for (const [name, value] of Object.entries(candidate)) {
      if (!expected.has(name)) continue;
      if (normalizeConditionValue(value) !== expected.get(name)) {
        return true;
      }
    }
  }
  return false;
}

function normalizeConditionValue(value) {
  return String(value ?? "")
    .trim()
    .replace(/\s+/g, " ")
    .toLowerCase();
}

function isTerminalStatus(status) {
  return ["completed", "failed", "skipped"].includes(String(status || "").toLowerCase());
}

function statusCellHtml(item) {
  const terminal = isTerminalStatus(item.status);
  const disabled = terminal ? " disabled" : "";
  const current = terminal ? String(item.status || "") : "pending";
  const options = ["pending", "completed", "failed", "skipped"]
    .map((status) => `<option value="${status}"${current === status ? " selected" : ""}>${status}</option>`)
    .join("");
  return `
    <div class="status-cell">
      <div class="status-label">${escapeHtml(item.status || "pending")}</div>
      <select class="status-select" data-status-rec="${escapeHtml(item.recommendation_id || "")}" data-terminal="${terminal ? "true" : "false"}"${disabled}>
        ${options}
      </select>
    </div>
  `;
}

function resultInputHtml(item) {
  const terminal = isTerminalStatus(item.status);
  const disabled = terminal ? " disabled" : "";
  return `<input class="result-input" data-rec="${escapeHtml(item.recommendation_id || "")}" data-terminal="${terminal ? "true" : "false"}" placeholder="yield" value="${escapeHtml(item.result || "")}"${disabled}>`;
}

function failureInputHtml(item) {
  const terminal = isTerminalStatus(item.status);
  const disabled = terminal ? " disabled" : "";
  return `<input class="failure-input" data-failure="${escapeHtml(item.recommendation_id || "")}" data-terminal="${terminal ? "true" : "false"}" placeholder="failure reason" value="${escapeHtml(item.failure_reason || "")}"${disabled}>`;
}

function notesInputHtml(item) {
  const terminal = isTerminalStatus(item.status);
  const disabled = terminal ? " disabled" : "";
  return `<input class="notes-input" data-note="${escapeHtml(item.recommendation_id || "")}" data-terminal="${terminal ? "true" : "false"}" placeholder="notes" value="${escapeHtml(item.notes || "")}"${disabled}>`;
}

function traceLinkHtml(roundId, item) {
  if (!roundId || !item?.recommendation_id || !hasLoadedProject()) return "";
  const url = `/api/projects/${encodeURIComponent(projectId())}/trace/${encodeURIComponent(roundId)}`;
  const reflection = item.reflection ? `<div class="mini-note">reflection saved</div>` : "";
  return `<a href="${url}" target="_blank" rel="noreferrer">trace</a>${reflection}`;
}

async function submitResults() {
  if (!latestBatch?.recommendations?.length) return setStatus("No recommendations to submit.", true);
  const results = collectResultRows();
  if (!results.length) return setStatus("Enter at least one result or note.", true);
  return withUiLock("Submitting results...", async () => {
    try {
      const data = await api(`/api/projects/${encodeURIComponent(projectId())}/tell`, {
        method: "POST",
        body: JSON.stringify({ results, defer_reflection: true }),
      });
      await refreshProjectView();
      const reflectionNote =
        data.reflection_status === "deferred" ? " Reflection will be added in the background." : "";
      setStatus(`Imported ${data.appended.length} results.${reflectionNote}`);
    } catch (error) {
      setStatus(error.message, true);
    }
  });
}

async function resetProject() {
  const id = projectId();
  if (!id) return setStatus("Project ID is required.", true);
  const confirmed = window.confirm(
    `Reset run state for project "${id}"?\n\n` +
      "This will clear pending/completed recommendations, traces, " +
      "and local third-party logs. Project setup, design space, and evidence cards " +
      "will be preserved. Historical observations are kept only if the checkbox is enabled. " +
      "A backup will be created before clearing."
  );
  if (!confirmed) return null;
  return withUiLock("Resetting project run state...", async () => {
    try {
      const data = await api(`/api/projects/${encodeURIComponent(id)}/reset`, {
        method: "POST",
        body: JSON.stringify({
          backup: true,
          keep_historical: Boolean($("#keepHistoricalOnReset")?.checked),
        }),
      });
      await refreshProjectView();
      const backupNote = data.backup_dir ? ` Backup: ${data.backup_dir}` : "";
      setStatus(`Run state reset.${backupNote}`);
    } catch (error) {
      setStatus(error.message, true);
    }
    return null;
  });
}

async function importHistoricalObservations() {
  if (!hasLoadedProject()) return setStatus("Load a project first.", true);
  const file = $("#historicalCsv").files?.[0];
  if (!file) return setStatus("Choose a historical observations CSV first.", true);
  return withUiLock("Importing historical observations...", async () => {
    try {
      const text = await file.text();
      const rows = parseCsv(text);
      const data = await api(`/api/projects/${encodeURIComponent(projectId())}/observations/import`, {
        method: "POST",
        body: JSON.stringify({ rows, source: "historical" }),
      });
      $("#historicalCsv").value = "";
      await refreshProjectView();
      const skipped = data.skipped_count ? ` Skipped ${data.skipped_count} duplicate rows.` : "";
      setStatus(`Imported ${data.imported_count} historical observations.${skipped}`);
    } catch (error) {
      setStatus(error.message, true);
    }
  });
}

async function loadObservations() {
  const id = projectId();
  if (!id) return;
  try {
    const data = await api(`/api/projects/${encodeURIComponent(id)}/observations`);
    renderObservations(data.observations || []);
    updateControls();
  } catch (error) {
    $("#observations").textContent = error.message;
    $("#observations").classList.add("empty");
    updateControls();
  }
}

async function reloadObservations() {
  if (!hasLoadedProject()) return setStatus("Load a project first.", true);
  return withUiLock("Reloading observations...", async () => {
    await loadObservations();
    setStatus("Observations reloaded.");
  });
}

function renderObservations(rows) {
  const node = $("#observations");
  if (!rows.length) {
    node.classList.add("empty");
    node.textContent = "No observations yet.";
    return;
  }
  node.classList.remove("empty");
  const columns = Object.keys(rows[0]);
  node.innerHTML = table(columns, rows.map((row) => columns.map((column) => row[column] ?? "")));
}

async function refreshProjects() {
  return withUiLock("Loading project list...", async () => {
    try {
      const data = await api("/api/projects");
      const ids = (data.projects || []).map((item) => item.project?.project_id).filter(Boolean);
      setStatus(ids.length ? `Projects: ${ids.join(", ")}` : "No projects found.");
    } catch (error) {
      setStatus(error.message, true);
    }
  });
}

async function createProjectFromCsv() {
  const file = $("#designCsv").files?.[0];
  const project_id = $("#newProjectId").value.trim();
  if (!file || !project_id) return setStatus("Project ID and design CSV are required.", true);
  return withUiLock("Creating project...", async () => {
    try {
      const text = await file.text();
      const design_records = parseCsv(text);
      const payload = {
        project_id,
        config: {
          project_id,
          objective_name: $("#objectiveName").value.trim() || "yield",
          planner_name: "atlas",
          controller_mode: "agentic",
          batch_size: Number.parseInt($("#batchSize").value || "6", 10),
        },
        design_records,
      };
      const summary = await api("/api/projects", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      $("#projectId").value = project_id;
      currentProject = summary;
      await loadEvidenceIndex();
      renderProject(summary);
      await Promise.all([loadRecommendations(), loadObservations()]);
      setStatus("Project created.");
    } catch (error) {
      setStatus(error.message, true);
    }
  });
}

function parseCsv(text) {
  const lines = text.split(/\r?\n/).filter((line) => line.trim());
  const headers = splitCsvLine(lines.shift() || []);
  return lines.map((line) => {
    const cells = splitCsvLine(line);
    return Object.fromEntries(headers.map((header, index) => [header, cells[index] || ""]));
  });
}

function splitCsvLine(line) {
  const cells = [];
  let current = "";
  let quoted = false;
  for (let i = 0; i < line.length; i += 1) {
    const char = line[i];
    if (char === '"' && line[i + 1] === '"') {
      current += '"';
      i += 1;
    } else if (char === '"') {
      quoted = !quoted;
    } else if (char === "," && !quoted) {
      cells.push(current.trim());
      current = "";
    } else {
      current += char;
    }
  }
  cells.push(current.trim());
  return cells;
}

function csvCell(value) {
  const text = typeof value === "object" && value !== null ? JSON.stringify(value) : String(value ?? "");
  if (/[",\n\r]/.test(text)) {
    return `"${text.replaceAll('"', '""')}"`;
  }
  return text;
}

function table(headers, rows, html = false) {
  const headerHtml = headers.map((header) => `<th>${escapeHtml(header)}</th>`).join("");
  const bodyHtml = rows
    .map(
      (row) =>
        `<tr>${row
          .map((cell) => `<td>${html ? cell : escapeHtml(String(cell ?? ""))}</td>`)
          .join("")}</tr>`
    )
    .join("");
  return `<table><thead><tr>${headerHtml}</tr></thead><tbody>${bodyHtml}</tbody></table>`;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function truncateText(value, limit) {
  const text = String(value || "").trim();
  if (text.length <= limit) return text;
  return `${text.slice(0, limit).trimEnd()}...`;
}

$("#loadProject").addEventListener("click", loadProject);
$("#refreshProjects").addEventListener("click", refreshProjects);
$("#askButton").addEventListener("click", askBatch);
$("#reloadRecommendations").addEventListener("click", reloadRecommendations);
$("#exportRecommendations").addEventListener("click", exportRecommendationsCsv);
$("#reloadObservations").addEventListener("click", reloadObservations);
$("#submitResults").addEventListener("click", submitResults);
$("#createProject").addEventListener("click", createProjectFromCsv);
$("#importHistorical").addEventListener("click", importHistoricalObservations);
$("#resetProject").addEventListener("click", resetProject);
document.addEventListener("input", (event) => {
  const target = event.target;
  if (
    target?.matches?.(
      "#projectId, #newProjectId, #designCsv, .result-input, .notes-input, .status-select, .failure-input"
    )
  ) {
    updateControls();
  }
});
document.addEventListener("change", (event) => {
  const target = event.target;
  if (target?.matches?.("#designCsv, #historicalCsv")) updateControls();
});
updateControls();
