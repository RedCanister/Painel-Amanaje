(function () {
    const ACTIVE_STATUSES = new Set(["queued", "running", "paused", "cancel_requested"]);
    const state = {
        runs: [],
        groups: { cpu: [], gpu: [] },
        selectedRunId: "",
        selectedDetail: null,
        workerRuntime: {},
        queueRuntime: {},
        accelerators: {},
        terminalCursor: 0,
        runPollTimer: null,
        terminalPollTimer: null,
        searchTimer: null,
        runController: null,
        terminalLoaded: false,
        analysisLoadedFor: "",
    };

    function el(id) {
        return document.getElementById(id);
    }

    function escapeHtml(value) {
        if (window.AmanajeUI?.escapeHtml) return window.AmanajeUI.escapeHtml(value);
        const div = document.createElement("div");
        div.textContent = String(value ?? "");
        return div.innerHTML;
    }

    async function fetchJson(url, options = {}, settings = {}) {
        if (window.AmanajeUI?.fetchJson) {
            return window.AmanajeUI.fetchJson(url, options, { source: "operations-page", ...settings });
        }
        const response = await fetch(url, options);
        const payload = await response.json();
        if (!response.ok || payload.status === "error") {
            throw new Error(payload.detail || `HTTP ${response.status}`);
        }
        return payload;
    }

    function formatDate(value) {
        if (!value) return "";
        const date = new Date(value);
        if (Number.isNaN(date.getTime())) return String(value);
        return date.toLocaleString();
    }

    function formatAge(seconds) {
        if (seconds === null || seconds === undefined || Number.isNaN(Number(seconds))) return "-";
        const value = Number(seconds);
        if (value < 60) return `${Math.round(value)}s`;
        if (value < 3600) return `${Math.round(value / 60)}m`;
        return `${Math.round(value / 3600)}h`;
    }

    function latestEvent(run) {
        return run?.status_reason || run?.progress?.latest_event || run?.failure?.message || run?.error || "";
    }

    function runStatus(run) {
        return String(run?.effective_status || run?.status || "queued");
    }

    function toneForRun(run) {
        const status = runStatus(run);
        if (status === "completed") return "completed";
        if (status === "failed" || status === "cancelled") return status;
        if (ACTIVE_STATUSES.has(status)) return status === "running" ? "running" : "queued";
        return "queued";
    }

    function buildQuery() {
        const params = new URLSearchParams();
        const status = el("operationsStatusFilter")?.value || "";
        const runType = el("operationsTypeFilter")?.value || "";
        const limit = el("operationsLimit")?.value || "120";
        const query = el("operationsSearch")?.value || "";
        params.set("view", "summary");
        if (status) params.set("status", status);
        if (runType) params.set("run_type", runType);
        if (limit) params.set("limit", limit);
        if (query) params.set("q", query);
        return params.toString();
    }

    function queueName(run) {
        return run?.queue_diagnostics?.queue || run?.queue?.queue || run?.queue?.name || "-";
    }

    function queuePosition(run) {
        const position = run?.queue_diagnostics?.position;
        return position ? `#${position}` : "-";
    }

    function renderSummary(payload) {
        const cpu = payload.groups?.cpu || [];
        const gpu = payload.groups?.gpu || [];
        const workerSummary = payload.worker_runtime?.summary || {};
        const queueSummary = payload.queue_runtime?.summary || {};
        const accelerator = payload.accelerators || {};
        const cuda = accelerator.cuda || {};
        const deviceCount = cuda.device_count ?? accelerator.device_count ?? 0;

        el("operationsRunCount").textContent = String((payload.runs || []).length);
        el("operationsCpuCount").textContent = String(cpu.length);
        el("operationsGpuCount").textContent = String(gpu.length);
        el("operationsCpuLabel").textContent = String(cpu.length);
        el("operationsGpuLabel").textContent = String(gpu.length);
        el("operationsWorkerSummary").textContent = `${workerSummary.live_workers || workerSummary.active_workers || 0} live, ${workerSummary.stale_workers || 0} stale`;
        el("operationsQueueSummary").textContent = `${queueSummary.queued_jobs || 0} queued, ${queueSummary.started_jobs || 0} running`;
        el("operationsAcceleratorSummary").textContent = cuda.available || accelerator.cuda_available
            ? `CUDA available (${deviceCount})`
            : "CPU only";
        el("operationsUpdated").textContent = `Updated ${new Date().toLocaleTimeString()}`;
    }

    function renderRunActions(run, compact = false) {
        const actions = run?.actions || {};
        const runId = escapeHtml(run?.run_id || "");
        const rerunTitle = escapeHtml(actions.rerun_reason || "");
        const labels = compact
            ? { pause: "Pause", resume: "Resume", cancel: "Cancel", rerun: "Re-run", analyze: "Analyze" }
            : { pause: "Pause", resume: "Resume", cancel: "Cancel", rerun: "Re-run", analyze: "Analyze" };
        return `
            ${actions.pause ? `<button type="button" data-operation-action="pause" data-run-id="${runId}">${labels.pause}</button>` : ""}
            ${actions.resume ? `<button type="button" data-operation-action="resume" data-run-id="${runId}">${labels.resume}</button>` : ""}
            ${actions.cancel ? `<button type="button" data-operation-action="cancel" data-run-id="${runId}">${labels.cancel}</button>` : ""}
            <button type="button" data-operation-action="rerun" data-run-id="${runId}" ${actions.rerun ? "" : "disabled"} title="${rerunTitle}">${labels.rerun}</button>
            <button type="button" data-operation-action="analyze" data-run-id="${runId}">${labels.analyze}</button>
        `;
    }

    function renderRunList(targetId, runs) {
        const target = el(targetId);
        if (!target) return;
        if (!runs.length) {
            target.innerHTML = '<div class="operations-empty">No operations match this lane.</div>';
            return;
        }
        target.innerHTML = runs.map((run) => {
            const runId = String(run.run_id || "");
            const active = runId === state.selectedRunId ? " active" : "";
            return `
                <article class="operation-run-card ${toneForRun(run)}${active}" data-run-select="${escapeHtml(runId)}">
                    <div class="operation-run-main">
                        <strong>${escapeHtml(run.title || run.run_type || "operation")}</strong>
                        <span class="pill">${escapeHtml(runStatus(run))}</span>
                    </div>
                    <div class="operation-run-meta">
                        <code>${escapeHtml(runId)}</code>
                        <span>${escapeHtml(run.stage || "-")}</span>
                        <span>${escapeHtml(queueName(run))} ${escapeHtml(queuePosition(run))}</span>
                        <span>${escapeHtml(run.compute_reason || "")}</span>
                    </div>
                    ${latestEvent(run) ? `<div class="operation-run-event">${escapeHtml(latestEvent(run))}</div>` : ""}
                    <div class="operation-run-actions">
                        ${renderRunActions(run, true)}
                    </div>
                </article>
            `;
        }).join("");
        target.querySelectorAll("[data-run-select]").forEach((card) => {
            card.addEventListener("click", (event) => {
                if (event.target.closest("button")) return;
                selectRun(card.dataset.runSelect);
            });
        });
        target.querySelectorAll("[data-operation-action]").forEach((button) => {
            button.addEventListener("click", () => handleAction(button.dataset.runId, button.dataset.operationAction));
        });
    }

    function selectedRun() {
        return state.runs.find((run) => String(run.run_id) === String(state.selectedRunId)) || null;
    }

    function renderDetail(run) {
        if (!run) {
            el("operationsDetailTitle").textContent = "Select an operation";
            el("operationsDetailSubtitle").textContent = "Choose a run from either lane to inspect execution output and analysis.";
            el("operationsDetailActions").innerHTML = "";
            el("operationsTimeline").innerHTML = "";
            el("operationsFailure").innerHTML = "";
            el("operationsArtifacts").innerHTML = "";
            el("operationsRaw").innerHTML = "";
            return;
        }
        el("operationsDetailTitle").textContent = run.title || run.run_type || run.run_id;
        el("operationsDetailSubtitle").textContent = `${run.run_id} | ${runStatus(run)} | ${queueName(run)} ${queuePosition(run)} | ${formatDate(run.updated_at || run.created_at)}`;
        el("operationsDetailActions").innerHTML = renderRunActions(run);
        el("operationsDetailActions").querySelectorAll("[data-operation-action]").forEach((button) => {
            button.addEventListener("click", () => handleAction(button.dataset.runId, button.dataset.operationAction));
        });
        renderTimeline(run);
        renderFailure(run);
        renderArtifacts(run, null);
        renderRaw(run);
    }

    function renderTimeline(run) {
        const progressTimeline = Array.isArray(run?.progress?.timeline) ? run.progress.timeline : [];
        const events = Array.isArray(run?.events) ? run.events : [];
        const timeline = [...events, ...progressTimeline].filter(Boolean);
        const target = el("operationsTimeline");
        if (!timeline.length) {
            target.innerHTML = '<div class="operations-empty">No timeline entries are available for the loaded view.</div>';
            return;
        }
        target.innerHTML = timeline.map((event) => `
            <div class="operations-analysis-card">
                <h4>${escapeHtml(event.stage || event.status || "event")}</h4>
                <div class="helper-text">${escapeHtml(formatDate(event.timestamp))}</div>
                <p>${escapeHtml(event.message || "")}</p>
            </div>
        `).join("");
    }

    function renderFailure(run) {
        const target = el("operationsFailure");
        if (!target) return;
        const failure = run?.failure || {};
        const traceback = failure.traceback || run?.result?.error || "";
        if (!failure.message && !run?.error && !traceback) {
            target.innerHTML = '<div class="operations-empty">No failure has been recorded for this run.</div>';
            return;
        }
        target.innerHTML = `
            <div class="operations-analysis-card operations-failure-card">
                <h4>${escapeHtml(failure.type || "Failure")}</h4>
                <p>${escapeHtml(failure.message || run.error || "The operation failed.")}</p>
                <div class="operations-failure-meta">
                    <span>Queue: ${escapeHtml(failure.queue || queueName(run))}</span>
                    <span>Worker: ${escapeHtml(failure.worker_name || "-")}</span>
                    <span>At: ${escapeHtml(formatDate(failure.failed_at))}</span>
                </div>
                ${traceback ? `<pre class="operations-traceback">${escapeHtml(traceback)}</pre>` : ""}
            </div>
        `;
    }

    function renderArtifacts(run, analysis) {
        const artifacts = analysis?.artifacts || run?.artifacts || {};
        const plots = analysis?.plots || run?.plots || [];
        const target = el("operationsArtifacts");
        target.innerHTML = `
            <div class="operations-analysis-card">
                <h4>Artifacts</h4>
                ${renderJson(artifacts, "Artifact Map")}
            </div>
            <div class="operations-analysis-card">
                <h4>Plots</h4>
                ${Array.isArray(plots) && plots.length ? renderJson(plots, "Plot Artifacts") : '<div class="helper-text">No plot artifacts are linked to this run.</div>'}
            </div>
        `;
        window.AmanajeUI?.enhanceCollapsibles?.(target);
    }

    function renderRaw(run) {
        el("operationsRaw").innerHTML = renderJson(run || {}, "Raw Run Ledger");
        window.AmanajeUI?.enhanceCollapsibles?.(el("operationsRaw"));
    }

    function renderJson(value, title) {
        if (window.AmanajeUI?.renderCollapsibleJson) {
            return window.AmanajeUI.renderCollapsibleJson(value, { title, collapsed: true });
        }
        return `<pre class="json-block">${escapeHtml(JSON.stringify(value, null, 2))}</pre>`;
    }

    function renderQueueBacklog(queueRuntime = {}) {
        const target = el("operationsQueueBacklog");
        if (!target) return;
        const queues = Array.isArray(queueRuntime.queues) ? queueRuntime.queues : [];
        if (!queues.length) {
            target.innerHTML = '<div class="operations-empty">No queue runtime data is available.</div>';
            return;
        }
        target.innerHTML = queues.map((queue) => {
            const jobs = Array.isArray(queue.jobs) ? queue.jobs : [];
            return `
                <article class="operations-analysis-card">
                    <h4>${escapeHtml(queue.name || "queue")}</h4>
                    <div class="operations-failure-meta">
                        <span>${escapeHtml(queue.queued_jobs || 0)} queued</span>
                        <span>${escapeHtml(queue.started_jobs || 0)} running</span>
                        <span>${escapeHtml(queue.failed_jobs || 0)} failed</span>
                    </div>
                    ${jobs.length ? `
                        <ol class="operations-job-list">
                            ${jobs.slice(0, 12).map((job) => `<li><code>${escapeHtml(job.job_id)}</code><span>${escapeHtml(job.status || "queued")}</span></li>`).join("")}
                        </ol>
                    ` : '<div class="helper-text">No queued jobs.</div>'}
                </article>
            `;
        }).join("");
    }

    function renderWorkerHosts(workerRuntime = {}) {
        const target = el("operationsWorkerHosts");
        if (!target) return;
        const workers = [
            ...(Array.isArray(workerRuntime.live_workers) ? workerRuntime.live_workers : []),
            ...(Array.isArray(workerRuntime.stale_workers) ? workerRuntime.stale_workers : []),
        ];
        if (!workers.length) {
            target.innerHTML = '<div class="operations-empty">No workers are registered.</div>';
            return;
        }
        target.innerHTML = workers.map((worker) => `
            <article class="operations-worker-card ${worker.stale ? "stale" : "live"}">
                <div>
                    <strong>${escapeHtml(worker.name || "worker")}</strong>
                    <span>${escapeHtml(worker.stale ? "stale" : "live")} | ${escapeHtml(worker.state || "unknown")}</span>
                </div>
                <div class="operations-failure-meta">
                    <span>Queues: ${escapeHtml((worker.queues || []).join(", ") || "-")}</span>
                    <span>Current: ${escapeHtml(worker.current_job_id || "-")}</span>
                    <span>Heartbeat: ${escapeHtml(formatAge(worker.heartbeat_age_seconds))}</span>
                    <span>Failed: ${escapeHtml(worker.failed_job_count || 0)}</span>
                </div>
            </article>
        `).join("");
    }

    function renderRecentFailures(runs = []) {
        const target = el("operationsRecentFailures");
        if (!target) return;
        const failures = runs.filter((run) => String(run.status || "") === "failed" || run.failure?.message || run.error).slice(0, 8);
        if (!failures.length) {
            target.innerHTML = '<div class="operations-empty">No recent failures in the current result set.</div>';
            return;
        }
        target.innerHTML = failures.map((run) => `
            <button type="button" class="operations-failure-row" data-run-select="${escapeHtml(run.run_id)}">
                <strong>${escapeHtml(run.title || run.run_id)}</strong>
                <span>${escapeHtml(run.failure?.message || run.error || latestEvent(run))}</span>
            </button>
        `).join("");
        target.querySelectorAll("[data-run-select]").forEach((button) => {
            button.addEventListener("click", () => selectRun(button.dataset.runSelect));
        });
    }

    function appendTerminalLines(lines, reset = false) {
        const terminal = el("operationsTerminal");
        if (!terminal) return;
        const usable = Array.isArray(lines) ? lines : [];
        const shouldStick = terminal.scrollTop + terminal.clientHeight >= terminal.scrollHeight - 48;
        if (reset) terminal.innerHTML = "";
        if (!usable.length && reset) {
            terminal.innerHTML = '<div class="helper-text">No terminal lines are available for this run yet.</div>';
        } else if (usable.length) {
            const html = usable.map((line) => `
                <div class="operations-terminal-line ${escapeHtml(line.stream || "system")}">
                    <span class="operations-terminal-time">${escapeHtml(formatDate(line.timestamp))}</span>
                    <span class="operations-terminal-stream">${escapeHtml(line.stream || "system")}</span>
                    <span class="operations-terminal-stage">${escapeHtml(line.stage || "-")}</span>
                    <span class="operations-terminal-message">${escapeHtml(line.message || "")}</span>
                </div>
            `).join("");
            if (reset) terminal.innerHTML = html;
            else terminal.insertAdjacentHTML("beforeend", html);
        }
        if (shouldStick || reset) terminal.scrollTop = terminal.scrollHeight;
    }

    async function loadTerminal(reset = false) {
        if (!state.selectedRunId) return;
        const cursor = reset ? 0 : state.terminalCursor;
        const payload = await fetchJson(`/operations/runs/${encodeURIComponent(state.selectedRunId)}/terminal?cursor=${cursor}&tail=300`, {}, { allowError: true });
        state.terminalCursor = Number(payload.next_cursor || state.terminalCursor || 0);
        state.terminalLoaded = true;
        appendTerminalLines(payload.lines || [], reset);
        el("operationsTerminalStatus").textContent = `${runStatus(payload.run) || "loaded"} | next ${state.terminalCursor}`;
    }

    function renderAnalysis(payload) {
        const analysis = payload.analysis || {};
        const cards = [
            ["Run Metrics", analysis.metrics || {}],
            ["Model Analysis", analysis.model || {}],
            ["Dataset Analysis", analysis.dataset || {}],
            ["Inference Context", analysis.inference || {}],
            ["Production Context", analysis.production || {}],
            ["Study Analysis", analysis.study || {}],
        ];
        el("operationsAnalysis").innerHTML = cards.map(([title, value]) => `
            <div class="operations-analysis-card">
                <h4>${escapeHtml(title)}</h4>
                ${renderJson(value, title)}
            </div>
        `).join("");
        window.AmanajeUI?.enhanceCollapsibles?.(el("operationsAnalysis"));
        renderArtifacts(payload.run, analysis);
    }

    async function loadAnalysis(force = false) {
        if (!state.selectedRunId) return;
        if (!force && state.analysisLoadedFor === state.selectedRunId) return;
        el("operationsAnalysis").innerHTML = '<div class="operations-empty">Loading analysis.</div>';
        const payload = await fetchJson(`/operations/runs/${encodeURIComponent(state.selectedRunId)}/analysis`);
        state.analysisLoadedFor = state.selectedRunId;
        renderAnalysis(payload);
    }

    async function loadRunDetail(runId) {
        if (!runId) return null;
        const summary = selectedRun();
        const detail = await fetchJson(`/runs/get/${encodeURIComponent(runId)}`, {}, { source: "operations-detail" });
        state.selectedDetail = { ...(summary || {}), ...detail, title: summary?.title || detail.run_type || detail.run_id, actions: summary?.actions || {} };
        renderDetail(state.selectedDetail);
        return state.selectedDetail;
    }

    function setActiveTab(tabName) {
        document.querySelectorAll("[data-operations-tab]").forEach((button) => {
            button.classList.toggle("active", button.dataset.operationsTab === tabName);
        });
        ["terminal", "timeline", "failure", "analysis", "artifacts", "raw"].forEach((name) => {
            const panel = el(`operationsTab${name.charAt(0).toUpperCase()}${name.slice(1)}`);
            if (panel) panel.hidden = name !== tabName;
        });
        if (tabName === "analysis") loadAnalysis().catch((error) => {
            el("operationsAnalysis").innerHTML = `<div class="alert alert-error">${escapeHtml(error.message)}</div>`;
        });
        if (["timeline", "failure", "raw", "artifacts"].includes(tabName) && !state.selectedDetail && state.selectedRunId) {
            loadRunDetail(state.selectedRunId).catch((error) => {
                el("operationsRaw").innerHTML = `<div class="alert alert-error">${escapeHtml(error.message)}</div>`;
            });
        }
    }

    async function handleAction(runId, action) {
        if (!runId || !action) return;
        if (action === "analyze") {
            selectRun(runId);
            setActiveTab("analysis");
            await loadAnalysis(true);
            return;
        }
        if (action === "rerun") {
            const result = await fetchJson(`/runs/${encodeURIComponent(runId)}/rerun`, { method: "POST" });
            if (result.run_id) {
                window.AmanajeUI?.watchRun?.(result.run_id, { source: "operations-rerun", source_run_id: runId });
                state.selectedRunId = result.run_id;
            }
            await loadRuns();
            return;
        }
        if (window.AmanajeUI?.controlRun) {
            await window.AmanajeUI.controlRun(runId, action);
        } else {
            await fetchJson(`/runs/${encodeURIComponent(runId)}/${encodeURIComponent(action)}`, { method: "POST" });
        }
        await loadRuns();
    }

    async function selectRun(runId) {
        state.selectedRunId = String(runId || "");
        state.selectedDetail = null;
        state.terminalCursor = 0;
        state.terminalLoaded = false;
        state.analysisLoadedFor = "";
        renderRunList("operationsCpuList", state.groups.cpu || []);
        renderRunList("operationsGpuList", state.groups.gpu || []);
        renderDetail(selectedRun());
        await Promise.allSettled([
            loadTerminal(true),
            loadRunDetail(state.selectedRunId),
        ]);
    }

    async function loadRuns() {
        if (state.runController) state.runController.abort();
        const controller = new AbortController();
        state.runController = controller;
        const query = buildQuery();
        try {
            const payload = await fetchJson(`/operations/runs${query ? `?${query}` : ""}`, { signal: controller.signal });
            state.runs = payload.runs || [];
            state.groups = payload.groups || { cpu: [], gpu: [] };
            state.workerRuntime = payload.worker_runtime || {};
            state.queueRuntime = payload.queue_runtime || {};
            state.accelerators = payload.accelerators || {};
            renderSummary(payload);
            renderQueueBacklog(state.queueRuntime);
            renderWorkerHosts(state.workerRuntime);
            renderRecentFailures(state.runs);
            if (!state.selectedRunId || !state.runs.some((run) => String(run.run_id) === state.selectedRunId)) {
                const firstActive = state.runs.find((run) => ACTIVE_STATUSES.has(runStatus(run)));
                state.selectedRunId = String((firstActive || state.runs[0] || {}).run_id || "");
                state.selectedDetail = null;
                state.terminalCursor = 0;
                state.terminalLoaded = false;
                state.analysisLoadedFor = "";
            }
            renderRunList("operationsCpuList", state.groups.cpu || []);
            renderRunList("operationsGpuList", state.groups.gpu || []);
            renderDetail(state.selectedDetail || selectedRun());
            if (state.selectedRunId && !state.terminalLoaded) {
                await loadTerminal(true).catch(() => {});
            }
            schedulePolling();
        } catch (error) {
            if (error?.name === "AbortError") return;
            showError(error);
        }
    }

    function schedulePolling() {
        if (state.runPollTimer) window.clearTimeout(state.runPollTimer);
        if (state.terminalPollTimer) window.clearTimeout(state.terminalPollTimer);
        const live = el("operationsLiveToggle")?.checked;
        if (!live || document.hidden) return;
        const hasActive = state.runs.some((run) => ACTIVE_STATUSES.has(runStatus(run)));
        state.runPollTimer = window.setTimeout(() => loadRuns(), hasActive ? 3500 : 10000);
        if (state.selectedRunId && hasActive) {
            state.terminalPollTimer = window.setTimeout(() => loadTerminal(false).catch(showError), 2500);
        }
    }

    function showError(error) {
        if (error?.name === "AbortError") return;
        el("operationsUpdated").textContent = error.message || String(error);
    }

    function debounceLoadRuns() {
        if (state.searchTimer) window.clearTimeout(state.searchTimer);
        state.searchTimer = window.setTimeout(() => {
            state.selectedRunId = "";
            state.selectedDetail = null;
            state.terminalCursor = 0;
            state.terminalLoaded = false;
            loadRuns();
        }, 250);
    }

    function bindEvents() {
        el("operationsRefreshButton")?.addEventListener("click", () => loadRuns());
        el("operationsSearch")?.addEventListener("input", debounceLoadRuns);
        ["operationsStatusFilter", "operationsTypeFilter", "operationsLimit"].forEach((id) => {
            el(id)?.addEventListener("change", debounceLoadRuns);
        });
        el("operationsLiveToggle")?.addEventListener("change", schedulePolling);
        document.addEventListener("visibilitychange", schedulePolling);
        document.querySelectorAll("[data-operations-tab]").forEach((button) => {
            button.addEventListener("click", () => setActiveTab(button.dataset.operationsTab));
        });
    }

    document.addEventListener("DOMContentLoaded", () => {
        bindEvents();
        loadRuns();
    });
})();
