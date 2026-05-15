(function () {
    const PANEL_OBJECTIVE = "Understand the current question through connected data, models, plots, simulations, metrics, and notes.";
    const PANEL_MODE_STORAGE_KEY = "amanajePanelMode";
    const SOURCE_COLLECTIONS = {
        dataset: ["datasets"],
        dash_workspace: [],
        learning_model: ["learning_models"],
        plot: ["plots"],
        study: ["studies"],
        inference: ["inferences"],
        simulation: ["inferences"],
        prediction: ["inferences"],
        metric: ["learning_models", "runs"],
        metadata: ["datasets", "learning_models", "studies", "inferences", "code_models", "runs"],
        note: [],
        custom_json: [],
    };

    const state = {
        dashboards: [],
        current: null,
        context: emptyContext(),
        dirty: false,
        mode: "run",
    };

    function emptyContext() {
        return {
            datasets: [],
            learning_models: [],
            studies: [],
            inferences: [],
            code_models: [],
            plots: [],
            runs: [],
        };
    }

    function escapeHtml(value) {
        if (window.AmanajeUI?.escapeHtml) return window.AmanajeUI.escapeHtml(value);
        return String(value ?? "")
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#39;");
    }

    async function fetchJson(url, options = {}) {
        if (window.AmanajeUI?.fetchJson) {
            return window.AmanajeUI.fetchJson(url, options, { source: "panel-page" });
        }
        const response = await fetch(url, {
            ...options,
            headers: {
                ...(options.body ? { "Content-Type": "application/json" } : {}),
                ...(options.headers || {}),
            },
        });
        const payload = await response.json();
        if (!response.ok || payload.status === "error") throw new Error(payload.detail || `HTTP ${response.status}`);
        return payload;
    }

    function byId(id) {
        return document.getElementById(id);
    }

    function showStatus(message, type = "info") {
        const target = byId("statusMessage");
        if (!target) return;
        target.textContent = message || "";
        target.className = `statusMessage panel-status-line ${type}`;
    }

    function dashboardUrl() {
        return (window.AmanajeSettings?.services?.plotly_dashboard_url || "http://localhost:8050").replace(/\/$/, "");
    }

    function dashPath(value) {
        const raw = String(value || "/").trim() || "/";
        return raw.startsWith("/") ? raw : `/${raw}`;
    }

    function dashUrlForPath(value) {
        return `${dashboardUrl()}${dashPath(value)}`;
    }

    function encodeFigure(figure) {
        const json = JSON.stringify(figure || {});
        const encoded = btoa(unescape(encodeURIComponent(json)));
        return encoded.replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/g, "");
    }

    function normalizeWidgetSize(size) {
        return String(size || "").trim().toLowerCase() === "full" ? "full" : "wide";
    }

    function normalizePanelMode(value, fallback = "run") {
        return value === "edit" || value === "run" ? value : fallback;
    }

    function readPreferredPanelMode(fallback) {
        try {
            return normalizePanelMode(window.localStorage.getItem(PANEL_MODE_STORAGE_KEY), fallback);
        } catch (_error) {
            return fallback;
        }
    }

    function persistPanelMode(mode) {
        try {
            window.localStorage.setItem(PANEL_MODE_STORAGE_KEY, mode);
        } catch (_error) {
            return;
        }
    }

    function isEditMode() {
        return state.mode === "edit";
    }

    function setPanelMode(mode, options = {}) {
        const persist = options.persist !== false;
        state.mode = normalizePanelMode(mode, "run");
        if (persist) persistPanelMode(state.mode);
        applyPanelMode();
        renderWidgetDeck();
    }

    function applyPanelMode() {
        const page = byId("panelPage");
        if (page) page.dataset.panelMode = state.mode;
        document.querySelectorAll("[data-panel-mode-button]").forEach((button) => {
            const active = button.getAttribute("data-panel-mode-button") === state.mode;
            button.classList.toggle("active", active);
            button.setAttribute("aria-pressed", active ? "true" : "false");
        });
    }

    function normalizeDashboard(dashboard) {
        const normalized = dashboard && typeof dashboard === "object" ? { ...dashboard } : newLocalDashboard();
        normalized.widgets = Array.isArray(normalized.widgets)
            ? normalized.widgets.map((widget) => ({
                ...widget,
                size: normalizeWidgetSize(widget?.size),
                source: widget?.source && typeof widget.source === "object" ? widget.source : {},
                settings: widget?.settings && typeof widget.settings === "object" ? widget.settings : {},
                cache: widget?.cache && typeof widget.cache === "object" ? widget.cache : {},
            }))
            : [];
        normalized.layout = {
            ...(normalized.layout || {}),
            version: 1,
            columns: 12,
            order: normalized.widgets.map((widget) => widget.id).filter(Boolean),
        };
        return normalized;
    }

    function defaultWidgets() {
        return [
            {
                id: "objective_focus",
                kind: "metadata",
                title: "Objective",
                size: "full",
                source: {},
                settings: { mode: "objective" },
                cache: {},
            },
            {
                id: "dash_visualization_studio",
                kind: "dash_workspace",
                title: "Dash Visualization Studio",
                size: "full",
                source: {},
                settings: {
                    path: "/",
                    description: "Interactive Dash workspace for plots, extensions, and panel visualizations.",
                },
                cache: {},
            },
            {
                id: "dataset_signal",
                kind: "dataset",
                title: "Dataset Signal",
                size: "wide",
                source: {},
                settings: { variant: "summary" },
                cache: {},
            },
            {
                id: "model_signal",
                kind: "learning_model",
                title: "Learning Model Signal",
                size: "wide",
                source: {},
                settings: { variant: "summary" },
                cache: {},
            },
            {
                id: "plot_signal",
                kind: "plot",
                title: "Plot Artifact",
                size: "wide",
                source: {},
                settings: { variant: "dash" },
                cache: {},
            },
            {
                id: "metric_stack",
                kind: "metric",
                title: "Metric Stack",
                size: "wide",
                source: {},
                settings: { metrics: [] },
                cache: {},
            },
            {
                id: "interpretation_notes",
                kind: "note",
                title: "Interpretation Notes",
                size: "wide",
                source: {},
                settings: { text: "" },
                cache: {},
            },
        ];
    }

    function newLocalDashboard() {
        const widgets = defaultWidgets();
        return {
            id: null,
            name: "Amanaje Panel",
            description: "Saved Painel Amanaje objective dashboard.",
            objective: PANEL_OBJECTIVE,
            tint: "amanaje",
            layout: { version: 1, columns: 12, density: "comfortable", order: widgets.map((widget) => widget.id) },
            widgets,
            panel_metadata: {},
        };
    }

    function currentWidgets() {
        if (!state.current) return [];
        if (!Array.isArray(state.current.widgets)) state.current.widgets = [];
        return state.current.widgets;
    }

    function syncCurrentFromForm() {
        if (!state.current) return;
        state.current.name = byId("panelDashboardName")?.value?.trim() || "Amanaje Panel";
        state.current.objective = byId("panelObjective")?.value?.trim() || "";
        state.current.tint = "amanaje";
        state.current.widgets = currentWidgets().map((widget) => ({
            ...widget,
            size: normalizeWidgetSize(widget.size),
        }));
        state.current.layout = {
            ...(state.current.layout || {}),
            version: 1,
            columns: 12,
            order: currentWidgets().map((widget) => widget.id),
        };
    }

    function markDirty(value = true) {
        state.dirty = value;
        const badge = byId("panelDirtyState");
        if (!badge) return;
        badge.textContent = value ? "Unsaved" : "Clean";
        badge.className = `status-pill ${value ? "inactive" : "active"}`;
    }

    function collectionLabel(collection) {
        return {
            datasets: "Dataset",
            learning_models: "Model",
            studies: "Study",
            inferences: "Inference",
            code_models: "Code",
            plots: "Plot",
            runs: "Run",
        }[collection] || collection;
    }

    function itemId(item) {
        return item?.id ?? item?.run_id ?? item?.metadata?.display_id ?? item?.artifact?.relative_path ?? item?.title ?? "";
    }

    function itemTitle(item) {
        return item?.name || item?.title || item?.run_id || item?.id || "Untitled";
    }

    function sourceValue(collection, item) {
        return `${collection}:${itemId(item)}`;
    }

    function parseSourceValue(value) {
        const text = String(value || "");
        const separator = text.indexOf(":");
        if (separator < 0) return {};
        return {
            collection: text.slice(0, separator),
            id: text.slice(separator + 1),
        };
    }

    function findSource(source = {}) {
        const collection = source.collection;
        const id = String(source.id ?? "");
        if (!collection || !id) return null;
        const items = state.context[collection] || [];
        return items.find((item) => String(itemId(item)) === id) || null;
    }

    function updateSourceOptions() {
        const kind = byId("panelWidgetKind")?.value || "dataset";
        const sourceSelect = byId("panelWidgetSource");
        if (!sourceSelect) return;
        const collections = SOURCE_COLLECTIONS[kind] || [];
        const options = ['<option value="">No source</option>'];
        collections.forEach((collection) => {
            const items = state.context[collection] || [];
            items.forEach((item) => {
                const label = `${collectionLabel(collection)}: ${itemTitle(item)}`;
                options.push(`<option value="${escapeHtml(sourceValue(collection, item))}">${escapeHtml(label)}</option>`);
            });
        });
        sourceSelect.innerHTML = options.join("");
    }

    function renderContextSummary() {
        const target = byId("panelContextSummary");
        if (!target) return;
        const rows = [
            ["Datasets", state.context.datasets.length],
            ["Models", state.context.learning_models.length],
            ["Plots", state.context.plots.length],
            ["Runs", state.context.runs.length],
            ["Inferences", state.context.inferences.length],
            ["Studies", state.context.studies.length],
        ];
        target.innerHTML = `
            <div class="panel-context-metrics">
                ${rows.map(([label, value]) => `
                    <div class="panel-context-metric">
                        <span>${escapeHtml(label)}</span>
                        <strong>${escapeHtml(value)}</strong>
                    </div>
                `).join("")}
            </div>
        `;
    }

    function renderDashboardSelect() {
        const select = byId("panelDashboardSelect");
        if (!select) return;
        if (!state.dashboards.length) {
            select.innerHTML = '<option value="">Unsaved panel</option>';
            return;
        }
        select.innerHTML = state.dashboards.map((dashboard) => `
            <option value="${escapeHtml(dashboard.id)}">${escapeHtml(dashboard.name || "Amanaje Panel")}</option>
        `).join("");
        if (state.current?.id) select.value = String(state.current.id);
    }

    function renderDashboard() {
        const dashboard = state.current || newLocalDashboard();
        const nameInput = byId("panelDashboardName");
        const objectiveInput = byId("panelObjective");
        if (nameInput) nameInput.value = dashboard.name || "Amanaje Panel";
        if (objectiveInput) objectiveInput.value = dashboard.objective || "";
        const objectiveRun = byId("panelObjectiveRun");
        if (objectiveRun) objectiveRun.textContent = dashboard.objective || PANEL_OBJECTIVE;
        byId("panelCanvasTitle").textContent = dashboard.name || "Saved Objective Dashboard";
        byId("panelWidgetCount").textContent = `${currentWidgets().length} widgets`;
        renderWidgetDeck();
        renderDashboardSelect();
        applyPanelMode();
    }

    function renderWidgetDeck() {
        const deck = byId("panelWidgetDeck");
        const empty = byId("panelEmptyState");
        const widgets = currentWidgets();
        if (!deck || !empty) return;
        empty.hidden = widgets.length > 0;
        deck.innerHTML = widgets.map((widget, index) => renderWidget(widget, index)).join("");
    }

    function renderWidget(widget, index) {
        const size = normalizeWidgetSize(widget.size);
        widget.size = size;
        return `
            <article class="panel-widget ${escapeHtml(size)}" data-widget-id="${escapeHtml(widget.id)}">
                <header class="panel-widget-header">
                    <div class="panel-widget-toolbar">
                        <div>
                            <span class="panel-widget-kind">${escapeHtml((widget.kind || "metadata").replace("_", " "))}</span>
                            <h4 class="panel-widget-title">${escapeHtml(widget.title || "Widget")}</h4>
                        </div>
                        <div class="panel-widget-actions" data-panel-edit-only="true">
                            <button type="button" data-panel-action="move-up" ${index === 0 ? "disabled" : ""}>Up</button>
                            <button type="button" data-panel-action="move-down" ${index === currentWidgets().length - 1 ? "disabled" : ""}>Down</button>
                            <select data-panel-action="resize" aria-label="Widget size">
                                ${["wide", "full"].map((option) => `
                                    <option value="${option}" ${size === option ? "selected" : ""}>${option}</option>
                                `).join("")}
                            </select>
                            <button type="button" data-panel-action="remove">Remove</button>
                        </div>
                    </div>
                </header>
                <div class="panel-widget-body">
                    ${renderWidgetBody(widget)}
                </div>
            </article>
        `;
    }

    function sourceEmpty(kind) {
        return `<div class="panel-widget-empty">Select a ${escapeHtml(kind.replace("_", " "))} source, then save this panel.</div>`;
    }

    function renderFacts(facts) {
        return `
            <div class="panel-fact-grid">
                ${facts.filter(([, value]) => value !== undefined && value !== null && value !== "").map(([label, value]) => `
                    <div class="panel-fact">
                        <span>${escapeHtml(label)}</span>
                        <strong>${escapeHtml(Array.isArray(value) ? value.join(", ") : value)}</strong>
                    </div>
                `).join("")}
            </div>
        `;
    }

    function renderMetricGrid(metrics = {}) {
        const entries = Array.isArray(metrics)
            ? metrics.map((item) => [item.label || item.key || "Metric", item.value])
            : Object.entries(metrics || {});
        if (!entries.length) return '<div class="panel-widget-empty">No metrics available.</div>';
        return `
            <div class="panel-metric-grid">
                ${entries.slice(0, 12).map(([label, value]) => `
                    <div class="panel-metric">
                        <span>${escapeHtml(label)}</span>
                        <strong>${escapeHtml(typeof value === "number" ? Number(value).toPrecision(5) : value)}</strong>
                    </div>
                `).join("")}
            </div>
        `;
    }

    function renderJson(value, title = "JSON", options = {}) {
        const content = window.AmanajeUI?.renderCollapsibleJson
            ? window.AmanajeUI.renderCollapsibleJson(value, { title, collapsed: true })
            : `<pre class="json-view">${escapeHtml(JSON.stringify(value ?? {}, null, 2))}</pre>`;
        return options.editOnly ? `<div data-panel-edit-only="true">${content}</div>` : content;
    }

    function renderDataset(widget) {
        const item = findSource(widget.source);
        if (!item) return sourceEmpty("dataset");
        return `
            ${renderFacts([
                ["Name", item.name],
                ["Type", item.dataset_type || item.object_type],
                ["Shape", Array.isArray(item.shape) ? item.shape.join(" x ") : item.shape],
                ["Features", Array.isArray(item.features_list) ? item.features_list.length : ""],
            ])}
            <div class="panel-widget-actions" style="margin-top:0.75rem;">
                <button type="button" data-panel-action="load-analysis">Load Analysis</button>
            </div>
            ${widget.cache?.analysis ? renderJson(widget.cache.analysis, "Dataset Analysis") : ""}
        `;
    }

    function renderLearningModel(widget) {
        const item = findSource(widget.source);
        if (!item) return sourceEmpty("learning model");
        return `
            ${renderFacts([
                ["Name", item.name],
                ["Type", item.model_type || item.object_type],
                ["Trained", item.is_trained],
                ["Tested", item.is_tested],
                ["Deployed", item.is_deployed],
            ])}
            <div style="margin-top:0.75rem;">${renderMetricGrid(item.metrics || {})}</div>
            <div class="panel-widget-actions" style="margin-top:0.75rem;">
                <button type="button" data-panel-action="load-analysis">Load Analysis</button>
            </div>
            ${widget.cache?.analysis ? renderJson(widget.cache.analysis, "Model Analysis") : ""}
        `;
    }

    function renderSeries(plot = {}) {
        const series = Array.isArray(plot.series) ? plot.series : [];
        if (!series.length) return renderJson(plot.figure || plot, "Plot Data");
        const values = series.map((item) => Math.abs(Number(item.value || 0)));
        const maxValue = Math.max(...values, 1);
        return `
            <div class="panel-series-bars">
                ${series.slice(0, 14).map((item) => {
                    const value = Number(item.value || 0);
                    const width = Math.max(4, Math.min(100, Math.abs(value) / maxValue * 100));
                    return `
                        <div class="panel-series-row">
                            <div class="panel-series-meta">
                                <span>${escapeHtml(item.label)}</span>
                                <strong>${escapeHtml(value.toPrecision(5))}</strong>
                            </div>
                            <div class="panel-series-track"><div class="panel-series-fill" style="width:${width}%"></div></div>
                        </div>
                    `;
                }).join("")}
            </div>
        `;
    }

    function renderTablePlot(plot = {}) {
        const rows = Array.isArray(plot.rows) ? plot.rows : [];
        const columns = Array.isArray(plot.columns) && plot.columns.length
            ? plot.columns
            : Object.keys(rows[0] || {});
        if (!columns.length) return renderJson(plot, "Table Plot");
        return `
            <div class="table-wrapper">
                <table>
                    <thead><tr>${columns.map((column) => `<th>${escapeHtml(column)}</th>`).join("")}</tr></thead>
                    <tbody>
                        ${rows.slice(0, 12).map((row) => `
                            <tr>${columns.map((column) => `<td>${escapeHtml(row?.[column])}</td>`).join("")}</tr>
                        `).join("")}
                    </tbody>
                </table>
            </div>
        `;
    }

    function renderPlot(widget) {
        const plot = findSource(widget.source);
        if (!plot) return sourceEmpty("plot");
        if (plot.kind === "legacy_image" && plot.artifact?.url) {
            return `<img class="panel-artifact-image" src="${escapeHtml(plot.artifact.url)}" alt="${escapeHtml(plot.title || "Plot artifact")}">`;
        }
        if (plot.kind === "table") return renderTablePlot(plot);
        if (plot.figure) {
            const src = `${dashboardUrl()}/embed?figure=${encodeURIComponent(encodeFigure(plot.figure))}&title=${encodeURIComponent(plot.title || "Plot")}`;
            return `
                <div class="panel-plot-shell">
                    <iframe class="panel-plot-frame" src="${escapeHtml(src)}" loading="lazy" title="${escapeHtml(plot.title || "Plot")}"></iframe>
                </div>
                ${renderJson(plotIdentity(plot), "Plot Metadata", { editOnly: true })}
            `;
        }
        return renderSeries(plot);
    }

    function renderDashWorkspace(widget) {
        const path = dashPath(widget.settings?.path || "/");
        const src = dashUrlForPath(path);
        return `
            <div class="panel-dash-workspace">
                ${widget.settings?.description ? `<p class="panel-dash-caption">${escapeHtml(widget.settings.description)}</p>` : ""}
                <iframe class="panel-dash-frame" src="${escapeHtml(src)}" loading="lazy" title="${escapeHtml(widget.title || "Dash workspace")}"></iframe>
                <div class="panel-widget-actions panel-dash-actions">
                    <a href="${escapeHtml(src)}" target="_blank" rel="noopener">Open Dash</a>
                    <a href="${escapeHtml(dashUrlForPath("/extensions"))}" target="_blank" rel="noopener">Extensions</a>
                </div>
            </div>
        `;
    }

    function plotIdentity(plot = {}) {
        return {
            title: plot.title,
            kind: plot.kind,
            plot_type: plot.plot_type || plot.metadata?.plot_type,
            job_id: plot.source?.job_id || plot.metadata?.job_id,
            artifact: plot.artifact?.relative_path || plot.artifact?.filename,
        };
    }

    function renderStudy(widget) {
        const item = findSource(widget.source);
        if (!item) return sourceEmpty("study");
        return `
            ${renderFacts([
                ["Name", item.name],
                ["Sampler", item.sampler],
                ["Objective", item.objective],
                ["Model", item.learning_model_id],
                ["Dataset", item.dataset_id],
            ])}
            ${renderJson({ best_trial: item.best_trial, best_params: item.best_params }, "Study Result")}
        `;
    }

    function renderInference(widget) {
        const item = findSource(widget.source);
        if (!item) return sourceEmpty("inference");
        return `
            ${renderFacts([
                ["Name", item.name],
                ["Model", item.learning_model_id],
                ["Dataset", item.dataset_id],
                ["Inputs", Array.isArray(item.input_features) ? item.input_features.length : ""],
                ["Outputs", Array.isArray(item.output_features) ? item.output_features.length : ""],
            ])}
            ${renderJson(item.inference_params || {}, "Inference Params")}
        `;
    }

    function renderSimulation(widget) {
        const item = findSource(widget.source);
        if (!item) return sourceEmpty(widget.kind || "simulation");
        const result = widget.cache?.simulation;
        return `
            ${renderFacts([
                ["Inference", item.name],
                ["Model", item.learning_model_id],
                ["Dataset", item.dataset_id],
            ])}
            <div class="panel-widget-actions" style="margin-top:0.75rem;">
                <button type="button" data-panel-action="run-simulation">Run Simulation</button>
            </div>
            ${result ? renderJson(result, "Simulation Result") : ""}
        `;
    }

    function renderMetric(widget) {
        const item = findSource(widget.source);
        const configured = widget.settings?.metrics;
        if (Array.isArray(configured) && configured.length) return renderMetricGrid(configured);
        if (item?.metrics) return renderMetricGrid(item.metrics);
        if (item?.result?.metrics) return renderMetricGrid(item.result.metrics);
        return '<div class="panel-widget-empty">Add metrics in Configuration JSON or choose a model/run source.</div>';
    }

    function renderNote(widget) {
        const text = widget.settings?.text || "";
        if (!isEditMode()) {
            return text
                ? `<div class="panel-run-note">${escapeHtml(text)}</div>`
                : '<div class="panel-widget-empty">No interpretation notes yet.</div>';
        }
        return `
            <textarea data-panel-note="${escapeHtml(widget.id)}" placeholder="Write interpretation notes for this objective.">${escapeHtml(text)}</textarea>
        `;
    }

    function renderCustomJson(widget) {
        const value = widget.settings?.json ?? widget.settings ?? {};
        if (!isEditMode()) return renderJson(value, "Custom JSON");
        return `
            <textarea data-panel-json="${escapeHtml(widget.id)}" placeholder='{"key": "value"}'>${escapeHtml(JSON.stringify(value, null, 2))}</textarea>
            ${renderJson(value, "Custom JSON Preview", { editOnly: true })}
        `;
    }

    function renderMetadata(widget) {
        if (widget.settings?.mode === "objective") {
            return renderFacts([
                ["Objective", state.current?.objective || PANEL_OBJECTIVE],
                ["Widgets", currentWidgets().length],
                ["Saved Panel", state.current?.id || "Unsaved"],
            ]);
        }
        const item = findSource(widget.source);
        return item ? renderJson(item, "Source Metadata") : renderJson(widget.settings || {}, "Metadata");
    }

    function renderWidgetBody(widget) {
        if (widget.kind === "dataset") return renderDataset(widget);
        if (widget.kind === "dash_workspace") return renderDashWorkspace(widget);
        if (widget.kind === "learning_model") return renderLearningModel(widget);
        if (widget.kind === "plot") return renderPlot(widget);
        if (widget.kind === "study") return renderStudy(widget);
        if (widget.kind === "inference") return renderInference(widget);
        if (widget.kind === "simulation" || widget.kind === "prediction") return renderSimulation(widget);
        if (widget.kind === "metric") return renderMetric(widget);
        if (widget.kind === "note") return renderNote(widget);
        if (widget.kind === "custom_json") return renderCustomJson(widget);
        return renderMetadata(widget);
    }

    async function loadContext() {
        const payload = await fetchJson("/panel/context");
        state.context = { ...emptyContext(), ...payload };
        renderContextSummary();
        updateSourceOptions();
    }

    async function loadDashboards() {
        const payload = await fetchJson("/panel/dashboards");
        state.dashboards = Array.isArray(payload.dashboards) ? payload.dashboards : [];
        renderDashboardSelect();
    }

    async function loadDashboard(id) {
        if (!id) {
            state.current = normalizeDashboard(newLocalDashboard());
            setPanelMode(readPreferredPanelMode("edit"), { persist: false });
            markDirty(true);
            renderDashboard();
            return;
        }
        const payload = await fetchJson(`/panel/dashboards/${encodeURIComponent(id)}`);
        state.current = normalizeDashboard(payload.dashboard || newLocalDashboard());
        setPanelMode(readPreferredPanelMode("run"), { persist: false });
        markDirty(false);
        renderDashboard();
    }

    function payloadForSave() {
        syncCurrentFromForm();
        return {
            name: state.current.name,
            description: state.current.description || "Saved Painel Amanaje objective dashboard.",
            objective: state.current.objective,
            tint: "amanaje",
            layout: state.current.layout || {},
            widgets: currentWidgets().map((widget) => ({
                ...widget,
                size: normalizeWidgetSize(widget.size),
            })),
            panel_metadata: state.current.panel_metadata || {},
        };
    }

    async function saveDashboard() {
        if (!state.current) state.current = newLocalDashboard();
        const payload = payloadForSave();
        const hasId = Boolean(state.current.id);
        const url = hasId ? `/panel/dashboards/${encodeURIComponent(state.current.id)}` : "/panel/dashboards";
        const method = hasId ? "PUT" : "POST";
        const response = await fetchJson(url, { method, body: JSON.stringify(payload) });
        state.current = normalizeDashboard(response.dashboard);
        await loadDashboards();
        markDirty(false);
        renderDashboard();
        showStatus("Panel saved.", "success");
    }

    async function deleteDashboard() {
        if (!state.current?.id) {
            await loadDashboard("");
            return;
        }
        await fetchJson(`/panel/dashboards/${encodeURIComponent(state.current.id)}`, { method: "DELETE" });
        await loadDashboards();
        if (state.dashboards.length) {
            await loadDashboard(state.dashboards[0].id);
        } else {
            await loadDashboard("");
        }
        showStatus("Panel deleted.", "info");
    }

    function parseWidgetConfig() {
        const raw = byId("panelWidgetConfig")?.value?.trim();
        if (!raw) return {};
        try {
            const parsed = JSON.parse(raw);
            return parsed && typeof parsed === "object" ? parsed : {};
        } catch (error) {
            showStatus(`Configuration JSON is invalid: ${error.message}`, "error");
            throw error;
        }
    }

    function addWidget() {
        if (!state.current) state.current = newLocalDashboard();
        const kind = byId("panelWidgetKind")?.value || "metadata";
        const source = parseSourceValue(byId("panelWidgetSource")?.value || "");
        const sourceItem = findSource(source);
        const fallbackTitle = sourceItem ? itemTitle(sourceItem) : kind.replace("_", " ");
        const title = byId("panelWidgetTitle")?.value?.trim() || fallbackTitle;
        const size = normalizeWidgetSize(byId("panelWidgetSize")?.value || "wide");
        const settings = parseWidgetConfig();
        if (kind === "note" && settings.text === undefined) settings.text = "";
        if (kind === "dash_workspace" && settings.path === undefined) settings.path = "/";
        const widget = {
            id: `widget_${Date.now().toString(36)}_${Math.random().toString(16).slice(2, 8)}`,
            kind,
            title,
            size,
            source,
            settings,
            cache: {},
        };
        currentWidgets().push(widget);
        byId("panelWidgetTitle").value = "";
        markDirty(true);
        renderDashboard();
    }

    function findWidget(widgetId) {
        return currentWidgets().find((widget) => widget.id === widgetId);
    }

    function widgetIndex(widgetId) {
        return currentWidgets().findIndex((widget) => widget.id === widgetId);
    }

    function moveWidget(widgetId, direction) {
        const widgets = currentWidgets();
        const index = widgetIndex(widgetId);
        const target = index + direction;
        if (index < 0 || target < 0 || target >= widgets.length) return;
        const [widget] = widgets.splice(index, 1);
        widgets.splice(target, 0, widget);
        markDirty(true);
        renderDashboard();
    }

    function removeWidget(widgetId) {
        const widgets = currentWidgets();
        const index = widgetIndex(widgetId);
        if (index < 0) return;
        widgets.splice(index, 1);
        markDirty(true);
        renderDashboard();
    }

    async function loadAnalysis(widget) {
        const source = findSource(widget.source);
        if (!source) return;
        let url = "";
        if (widget.kind === "dataset") url = `/analysis/data?dataset_id=${encodeURIComponent(source.id)}`;
        if (widget.kind === "learning_model") url = `/analysis/model?model_id=${encodeURIComponent(source.id)}`;
        if (!url) return;
        widget.cache = { ...(widget.cache || {}), analysis: await fetchJson(url) };
        markDirty(true);
        renderDashboard();
    }

    async function runSimulation(widget) {
        const source = findSource(widget.source);
        if (!source) return;
        const modelId = source.learning_model_id || widget.source?.model_id;
        const datasetId = source.dataset_id || widget.source?.dataset_id;
        if (!modelId || !datasetId) {
            showStatus("Simulation widgets need an inference source with a model and dataset.", "warning");
            return;
        }
        const result = await fetchJson("/production/simulate", {
            method: "POST",
            body: JSON.stringify({ modelId, datasetId }),
        });
        widget.cache = { ...(widget.cache || {}), simulation: result };
        markDirty(true);
        renderDashboard();
    }

    function handleDeckClick(event) {
        const action = event.target?.getAttribute?.("data-panel-action");
        if (!action) return;
        const wrapper = event.target.closest("[data-widget-id]");
        const widgetId = wrapper?.getAttribute("data-widget-id");
        const widget = findWidget(widgetId);
        if (!widget) return;
        if (action === "move-up") moveWidget(widgetId, -1);
        if (action === "move-down") moveWidget(widgetId, 1);
        if (action === "remove") removeWidget(widgetId);
        if (action === "load-analysis") loadAnalysis(widget).catch((error) => showStatus(error.message, "error"));
        if (action === "run-simulation") runSimulation(widget).catch((error) => showStatus(error.message, "error"));
    }

    function handleDeckChange(event) {
        const action = event.target?.getAttribute?.("data-panel-action");
        if (action !== "resize") return;
        const widgetId = event.target.closest("[data-widget-id]")?.getAttribute("data-widget-id");
        const widget = findWidget(widgetId);
        if (!widget) return;
        widget.size = normalizeWidgetSize(event.target.value || "wide");
        markDirty(true);
        renderDashboard();
    }

    function handleDeckInput(event) {
        const noteId = event.target?.getAttribute?.("data-panel-note");
        const jsonId = event.target?.getAttribute?.("data-panel-json");
        if (noteId) {
            const widget = findWidget(noteId);
            if (!widget) return;
            widget.settings = { ...(widget.settings || {}), text: event.target.value };
            markDirty(true);
        }
        if (jsonId) {
            const widget = findWidget(jsonId);
            if (!widget) return;
            try {
                widget.settings = { json: JSON.parse(event.target.value || "{}") };
                markDirty(true);
            } catch (_error) {
                widget.settings = { raw: event.target.value };
                markDirty(true);
            }
        }
    }

    function bindEvents() {
        byId("panelDashboardSelect")?.addEventListener("change", (event) => {
            loadDashboard(event.target.value).catch((error) => showStatus(error.message, "error"));
        });
        byId("panelNewDashboard")?.addEventListener("click", () => {
            state.current = normalizeDashboard(newLocalDashboard());
            setPanelMode("edit");
            markDirty(true);
            renderDashboard();
        });
        byId("panelSaveDashboard")?.addEventListener("click", () => {
            saveDashboard().catch((error) => showStatus(error.message, "error"));
        });
        byId("panelDeleteDashboard")?.addEventListener("click", () => {
            deleteDashboard().catch((error) => showStatus(error.message, "error"));
        });
        byId("panelRefreshContext")?.addEventListener("click", () => {
            loadContext().then(() => showStatus("Panel artifacts refreshed.", "success")).catch((error) => showStatus(error.message, "error"));
        });
        byId("panelAddWidget")?.addEventListener("click", () => {
            try {
                addWidget();
            } catch (_error) {
                return;
            }
        });
        document.querySelectorAll("[data-panel-mode-button]").forEach((button) => {
            button.addEventListener("click", () => setPanelMode(button.getAttribute("data-panel-mode-button")));
        });
        byId("panelWidgetKind")?.addEventListener("change", updateSourceOptions);
        byId("panelDashboardName")?.addEventListener("input", () => {
            syncCurrentFromForm();
            byId("panelCanvasTitle").textContent = state.current?.name || "Saved Objective Dashboard";
            markDirty(true);
        });
        byId("panelObjective")?.addEventListener("input", () => {
            syncCurrentFromForm();
            const objectiveRun = byId("panelObjectiveRun");
            if (objectiveRun) objectiveRun.textContent = state.current?.objective || PANEL_OBJECTIVE;
            markDirty(true);
            renderWidgetDeck();
        });
        byId("panelWidgetDeck")?.addEventListener("click", handleDeckClick);
        byId("panelWidgetDeck")?.addEventListener("change", handleDeckChange);
        byId("panelWidgetDeck")?.addEventListener("input", handleDeckInput);
    }

    async function boot() {
        if (!byId("panelPage")) return;
        bindEvents();
        applyPanelMode();
        showStatus("Loading panel workspace...", "info");
        await loadContext();
        await loadDashboards();
        if (state.dashboards.length) {
            await loadDashboard(state.dashboards[0].id);
        } else {
            await loadDashboard("");
        }
        showStatus("", "info");
    }

    document.addEventListener("DOMContentLoaded", () => {
        boot().catch((error) => {
            state.current = normalizeDashboard(newLocalDashboard());
            setPanelMode(readPreferredPanelMode("edit"), { persist: false });
            renderDashboard();
            markDirty(true);
            showStatus(error.message || "Unable to load panel workspace.", "error");
        });
    });
})();
