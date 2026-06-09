(function () {
    const PANEL_OBJECTIVE = "Understand the active operation through connected data, models, plots, simulations, metrics, evidence, and notes.";
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
        filter_control: [],
        comparison: ["plots", "datasets", "learning_models", "runs", "inferences", "studies"],
        metric: ["learning_models", "runs"],
        metadata: ["datasets", "learning_models", "studies", "inferences", "code_models", "runs"],
        note: [],
        custom_json: [],
    };
    const RESOURCE_COLLECTIONS = ["inferences", "datasets", "learning_models", "plots", "studies", "runs", "code_models"];
    const FOCUS_COLLECTIONS = new Set(["inferences", "datasets", "learning_models"]);
    const KIND_FOR_COLLECTION = {
        code_models: "metadata",
        datasets: "dataset",
        inferences: "inference",
        learning_models: "learning_model",
        plots: "plot",
        runs: "metric",
        studies: "study",
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
        normalized.panel_metadata = normalized.panel_metadata && typeof normalized.panel_metadata === "object"
            ? normalized.panel_metadata
            : {};
        return normalized;
    }

    function defaultWidgets() {
        return [
            {
                id: "objective_focus",
                kind: "metadata",
                title: "Operation Objective",
                size: "full",
                source: {},
                settings: { mode: "objective" },
                cache: {},
            },
            {
                id: "dash_visualization_studio",
                kind: "dash_workspace",
                title: "Plotly Dash Workspace",
                size: "full",
                source: {},
                settings: {
                    path: "/",
                    description: "Interactive Plotly Dash workspace for visual tiles, Atlas extensions, and drilldown views.",
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
                id: "prediction_surface",
                kind: "prediction",
                title: "Prediction Surface",
                size: "wide",
                source: {},
                settings: { variant: "simulation_summary" },
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
            name: "Atlas Prediction Review",
            description: "Saved Operational Atlas panel workspace.",
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
        state.current.name = byId("panelDashboardName")?.value?.trim() || "Atlas Prediction Review";
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
        state.current.panel_metadata = state.current.panel_metadata && typeof state.current.panel_metadata === "object"
            ? state.current.panel_metadata
            : {};
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

    function panelMetadata() {
        if (!state.current) state.current = newLocalDashboard();
        if (!state.current.panel_metadata || typeof state.current.panel_metadata !== "object") {
            state.current.panel_metadata = {};
        }
        return state.current.panel_metadata;
    }

    function optionMarkupForCollections(collections = [], emptyLabel = "No source") {
        const options = ['<option value="">No source</option>'];
        collections.forEach((collection) => {
            const items = state.context[collection] || [];
            items.forEach((item) => {
                const label = `${collectionLabel(collection)}: ${itemTitle(item)}`;
                options.push(`<option value="${escapeHtml(sourceValue(collection, item))}">${escapeHtml(label)}</option>`);
            });
        });
        options[0] = `<option value="">${escapeHtml(emptyLabel)}</option>`;
        return options.join("");
    }

    function updateCenterOptions() {
        const kind = byId("panelCenterKind")?.value || "inferences";
        const target = byId("panelCenterSource");
        if (!target) return;
        target.innerHTML = optionMarkupForCollections([kind], "No focus source");
        const center = panelMetadata().center || {};
        if (center.collection === kind && center.id) target.value = `${center.collection}:${center.id}`;
    }

    function updateSourceOptions() {
        const kind = byId("panelWidgetKind")?.value || "dash_workspace";
        const sourceSelect = byId("panelWidgetSource");
        const comparisonFields = byId("panelComparisonFields");
        if (!sourceSelect) return;
        const collections = SOURCE_COLLECTIONS[kind] || [];
        sourceSelect.innerHTML = optionMarkupForCollections(collections, "No source");
        if (comparisonFields) comparisonFields.hidden = kind !== "comparison";
        const comparisonMarkup = optionMarkupForCollections(["plots", "datasets", "learning_models", "runs", "inferences", "studies"], "Choose source");
        ["panelComparisonSourceA", "panelComparisonSourceB"].forEach((id) => {
            const select = byId(id);
            if (select) select.innerHTML = comparisonMarkup;
        });
    }

    function renderPanelFocus() {
        const focus = byId("panelFocusSummary");
        const center = panelMetadata().center || {};
        const item = findSource(center);
        if (focus) {
            focus.innerHTML = item
                ? `<span>${escapeHtml(collectionLabel(center.collection))}</span><strong>${escapeHtml(itemTitle(item))}</strong>`
                : "No focus object selected.";
        }
        const filters = panelMetadata().filters || {};
        const filterSummary = byId("panelFilterSummary");
        if (filterSummary) {
            filterSummary.textContent = filters.column
                ? `${filters.column}: ${filters.start || "start"} to ${filters.end || "end"}`
                : "No Atlas filter applied.";
        }
        if (byId("panelFilterColumn")) byId("panelFilterColumn").value = filters.column || "";
        if (byId("panelFilterStart")) byId("panelFilterStart").value = filters.start || "";
        if (byId("panelFilterEnd")) byId("panelFilterEnd").value = filters.end || "";
        updateCenterOptions();
        updatePanelRangeControls();
    }

    function applyPanelRowFilters(rows = []) {
        const filters = panelMetadata().filters || {};
        const column = String(filters.column || "").trim();
        if (!column || !Array.isArray(rows) || !rows.length) return rows;
        const start = filters.start;
        const end = filters.end;
        return rows.filter((row, index) => {
            const value = row && Object.prototype.hasOwnProperty.call(row, column) ? row[column] : index;
            const numericValue = Number(value);
            const numericStart = start === "" || start === undefined ? null : Number(start);
            const numericEnd = end === "" || end === undefined ? null : Number(end);
            if (Number.isFinite(numericValue) && (Number.isFinite(numericStart) || Number.isFinite(numericEnd))) {
                if (Number.isFinite(numericStart) && numericValue < numericStart) return false;
                if (Number.isFinite(numericEnd) && numericValue > numericEnd) return false;
                return true;
            }
            const timeValue = Date.parse(value);
            const timeStart = start ? Date.parse(start) : NaN;
            const timeEnd = end ? Date.parse(end) : NaN;
            if (Number.isFinite(timeValue) && (Number.isFinite(timeStart) || Number.isFinite(timeEnd))) {
                if (Number.isFinite(timeStart) && timeValue < timeStart) return false;
                if (Number.isFinite(timeEnd) && timeValue > timeEnd) return false;
            }
            return true;
        });
    }

    function normalizeRows(value) {
        if (!Array.isArray(value)) return [];
        return value
            .map((row) => (row && typeof row === "object" && !Array.isArray(row)) ? row : { value: row })
            .filter(Boolean);
    }

    function collectRowsForRangeControls() {
        const rows = [];
        const pushRows = (value) => {
            rows.push(...normalizeRows(value));
        };
        currentWidgets().forEach((widget) => {
            pushRows(widget.cache?.analysis?.preview_rows);
            pushRows(widget.cache?.analysis?.manifest?.preview_rows);
            pushRows(widget.cache?.analysis?.plots?.flatMap?.((plot) => plot.rows || []) || []);
            pushRows(widget.cache?.simulation?.table_rows);
            pushRows(widget.cache?.simulation?.series);
            pushRows(widget.cache?.simulation?.scenarios?.flatMap?.((scenario) => scenario.table_rows || scenario.series || []) || []);
            const customJson = widget.settings?.json ?? widget.settings?.rows;
            pushRows(customJson);
        });
        (state.context.plots || []).forEach((plot) => {
            pushRows(plot.rows);
        });
        (state.context.runs || []).forEach((run) => {
            pushRows(run.result?.table_rows);
            pushRows(run.result?.series);
        });
        return rows.slice(0, 5000);
    }

    function rangeValueForRow(row, column, index) {
        if (!column) return undefined;
        if (row && Object.prototype.hasOwnProperty.call(row, column)) return row[column];
        if (["index", "row", "row_index"].includes(String(column).toLowerCase())) return index;
        return undefined;
    }

    function collectFilterRangeValues(column) {
        const rows = collectRowsForRangeControls();
        const rawValues = rows
            .map((row, index) => rangeValueForRow(row, column, index))
            .filter((value) => value !== undefined && value !== null && value !== "");
        if (!rawValues.length) return null;
        const numericValues = rawValues.map((value) => Number(value)).filter((value) => Number.isFinite(value));
        if (numericValues.length >= Math.max(1, Math.ceil(rawValues.length * 0.7))) {
            return { type: "number", min: Math.min(...numericValues), max: Math.max(...numericValues) };
        }
        const dateValues = rawValues.map((value) => Date.parse(value)).filter((value) => Number.isFinite(value));
        if (dateValues.length >= Math.max(1, Math.ceil(rawValues.length * 0.7))) {
            return { type: "date", min: Math.min(...dateValues), max: Math.max(...dateValues) };
        }
        return null;
    }

    function formatRangeValue(range, value) {
        if (!range || !Number.isFinite(Number(value))) return "";
        if (range.type === "date") {
            return new Date(Number(value)).toISOString().slice(0, 10);
        }
        return Number(value).toLocaleString("en-US", { maximumFractionDigits: 6, useGrouping: false });
    }

    function parseRangeInput(range, value, fallback) {
        if (value === undefined || value === null || value === "") return fallback;
        if (range?.type === "date") {
            const parsed = Date.parse(value);
            return Number.isFinite(parsed) ? parsed : fallback;
        }
        const parsed = Number(value);
        return Number.isFinite(parsed) ? parsed : fallback;
    }

    function updatePanelRangeControls() {
        const column = byId("panelFilterColumn")?.value?.trim() || "";
        const controls = byId("panelFilterRangeControls");
        const startRange = byId("panelFilterRangeStart");
        const endRange = byId("panelFilterRangeEnd");
        if (!controls || !startRange || !endRange) return;
        const range = collectFilterRangeValues(column);
        if (!column || !range || range.min === range.max) {
            controls.hidden = true;
            return;
        }
        controls.hidden = false;
        controls.dataset.rangeType = range.type;
        const step = range.type === "date" ? 86400000 : Math.max((range.max - range.min) / 100, 0.000001);
        [startRange, endRange].forEach((input) => {
            input.min = String(range.min);
            input.max = String(range.max);
            input.step = String(step);
        });
        const startValue = parseRangeInput(range, byId("panelFilterStart")?.value, range.min);
        const endValue = parseRangeInput(range, byId("panelFilterEnd")?.value, range.max);
        startRange.value = String(Math.min(Math.max(startValue, range.min), range.max));
        endRange.value = String(Math.min(Math.max(endValue, range.min), range.max));
        const minLabel = byId("panelFilterRangeMin");
        const maxLabel = byId("panelFilterRangeMax");
        if (minLabel) minLabel.textContent = formatRangeValue(range, range.min);
        if (maxLabel) maxLabel.textContent = formatRangeValue(range, range.max);
    }

    function syncFilterInputsFromRange() {
        const controls = byId("panelFilterRangeControls");
        const startRange = byId("panelFilterRangeStart");
        const endRange = byId("panelFilterRangeEnd");
        if (!controls || controls.hidden || !startRange || !endRange) return;
        const range = {
            type: controls.dataset.rangeType || "number",
            min: Number(startRange.min),
            max: Number(startRange.max),
        };
        let startValue = Number(startRange.value);
        let endValue = Number(endRange.value);
        if (startValue > endValue) {
            [startValue, endValue] = [endValue, startValue];
            startRange.value = String(startValue);
            endRange.value = String(endValue);
        }
        if (byId("panelFilterStart")) byId("panelFilterStart").value = formatRangeValue(range, startValue);
        if (byId("panelFilterEnd")) byId("panelFilterEnd").value = formatRangeValue(range, endValue);
    }

    function resourceSearchText() {
        return String(byId("panelResourceSearch")?.value || "").trim().toLowerCase();
    }

    function resourceMatchesSearch(collection, item, search) {
        if (!search) return true;
        const haystack = [
            collectionLabel(collection),
            itemTitle(item),
            itemId(item),
            item?.object_type,
            item?.dataset_type,
            item?.model_type,
            item?.kind,
            item?.status,
        ].filter(Boolean).join(" ").toLowerCase();
        return haystack.includes(search);
    }

    function renderResourceInventory() {
        const search = resourceSearchText();
        const sections = RESOURCE_COLLECTIONS.map((collection) => {
            const items = (state.context[collection] || []).filter((item) => resourceMatchesSearch(collection, item, search));
            const visibleItems = items.slice(0, 14);
            return `
                <section class="panel-resource-group">
                    <header>
                        <span>${escapeHtml(collectionLabel(collection))}</span>
                        <strong>${escapeHtml(items.length)}</strong>
                    </header>
                    <div class="panel-resource-items">
                        ${visibleItems.length ? visibleItems.map((item) => {
                            const value = sourceValue(collection, item);
                            return `
                                <button type="button" class="panel-resource-item" data-panel-resource-value="${escapeHtml(value)}" data-panel-resource-collection="${escapeHtml(collection)}">
                                    <span>${escapeHtml(collectionLabel(collection))}</span>
                                    <strong>${escapeHtml(itemTitle(item))}</strong>
                                    <small>${escapeHtml(itemId(item) || item?.object_type || "artifact")}</small>
                                </button>
                            `;
                        }).join("") : '<div class="panel-resource-empty">No matching resources.</div>'}
                    </div>
                </section>
            `;
        }).join("");
        return `<div class="panel-resource-list" id="panelResourceList">${sections}</div>`;
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
            ${renderResourceInventory()}
        `;
    }

    function selectResourceFromRail(button) {
        const value = button?.getAttribute?.("data-panel-resource-value") || "";
        const collection = button?.getAttribute?.("data-panel-resource-collection") || "";
        if (!value || !collection) return;

        if (FOCUS_COLLECTIONS.has(collection)) {
            const centerKind = byId("panelCenterKind");
            if (centerKind) centerKind.value = collection;
            updateCenterOptions();
            const centerSource = byId("panelCenterSource");
            if (centerSource) centerSource.value = value;
            panelMetadata().center = parseSourceValue(value);
        }

        const targetKind = KIND_FOR_COLLECTION[collection];
        const widgetKind = byId("panelWidgetKind");
        if (targetKind && widgetKind) {
            widgetKind.value = targetKind;
            updateSourceOptions();
            const widgetSource = byId("panelWidgetSource");
            if (widgetSource && Array.from(widgetSource.options).some((option) => option.value === value)) {
                widgetSource.value = value;
            }
        }

        markDirty(true);
        renderDashboard();
    }

    function renderDashboardSelect() {
        const select = byId("panelDashboardSelect");
        if (!select) return;
        if (!state.dashboards.length) {
            select.innerHTML = '<option value="">Unsaved Atlas report</option>';
            return;
        }
        select.innerHTML = state.dashboards.map((dashboard) => `
            <option value="${escapeHtml(dashboard.id)}">${escapeHtml(dashboard.name || "Atlas Prediction Review")}</option>
        `).join("");
        if (state.current?.id) select.value = String(state.current.id);
    }

    function dashboardSummary(dashboard) {
        return {
            id: dashboard?.id,
            name: dashboard?.name || "Atlas Prediction Review",
            objective: dashboard?.objective || "",
            tint: dashboard?.tint || "amanaje",
            updated_at: dashboard?.date || dashboard?.updated_at || null,
            widget_count: Array.isArray(dashboard?.widgets) ? dashboard.widgets.length : dashboard?.widget_count,
        };
    }

    function upsertDashboardSummary(dashboard) {
        const summary = dashboardSummary(dashboard);
        if (!summary.id) return;
        state.dashboards = [
            summary,
            ...state.dashboards.filter((item) => String(item.id) !== String(summary.id)),
        ];
        renderDashboardSelect();
    }

    function renderDashboard() {
        const dashboard = state.current || newLocalDashboard();
        const nameInput = byId("panelDashboardName");
        const objectiveInput = byId("panelObjective");
        if (nameInput) nameInput.value = dashboard.name || "Atlas Prediction Review";
        if (objectiveInput) objectiveInput.value = dashboard.objective || "";
        const objectiveRun = byId("panelObjectiveRun");
        if (objectiveRun) objectiveRun.textContent = dashboard.objective || PANEL_OBJECTIVE;
        byId("panelCanvasTitle").textContent = dashboard.name || "Saved Atlas Report";
        byId("panelWidgetCount").textContent = `${currentWidgets().length} tiles`;
        renderPanelFocus();
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
        window.AmanajeUI?.activateTabs?.(deck);
        window.AmanajeUI?.enhanceCollapsibles?.(deck);
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
                            <h4 class="panel-widget-title">${escapeHtml(widget.title || "Atlas Tile")}</h4>
                        </div>
                        <div class="panel-widget-actions" data-panel-edit-only="true">
                            <button type="button" data-panel-action="move-up" ${index === 0 ? "disabled" : ""}>Up</button>
                            <button type="button" data-panel-action="move-down" ${index === currentWidgets().length - 1 ? "disabled" : ""}>Down</button>
                            <select data-panel-action="resize" aria-label="Tile size">
                                ${["wide", "full"].map((option) => `
                                    <option value="${option}" ${size === option ? "selected" : ""}>${option === "full" ? "Full" : "Wide"}</option>
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
        return `<div class="panel-widget-empty">Bind an Atlas ${escapeHtml(kind.replace("_", " "))} source, then save this report.</div>`;
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

    function rowsFromResult(value = {}) {
        if (Array.isArray(value)) return normalizeRows(value);
        if (!value || typeof value !== "object") return [];
        return normalizeRows(value.table_rows || value.preview_rows || value.rows || value.series || []);
    }

    function renderResultRows(rows = [], title = "Rows") {
        const normalizedRows = normalizeRows(rows);
        if (!normalizedRows.length) return "";
        const columns = Array.from(new Set(normalizedRows.flatMap((row) => Object.keys(row)))).slice(0, 10);
        return `
            <div class="panel-result-table">
                <h5>${escapeHtml(title)}</h5>
                ${renderTablePlot({ kind: "table", columns, rows: normalizedRows })}
            </div>
        `;
    }

    function renderPlotDeckValue(plots = []) {
        if (!Array.isArray(plots) || !plots.length) return "";
        return window.AmanajeUI?.renderPlotDeck
            ? window.AmanajeUI.renderPlotDeck(plots)
            : plots.map((plot) => {
                if (plot.kind === "table") return renderTablePlot(plot);
                if (plot.figure) return renderJson(plot.figure, plot.title || "Plot Figure");
                return renderSeries(plot);
            }).join("");
    }

    function renderPanelSimulationResult(result = {}) {
        const scenarios = Array.isArray(result.scenarios) ? result.scenarios : [];
        const summary = result.prediction_summary || result.primary_result || {};
        const tabs = [
            {
                key: "summary",
                label: "Summary",
                content: `
                    ${renderFacts([
                        ["Family", result.family || result.task_type],
                        ["Mode", result.mode],
                        ["Output", result.output_feature || summary.label],
                        ["Scenarios", scenarios.length || (result.series?.length ? 1 : 0)],
                    ])}
                    <div class="panel-result-gap">${renderMetricGrid({
                        final_prediction: summary.final_prediction ?? result.primary_result?.value,
                        prediction_delta: summary.prediction_delta ?? result.primary_result?.delta,
                        step_count: summary.step_count ?? result.steps,
                    })}</div>
                    ${result.readiness ? renderJson(result.readiness, "Readiness") : ""}
                `,
            },
            ...scenarios.map((scenario, index) => ({
                key: `scenario_${index}`,
                label: scenario.name || `Scenario ${index + 1}`,
                content: `
                    ${renderFacts([
                        ["Steps", scenario.steps || scenario.prediction_summary?.step_count],
                        ["Final", scenario.prediction_summary?.final_prediction],
                        ["Delta", scenario.prediction_summary?.prediction_delta],
                        ["Changed", (scenario.changed_features || []).length],
                    ])}
                    ${renderPlotDeckValue(scenario.plots || [])}
                    ${renderResultRows(scenario.table_rows || scenario.series || [], "Scenario Rows")}
                    ${renderJson(scenario, "Raw Scenario")}
                `,
            })),
            {
                key: "rows",
                label: "Rows",
                content: renderResultRows(rowsFromResult(result), "Simulation Rows") || '<div class="panel-widget-empty">No simulation rows available.</div>',
            },
            {
                key: "plots",
                label: "Plots",
                content: renderPlotDeckValue(result.plots || []) || '<div class="panel-widget-empty">No plot deck available.</div>',
            },
            { key: "raw", label: "Raw", content: renderJson(result, "Raw Result") },
        ];
        return window.AmanajeUI?.renderResultTabs
            ? window.AmanajeUI.renderResultTabs(tabs, { idPrefix: `panel_result_${Math.random().toString(16).slice(2)}` })
            : tabs.map((tab) => `<section><h5>${escapeHtml(tab.label)}</h5>${tab.content}</section>`).join("");
    }

    function renderResultValue(value, title = "Result") {
        if (value === null || value === undefined || value === "") return "";
        if (Array.isArray(value)) return renderResultRows(value, title) || renderJson(value, title);
        if (typeof value !== "object") return `<div class="panel-run-note">${escapeHtml(value)}</div>`;
        if (Array.isArray(value.scenarios) || Array.isArray(value.series) || Array.isArray(value.table_rows) || value.prediction_summary || value.primary_result) {
            return renderPanelSimulationResult(value);
        }
        const chunks = [];
        if (value.metrics || value.summary?.metrics) {
            chunks.push(renderMetricGrid(value.metrics || value.summary.metrics));
        }
        if (Array.isArray(value.plots) && value.plots.length) {
            chunks.push(renderPlotDeckValue(value.plots));
        }
        const rows = rowsFromResult(value);
        if (rows.length) {
            chunks.push(renderResultRows(rows, title));
        }
        if (chunks.length) {
            chunks.push(renderJson(value, `${title} Metadata`, { editOnly: true }));
            return chunks.join("");
        }
        return renderJson(value, title);
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
            <div class="panel-widget-actions panel-widget-action-row">
                <button type="button" data-panel-action="load-analysis">Load Evidence</button>
            </div>
            ${widget.cache?.analysis ? renderResultValue(widget.cache.analysis, "Dataset Analysis") : ""}
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
            <div class="panel-result-gap">${renderMetricGrid(item.metrics || {})}</div>
            <div class="panel-widget-actions panel-widget-action-row">
                <button type="button" data-panel-action="load-analysis">Load Evidence</button>
            </div>
            ${widget.cache?.analysis ? renderResultValue(widget.cache.analysis, "Model Analysis") : ""}
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

    function renderCellValue(value) {
        if (value && typeof value === "object") return JSON.stringify(value);
        return value;
    }

    function renderTablePlot(plot = {}) {
        const rows = applyPanelRowFilters(Array.isArray(plot.rows) ? plot.rows : []);
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
                            <tr>${columns.map((column) => `<td>${escapeHtml(renderCellValue(row?.[column]))}</td>`).join("")}</tr>
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

    function renderSourcePreview(source = {}) {
        const item = findSource(source);
        if (!item) return '<div class="panel-widget-empty">Choose an Atlas comparison source.</div>';
        if (source.collection === "plots") {
            if (item.kind === "legacy_image" && item.artifact?.url) {
                return `<img class="panel-artifact-image" src="${escapeHtml(item.artifact.url)}" alt="${escapeHtml(item.title || "Plot artifact")}">`;
            }
            if (item.kind === "table") return renderTablePlot(item);
            if (item.figure) {
                const src = `${dashboardUrl()}/embed?figure=${encodeURIComponent(encodeFigure(item.figure))}&title=${encodeURIComponent(item.title || "Plot")}`;
                return `<iframe class="panel-plot-frame preview" src="${escapeHtml(src)}" loading="lazy" title="${escapeHtml(item.title || "Plot")}"></iframe>`;
            }
            return renderSeries(item);
        }
        return renderFacts([
            ["Type", collectionLabel(source.collection)],
            ["Name", itemTitle(item)],
            ["ID", itemId(item)],
            ["Status", item.status || item.is_trained || item.dataset_type || ""],
        ]);
    }

    function renderComparison(widget) {
        const left = widget.settings?.left || widget.settings?.source_a || widget.source || {};
        const right = widget.settings?.right || widget.settings?.source_b || {};
        return `
            <div class="panel-comparison-grid">
                <section class="panel-comparison-pane">
                    <span>${escapeHtml(collectionLabel(left.collection || "source"))}</span>
                    <h5>${escapeHtml(itemTitle(findSource(left)) || "Compare A")}</h5>
                    ${renderSourcePreview(left)}
                </section>
                <section class="panel-comparison-pane">
                    <span>${escapeHtml(collectionLabel(right.collection || "source"))}</span>
                    <h5>${escapeHtml(itemTitle(findSource(right)) || "Compare B")}</h5>
                    ${renderSourcePreview(right)}
                </section>
            </div>
        `;
    }

    function renderDashWorkspace(widget) {
        const path = dashPath(widget.settings?.path || "/");
        const src = dashUrlForPath(path);
        return `
            <div class="panel-dash-workspace">
                ${widget.settings?.description ? `<p class="panel-dash-caption">${escapeHtml(widget.settings.description)}</p>` : ""}
                <iframe class="panel-dash-frame" src="${escapeHtml(src)}" loading="lazy" title="${escapeHtml(widget.title || "Dash workspace")}"></iframe>
                <div class="panel-widget-actions panel-dash-actions">
                    <a href="${escapeHtml(src)}" target="_blank" rel="noopener">Open Workspace</a>
                    <a href="${escapeHtml(dashUrlForPath("/extensions"))}" target="_blank" rel="noopener">Atlas Extensions</a>
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
            <div class="panel-widget-actions panel-widget-action-row">
                <button type="button" data-panel-action="run-simulation">Run Scenario</button>
            </div>
            ${result ? renderResultValue(result, "Simulation Result") : ""}
        `;
    }

    function renderFilterControl(widget) {
        const saved = widget.settings || {};
        const active = panelMetadata().filters || {};
        const column = String(saved.column || active.column || "").trim();
        const start = String(saved.start ?? active.start ?? "");
        const end = String(saved.end ?? active.end ?? "");
        const range = collectFilterRangeValues(column);
        const startValue = range ? parseRangeInput(range, start, range.min) : 0;
        const endValue = range ? parseRangeInput(range, end, range.max) : 100;
        return `
            <div class="panel-filter-widget">
                <div class="form-group">
                    <label>Column or Index</label>
                    <input type="text" value="${escapeHtml(column)}" placeholder="timestamp" data-panel-filter-widget-field="column">
                </div>
                <div class="panel-filter-grid">
                    <div class="form-group">
                        <label>Start</label>
                        <input type="text" value="${escapeHtml(start)}" placeholder="start" data-panel-filter-widget-field="start">
                    </div>
                    <div class="form-group">
                        <label>End</label>
                        <input type="text" value="${escapeHtml(end)}" placeholder="end" data-panel-filter-widget-field="end">
                    </div>
                </div>
                ${range && range.min !== range.max ? `
                    <div class="panel-range-controls">
                        <div class="panel-range-labels">
                            <span>${escapeHtml(formatRangeValue(range, range.min))}</span>
                            <strong>Actual range</strong>
                            <span>${escapeHtml(formatRangeValue(range, range.max))}</span>
                        </div>
                        <div class="panel-range-track">
                            <input type="range" min="${escapeHtml(range.min)}" max="${escapeHtml(range.max)}" step="${escapeHtml(range.type === "date" ? 86400000 : Math.max((range.max - range.min) / 100, 0.000001))}" value="${escapeHtml(startValue)}" data-panel-filter-widget-range="start" data-range-type="${escapeHtml(range.type)}">
                            <input type="range" min="${escapeHtml(range.min)}" max="${escapeHtml(range.max)}" step="${escapeHtml(range.type === "date" ? 86400000 : Math.max((range.max - range.min) / 100, 0.000001))}" value="${escapeHtml(endValue)}" data-panel-filter-widget-range="end" data-range-type="${escapeHtml(range.type)}">
                        </div>
                    </div>
                ` : '<div class="helper-text">A range slider appears after this column is found in saved widget results.</div>'}
                <div class="panel-widget-actions panel-widget-action-row">
                    <button type="button" data-panel-action="apply-widget-filter">Apply Atlas Filter</button>
                    <button type="button" data-panel-action="clear-widget-filter">Clear Filter</button>
                </div>
            </div>
        `;
    }

    function renderMetric(widget) {
        const item = findSource(widget.source);
        const configured = widget.settings?.metrics;
        if (Array.isArray(configured) && configured.length) return renderMetricGrid(configured);
        if (item?.metrics) return renderMetricGrid(item.metrics);
        if (item?.result?.metrics) return renderMetricGrid(item.result.metrics);
        return '<div class="panel-widget-empty">Add metrics in Advanced JSON or choose an Atlas model/run source.</div>';
    }

    function renderNote(widget) {
        const text = widget.settings?.text || "";
        if (!isEditMode()) {
            return text
                ? `<div class="panel-run-note">${escapeHtml(text)}</div>`
                : '<div class="panel-widget-empty">No Atlas interpretation notes yet.</div>';
        }
        return `
            <textarea data-panel-note="${escapeHtml(widget.id)}" placeholder="Write interpretation notes for this objective.">${escapeHtml(text)}</textarea>
        `;
    }

    function renderCustomJson(widget) {
        const value = widget.settings?.json ?? widget.settings ?? {};
        if (!isEditMode()) return renderResultValue(value, "Custom JSON");
        return `
            <textarea data-panel-json="${escapeHtml(widget.id)}" placeholder='{"key": "value"}'>${escapeHtml(JSON.stringify(value, null, 2))}</textarea>
            ${renderResultValue(value, "Custom JSON Preview")}
        `;
    }

    function renderMetadata(widget) {
        if (widget.settings?.mode === "objective") {
            return renderFacts([
                ["Objective", state.current?.objective || PANEL_OBJECTIVE],
                ["Tiles", currentWidgets().length],
                ["Atlas Report", state.current?.id || "Unsaved"],
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
        if (widget.kind === "comparison") return renderComparison(widget);
        if (widget.kind === "study") return renderStudy(widget);
        if (widget.kind === "inference") return renderInference(widget);
        if (widget.kind === "simulation" || widget.kind === "prediction") return renderSimulation(widget);
        if (widget.kind === "filter_control") return renderFilterControl(widget);
        if (widget.kind === "metric") return renderMetric(widget);
        if (widget.kind === "note") return renderNote(widget);
        if (widget.kind === "custom_json") return renderCustomJson(widget);
        return renderMetadata(widget);
    }

    async function loadContext() {
        const payload = await fetchJson("/panel/context");
        state.context = { ...emptyContext(), ...payload };
        renderContextSummary();
        updateCenterOptions();
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
            description: state.current.description || "Saved Operational Atlas panel workspace.",
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
        upsertDashboardSummary(state.current);
        markDirty(false);
        renderDashboard();
        showStatus("Atlas report saved.", "success");
        loadDashboards().catch((error) => showStatus(error.message, "error"));
    }

    async function deleteDashboard() {
        if (!state.current?.id) {
            await loadDashboard("");
            return;
        }
        const deletedId = state.current.id;
        await fetchJson(`/panel/dashboards/${encodeURIComponent(state.current.id)}`, { method: "DELETE" });
        state.dashboards = state.dashboards.filter((item) => String(item.id) !== String(deletedId));
        renderDashboardSelect();
        if (state.dashboards.length) {
            await loadDashboard(state.dashboards[0].id);
        } else {
            await loadDashboard("");
        }
        loadDashboards().catch((error) => showStatus(error.message, "error"));
        showStatus("Atlas report deleted.", "info");
    }

    function applyCenterObject() {
        const center = parseSourceValue(byId("panelCenterSource")?.value || "");
        panelMetadata().center = center;
        markDirty(true);
        renderDashboard();
    }

    function clearCenterObject() {
        delete panelMetadata().center;
        markDirty(true);
        renderDashboard();
    }

    function applyDisplayFilters() {
        const column = byId("panelFilterColumn")?.value?.trim() || "";
        const start = byId("panelFilterStart")?.value?.trim() || "";
        const end = byId("panelFilterEnd")?.value?.trim() || "";
        if (!column) {
            delete panelMetadata().filters;
        } else {
            panelMetadata().filters = { column, start, end };
        }
        markDirty(true);
        renderDashboard();
    }

    function clearDisplayFilters() {
        delete panelMetadata().filters;
        markDirty(true);
        renderDashboard();
    }

    function readFilterFieldsFromWidget(wrapper) {
        const field = (name) => wrapper?.querySelector(`[data-panel-filter-widget-field="${name}"]`)?.value?.trim() || "";
        return { column: field("column"), start: field("start"), end: field("end") };
    }

    function applyFilterWidget(widget, wrapper) {
        const filters = readFilterFieldsFromWidget(wrapper);
        widget.settings = { ...(widget.settings || {}), ...filters, range_mode: "actual" };
        if (!filters.column) {
            delete panelMetadata().filters;
        } else {
            panelMetadata().filters = filters;
        }
        markDirty(true);
        renderDashboard();
    }

    function clearFilterWidget(widget) {
        widget.settings = { ...(widget.settings || {}), column: "", start: "", end: "", range_mode: "actual" };
        delete panelMetadata().filters;
        markDirty(true);
        renderDashboard();
    }

    function syncWidgetFilterRange(input) {
        const wrapper = input.closest("[data-widget-id]");
        if (!wrapper) return;
        const startRange = wrapper.querySelector('[data-panel-filter-widget-range="start"]');
        const endRange = wrapper.querySelector('[data-panel-filter-widget-range="end"]');
        if (!startRange || !endRange) return;
        const range = {
            type: input.getAttribute("data-range-type") || "number",
            min: Number(startRange.min),
            max: Number(startRange.max),
        };
        let startValue = Number(startRange.value);
        let endValue = Number(endRange.value);
        if (startValue > endValue) {
            [startValue, endValue] = [endValue, startValue];
            startRange.value = String(startValue);
            endRange.value = String(endValue);
        }
        const startField = wrapper.querySelector('[data-panel-filter-widget-field="start"]');
        const endField = wrapper.querySelector('[data-panel-filter-widget-field="end"]');
        if (startField) startField.value = formatRangeValue(range, startValue);
        if (endField) endField.value = formatRangeValue(range, endValue);
    }

    function parseWidgetConfig() {
        const raw = byId("panelWidgetConfig")?.value?.trim();
        if (!raw) return {};
        try {
            const parsed = JSON.parse(raw);
            return parsed && typeof parsed === "object" ? parsed : {};
        } catch (error) {
            showStatus(`Advanced JSON is invalid: ${error.message}`, "error");
            throw error;
        }
    }

    function addWidget() {
        if (!state.current) state.current = newLocalDashboard();
        const kind = byId("panelWidgetKind")?.value || "dash_workspace";
        const source = parseSourceValue(byId("panelWidgetSource")?.value || "");
        const sourceItem = findSource(source);
        const fallbackTitle = sourceItem ? itemTitle(sourceItem) : kind.replace("_", " ");
        const title = byId("panelWidgetTitle")?.value?.trim() || fallbackTitle;
        const size = normalizeWidgetSize(byId("panelWidgetSize")?.value || "wide");
        const settings = parseWidgetConfig();
        if (kind === "note" && settings.text === undefined) settings.text = "";
        if (kind === "dash_workspace" && settings.path === undefined) settings.path = "/";
        if (kind === "filter_control") {
            Object.assign(settings, {
                range_mode: settings.range_mode || "actual",
                ...(panelMetadata().filters || {}),
            });
        }
        if (kind === "comparison") {
            settings.left = parseSourceValue(byId("panelComparisonSourceA")?.value || byId("panelWidgetSource")?.value || "");
            settings.right = parseSourceValue(byId("panelComparisonSourceB")?.value || "");
        }
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
            showStatus("Prediction tiles need an inference source with a model and dataset.", "warning");
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
        if (action === "apply-widget-filter") applyFilterWidget(widget, wrapper);
        if (action === "clear-widget-filter") clearFilterWidget(widget);
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
        if (event.target?.getAttribute?.("data-panel-filter-widget-range")) {
            syncWidgetFilterRange(event.target);
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
            loadContext().then(() => showStatus("Atlas artifacts refreshed.", "success")).catch((error) => showStatus(error.message, "error"));
        });
        byId("panelResourceSearch")?.addEventListener("input", renderContextSummary);
        byId("panelContextSummary")?.addEventListener("click", (event) => {
            const resourceButton = event.target?.closest?.("[data-panel-resource-value]");
            if (resourceButton) selectResourceFromRail(resourceButton);
        });
        byId("panelCenterKind")?.addEventListener("change", updateCenterOptions);
        byId("panelApplyCenter")?.addEventListener("click", applyCenterObject);
        byId("panelClearCenter")?.addEventListener("click", clearCenterObject);
        byId("panelApplyFilters")?.addEventListener("click", applyDisplayFilters);
        byId("panelClearFilters")?.addEventListener("click", clearDisplayFilters);
        byId("panelFilterColumn")?.addEventListener("input", updatePanelRangeControls);
        byId("panelFilterStart")?.addEventListener("input", updatePanelRangeControls);
        byId("panelFilterEnd")?.addEventListener("input", updatePanelRangeControls);
        byId("panelFilterRangeStart")?.addEventListener("input", syncFilterInputsFromRange);
        byId("panelFilterRangeEnd")?.addEventListener("input", syncFilterInputsFromRange);
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
            byId("panelCanvasTitle").textContent = state.current?.name || "Saved Atlas Report";
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
            showStatus(error.message || "Unable to load Operational Atlas workspace.", "error");
        });
    });
})();
