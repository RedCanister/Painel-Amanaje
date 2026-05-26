(function () {
    const LINE_LIMIT = 12;
    const CHAR_LIMIT = 500;
    let jsonExplorerCounter = 0;

    function escapeHtml(value) {
        return String(value ?? "")
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#39;");
    }

    function normalizeBadge(label, value) {
        if (value === undefined || value === null || value === "") {
            return "";
        }
        return `<span class="scale-badge">${escapeHtml(label)}: ${escapeHtml(value)}</span>`;
    }

    function renderMetricCards(cards = []) {
        const usableCards = Array.isArray(cards) ? cards : [];
        if (!usableCards.length) return "";
        return `
            <div class="metric-card-deck">
                ${usableCards.map((card) => `
                    <article class="metric-card-shared">
                        <h4>${escapeHtml(card.label || card.key || "Metric")}</h4>
                        <div class="metric-value">${escapeHtml(card.value)}</div>
                        <div class="scale-badges">
                            ${normalizeBadge("Unit", card.unit)}
                            ${normalizeBadge("Reference", card.reference)}
                            ${normalizeBadge("Range", card.expected_range)}
                            ${normalizeBadge("Direction", card.direction)}
                        </div>
                        ${card.explanation ? `<div class="helper-text">${escapeHtml(card.explanation)}</div>` : ""}
                    </article>
                `).join("")}
            </div>
        `;
    }

    function renderCollapsibleContent(contentHtml, options = {}) {
        const title = options.title || "Details";
        const collapsed = options.collapsed !== false;
        const containerClass = options.containerClass || "";
        return `
            <div class="amanaje-collapsible ${containerClass}" data-collapsible-wrapper="true">
                <div class="amanaje-collapse-header">
                    <span class="amanaje-collapse-title">${escapeHtml(title)}</span>
                    <button type="button" class="amanaje-collapse-toggle" data-collapsible-toggle="true" aria-expanded="${collapsed ? "false" : "true"}">
                        ${collapsed ? "Expand" : "Collapse"}
                    </button>
                </div>
                <div class="amanaje-collapse-body" ${collapsed ? "hidden" : ""}>
                    ${contentHtml}
                </div>
            </div>
        `;
    }

    function isPlainObject(value) {
        return Boolean(value) && typeof value === "object" && !Array.isArray(value);
    }

    function isScalar(value) {
        return value === null || ["string", "number", "boolean"].includes(typeof value);
    }

    function parseJsonLike(data) {
        if (typeof data !== "string") {
            return { value: data, jsonString: JSON.stringify(data ?? {}, null, 2) };
        }
        const trimmed = data.trim();
        if (looksLikeJson(trimmed)) {
            try {
                return { value: JSON.parse(trimmed), jsonString: data };
            } catch (_error) {
                return { value: data, jsonString: data };
            }
        }
        return { value: data, jsonString: data };
    }

    function columnName(column, index = 0) {
        if (isPlainObject(column)) {
            return String(column.key || column.name || column.id || column.label || column.title || `column_${index + 1}`);
        }
        return String(column || `column_${index + 1}`);
    }

    function compactJsonCell(value) {
        if (value === undefined || value === null) return "";
        if (isScalar(value)) return value;
        try {
            return JSON.stringify(value);
        } catch (_error) {
            return String(value);
        }
    }

    function unionColumns(rows = []) {
        const seen = new Set();
        const columns = [];
        rows.forEach((row) => {
            if (!isPlainObject(row)) return;
            Object.keys(row).forEach((key) => {
                if (seen.has(key)) return;
                seen.add(key);
                columns.push(key);
            });
        });
        return columns;
    }

    function normalizeRows(rows = [], columns = []) {
        return rows.map((row) => {
            if (Array.isArray(row)) {
                return columns.reduce((record, column, index) => {
                    record[column] = compactJsonCell(row[index]);
                    return record;
                }, {});
            }
            return columns.reduce((record, column) => {
                record[column] = compactJsonCell(isPlainObject(row) ? row[column] : undefined);
                return record;
            }, {});
        });
    }

    function tableResult(columns = [], rows = [], reason = "tabular") {
        return {
            columns,
            rows,
            shape: { rows: rows.length, columns: columns.length },
            tabular: Boolean(columns.length),
            reason,
        };
    }

    function normalizeJsonTable(data) {
        const parsed = parseJsonLike(data).value;
        if (isPlainObject(parsed) && Array.isArray(parsed.columns) && Array.isArray(parsed.rows)) {
            const columns = parsed.columns.map(columnName);
            return tableResult(columns, normalizeRows(parsed.rows, columns), "columns_rows");
        }
        if (Array.isArray(parsed) && parsed.length && parsed.every(isPlainObject)) {
            const columns = unionColumns(parsed);
            return tableResult(columns, normalizeRows(parsed, columns), "array_of_objects");
        }
        if (isPlainObject(parsed)) {
            const entries = Object.entries(parsed);
            if (entries.length && entries.every(([_key, value]) => isScalar(value))) {
                return tableResult(
                    ["key", "value"],
                    entries.map(([key, value]) => ({ key, value: compactJsonCell(value) })),
                    "flat_object"
                );
            }
        }
        return { columns: [], rows: [], shape: { rows: 0, columns: 0 }, tabular: false, reason: "not_tabular" };
    }

    function renderJsonTable(table = {}, options = {}) {
        if (!table.tabular) return '<div class="helper-text">This JSON payload is not table-shaped.</div>';
        const maxRows = Number(options.maxTableRows || 120);
        const rows = (table.rows || []).slice(0, maxRows);
        const columns = table.columns || [];
        return `
            <div class="json-table-meta helper-text">
                ${escapeHtml(table.shape.rows)} rows x ${escapeHtml(table.shape.columns)} columns
                ${table.shape.rows > rows.length ? ` - showing first ${escapeHtml(rows.length)}` : ""}
            </div>
            <div class="table-scroll json-table-scroll">
                <table class="full-width-table json-table-view">
                    <thead>
                        <tr>${columns.map((column) => `<th>${escapeHtml(column)}</th>`).join("")}</tr>
                    </thead>
                    <tbody>
                        ${rows.map((row) => `
                            <tr>
                                ${columns.map((column) => `<td>${escapeHtml(row?.[column])}</td>`).join("")}
                            </tr>
                        `).join("")}
                    </tbody>
                </table>
            </div>
        `;
    }

    function simpleHash(value) {
        const text = String(value || "");
        let hash = 0;
        for (let index = 0; index < text.length; index += 1) {
            hash = ((hash << 5) - hash) + text.charCodeAt(index);
            hash |= 0;
        }
        return Math.abs(hash).toString(16);
    }

    function renderPlotlyTableSpec(table = {}, title = "JSON Table") {
        const columns = table.columns || [];
        const rows = table.rows || [];
        return {
            id: `json_table_${simpleHash(`${title}:${table.shape?.rows || 0}:${columns.join("|")}`)}`,
            kind: "plotly_figure",
            title,
            description: "Plotly table generated from a JSON payload.",
            figure: {
                data: [
                    {
                        type: "table",
                        header: {
                            values: columns,
                            fill: { color: "#e0f2fe" },
                            align: "left",
                            font: { color: "#0c4a6e" },
                        },
                        cells: {
                            values: columns.map((column) => rows.map((row) => row?.[column] ?? "")),
                            fill: { color: "white" },
                            align: "left",
                        },
                    },
                ],
                layout: {
                    title: { text: title },
                    margin: { l: 10, r: 10, t: 48, b: 10 },
                    template: "plotly_white",
                },
                config: { displaylogo: false, responsive: true },
            },
            source: { domain: "json_explorer", registry_type: table.reason || "json" },
            metadata: {
                display_id: `json_table_${simpleHash(`${title}:${table.shape?.rows || 0}:${columns.join("|")}`)}`,
                kind: "plotly_figure",
                plot_type: "table",
                domain: "json_explorer",
                registry_type: table.reason || "json",
            },
        };
    }

    function renderPlotlyJsonTable(table = {}, options = {}) {
        const maxRows = Number(options.maxPlotlyRows || 80);
        const maxColumns = Number(options.maxPlotlyColumns || 24);
        if (!table.tabular) return '<div class="helper-text">This JSON payload is not table-shaped.</div>';
        if ((table.shape.rows || 0) > maxRows || (table.shape.columns || 0) > maxColumns) {
            return `
                <div class="helper-text json-plotly-fallback">
                    Plotly table preview is limited to ${escapeHtml(maxRows)} rows and ${escapeHtml(maxColumns)} columns.
                    Use the Table or Common JSON view for this larger payload.
                </div>
            `;
        }
        const spec = renderPlotlyTableSpec(table, options.title || "JSON Table");
        try {
            if (encodeForQuery(spec.figure).length > 12000) {
                return `
                    <div class="helper-text json-plotly-fallback">
                        Plotly table preview is too large for the embedded Visualization renderer URL.
                        Use the Table or Common JSON view for this payload.
                    </div>
                `;
            }
        } catch (_error) {
            return `
                <div class="helper-text json-plotly-fallback">
                    Plotly table preview could not be generated for this payload.
                    Use the Table or Common JSON view instead.
                </div>
            `;
        }
        return renderPlotlyFrame(spec);
    }

    function renderJsonExplorer(data, options = {}) {
        const title = options.title || "JSON";
        const parsed = parseJsonLike(data);
        const commonJson = `
            <pre class="${escapeHtml(options.preClass || "json-block")} json-common-view">${escapeHtml(parsed.jsonString)}</pre>
        `;
        const table = normalizeJsonTable(parsed.value);
        if (!table.tabular) {
            return `<div class="json-explorer" data-json-explorer="true">${commonJson}</div>`;
        }
        const preferredView = ["table", "common", "plotly"].includes(options.defaultView)
            ? options.defaultView
            : "table";
        jsonExplorerCounter += 1;
        const tabPrefix = options.idPrefix || `json_${simpleHash(parsed.jsonString).slice(0, 10)}_${jsonExplorerCounter}`;
        return `
            <div class="json-explorer" data-json-explorer="true">
                ${renderResultTabs(
                    [
                        { key: "table", label: "Table", content: renderJsonTable(table, options) },
                        { key: "common", label: "Common JSON", content: commonJson },
                        { key: "plotly", label: "Plotly Table", content: renderPlotlyJsonTable(table, { ...options, title }) },
                    ],
                    { idPrefix: tabPrefix, activeKey: preferredView }
                )}
            </div>
        `;
    }

    function renderCollapsibleJson(data, options = {}) {
        const title = options.title || "JSON";
        const explorer = options.explorer !== false;
        const content = explorer
            ? renderJsonExplorer(data, options)
            : `<pre class="${escapeHtml(options.preClass || "json-block")}">${escapeHtml(typeof data === "string" ? data : JSON.stringify(data ?? {}, null, 2))}</pre>`;
        return renderCollapsibleContent(
            content,
            { title, collapsed: options.collapsed !== false, containerClass: options.containerClass || "" }
        );
    }

    function cloneJson(value) {
        try {
            return JSON.parse(JSON.stringify(value ?? {}));
        } catch (_error) {
            return {};
        }
    }

    function cssVar(name, fallback = "") {
        if (typeof window === "undefined" || !window.getComputedStyle) return fallback;
        return window.getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;
    }

    function getVisualizationTheme() {
        const primary = cssVar("--amanaje-primary", "#0077b6");
        const primaryStrong = cssVar("--amanaje-primary-strong", "#0c4a6e");
        const primarySoft = cssVar("--amanaje-primary-soft", "#e0f2fe");
        const primaryAccent = cssVar("--amanaje-primary-accent", "#0ea5e9");
        const surface = cssVar("--amanaje-surface", "#ffffff");
        const surface2 = cssVar("--amanaje-surface-2", "#f8fafc");
        const text = cssVar("--amanaje-text-strong", "#111827");
        const muted = cssVar("--amanaje-muted", "#6b7280");
        const border = cssVar("--amanaje-border", "#dbe4ea");
        const font = cssVar("--amanaje-body-font", "Inter, system-ui, sans-serif");
        return {
            primary,
            primaryStrong,
            primarySoft,
            primaryAccent,
            surface,
            surface2,
            text,
            muted,
            border,
            font,
            colorway: [primary, primaryAccent, primaryStrong, "#14b8a6", "#f59e0b", "#ef4444", "#8b5cf6"],
        };
    }

    function isThemeDefaultColor(value) {
        if (typeof value !== "string") return false;
        return ["#2a9d8f", "#0077b6", "#0ea5e9", "#1d4ed8"].includes(value.trim().toLowerCase());
    }

    function applyAxisTheme(axis = {}, theme) {
        return {
            ...axis,
            gridcolor: axis.gridcolor || theme.border,
            linecolor: axis.linecolor || theme.border,
            zerolinecolor: axis.zerolinecolor || theme.border,
            tickfont: { color: theme.muted, ...(axis.tickfont || {}) },
            title: typeof axis.title === "object"
                ? { ...axis.title, font: { color: theme.text, ...(axis.title.font || {}) } }
                : axis.title,
        };
    }

    function applyTraceTheme(trace = {}, theme, index = 0) {
        const color = theme.colorway[index % theme.colorway.length];
        const nextTrace = { ...trace };
        const marker = isPlainObject(nextTrace.marker) ? { ...nextTrace.marker } : {};
        const line = isPlainObject(nextTrace.line) ? { ...nextTrace.line } : {};

        if (["bar", "histogram", "box", "violin"].includes(nextTrace.type || "") && (!marker.color || isThemeDefaultColor(marker.color))) {
            marker.color = color;
        }
        if ((nextTrace.type === "scatter" || !nextTrace.type) && (!line.color || isThemeDefaultColor(line.color))) {
            line.color = color;
        }
        if (nextTrace.type === "scatter" && (!marker.color || isThemeDefaultColor(marker.color))) {
            marker.color = color;
        }
        if (nextTrace.type === "pie" && !marker.colors) {
            marker.colors = theme.colorway;
        }
        if (nextTrace.type === "heatmap" && (!nextTrace.colorscale || nextTrace.colorscale === "Blues")) {
            nextTrace.colorscale = [[0, theme.surface2], [1, theme.primary]];
        }
        if (nextTrace.type === "table") {
            const header = isPlainObject(nextTrace.header) ? { ...nextTrace.header } : {};
            const cells = isPlainObject(nextTrace.cells) ? { ...nextTrace.cells } : {};
            const headerFill = isPlainObject(header.fill) ? { ...header.fill } : {};
            const cellFill = isPlainObject(cells.fill) ? { ...cells.fill } : {};
            const headerFont = isPlainObject(header.font) ? { ...header.font } : {};
            const cellFont = isPlainObject(cells.font) ? { ...cells.font } : {};
            header.fill = { ...headerFill, color: headerFill.color || theme.primarySoft };
            header.font = { ...headerFont, color: headerFont.color || theme.primaryStrong };
            cells.fill = { ...cellFill, color: cellFill.color || theme.surface };
            cells.font = { ...cellFont, color: cellFont.color || theme.text };
            nextTrace.header = header;
            nextTrace.cells = cells;
        } else {
            if (Object.keys(marker).length) nextTrace.marker = marker;
            if (Object.keys(line).length) nextTrace.line = line;
        }
        return nextTrace;
    }

    function applyPlotTheme(figure = {}) {
        const theme = getVisualizationTheme();
        const themedFigure = cloneJson(figure);
        const layout = isPlainObject(themedFigure.layout) ? themedFigure.layout : {};
        themedFigure.layout = {
            ...layout,
            paper_bgcolor: layout.paper_bgcolor || theme.surface,
            plot_bgcolor: layout.plot_bgcolor || theme.surface,
            colorway: Array.isArray(layout.colorway) && layout.colorway.length ? layout.colorway : theme.colorway,
            font: { family: theme.font, color: theme.text, ...(layout.font || {}) },
            title: typeof layout.title === "object"
                ? { ...layout.title, font: { color: theme.text, ...(layout.title.font || {}) } }
                : layout.title,
            xaxis: applyAxisTheme(layout.xaxis || {}, theme),
            yaxis: applyAxisTheme(layout.yaxis || {}, theme),
        };
        themedFigure.data = Array.isArray(themedFigure.data)
            ? themedFigure.data.map((trace, index) => applyTraceTheme(trace, theme, index))
            : [];
        return themedFigure;
    }

    function dashboardBaseUrl() {
        const configured = window.AmanajeSettings?.services?.plotly_dashboard_url;
        if (configured) return String(configured).replace(/\/$/, "");
        const location = window.location || {};
        const protocol = location.protocol || "http:";
        const hostname = location.hostname || "localhost";
        return `${protocol}//${hostname}:8050`;
    }

    function encodeForQuery(value) {
        const json = JSON.stringify(value ?? {});
        return btoa(unescape(encodeURIComponent(json)));
    }

    function renderSeriesBars(plot = {}) {
        const series = Array.isArray(plot.series) ? plot.series : [];
        if (!series.length) return '<div class="helper-text">No numeric series available.</div>';
        const maxValue = Math.max(...series.map((item) => Math.abs(Number(item.value) || 0)), 1);
        return `
            <div class="plot-bars">
                ${series.map((item) => {
                    const value = Number(item.value) || 0;
                    const width = Math.max((Math.abs(value) / maxValue) * 100, value !== 0 ? 6 : 0);
                    return `
                        <div class="plot-row">
                            <div class="plot-meta">
                                <span>${escapeHtml(item.label)}</span>
                                <span>${escapeHtml(value.toFixed(3).replace(/\\.000$/, ""))}</span>
                            </div>
                            <div class="plot-track">
                                <div class="plot-fill" style="width:${width}%;"></div>
                            </div>
                        </div>
                    `;
                }).join("")}
            </div>
        `;
    }

    function renderPlotlyFallback(plot = {}) {
        const figure = isPlainObject(plot.figure) ? plot.figure : {};
        const traces = Array.isArray(figure.data) ? figure.data : [];
        const traceTypes = traces.map((trace) => trace?.type || "scatter").filter(Boolean);
        const seriesPreview = Array.isArray(plot.series) && plot.series.length
            ? renderSeriesBars(plot)
            : "";
        return `
            <div class="plotly-fallback" data-plotly-fallback="true">
                <div class="helper-text">
                    Interactive Plotly rendering is available through the Visualization renderer.
                    This page keeps a local preview visible so it still loads cleanly if that service is offline.
                </div>
                <div class="plotly-fallback-meta">
                    ${metadataBadge("Traces", traces.length)}
                    ${metadataBadge("Types", traceTypes.join(", ") || "n/a")}
                </div>
                ${seriesPreview || renderCollapsibleJson(figure, { title: "Figure JSON", collapsed: true })}
            </div>
        `;
    }

    function renderPlotlyFrame(plot = {}) {
        if (!plot.figure) return renderSeriesBars(plot);
        try {
            const encodedFigure = encodeForQuery(applyPlotTheme(plot.figure));
            if (encodedFigure.length > 12000) {
                return renderSeriesBars(plot);
            }
            const src = `${dashboardBaseUrl()}/embed?figure=${encodeURIComponent(encodedFigure)}&title=${encodeURIComponent(plot.title || "Plot")}`;
            return `
                <div class="plotly-embed-shell">
                    <iframe
                        class="plotly-frame"
                        title="${escapeHtml(plot.title || "Plot")}"
                        loading="lazy"
                        src="${escapeHtml(src)}"
                    ></iframe>
                    <div class="plotly-renderer-actions">
                        <a class="plotly-renderer-link" href="${escapeHtml(src)}" target="_blank" rel="noopener noreferrer">
                            Open interactive renderer
                        </a>
                    </div>
                    <details class="plotly-local-fallback">
                        <summary>Local fallback preview</summary>
                        ${renderPlotlyFallback(plot)}
                    </details>
                </div>
            `;
        } catch (_error) {
            return renderSeriesBars(plot);
        }
    }

    function plotIdentityMetadata(plot = {}) {
        const source = isPlainObject(plot.source) ? plot.source : {};
        const artifact = isPlainObject(plot.artifact) ? plot.artifact : {};
        const metadata = isPlainObject(plot.metadata) ? { ...plot.metadata } : {};
        const candidates = {
            display_id: metadata.display_id || plot.id,
            domain: metadata.domain || source.domain,
            kind: metadata.kind || plot.kind,
            plot_type: metadata.plot_type || plot.plot_type,
            job_id: metadata.job_id || source.job_id,
            model_id: metadata.model_id || source.model_id,
            dataset_id: metadata.dataset_id || source.dataset_id,
            inference_id: metadata.inference_id || source.inference_id,
            registry_type: metadata.registry_type || source.registry_type,
            registry_id: metadata.registry_id || source.registry_id,
            filename: metadata.filename || artifact.filename,
            relative_path: metadata.relative_path || artifact.relative_path,
            extension: metadata.extension || artifact.extension,
            size_bytes: metadata.size_bytes || artifact.size_bytes,
            modified_at: metadata.modified_at || artifact.modified_at,
            artifact_url: metadata.artifact_url || artifact.url || artifact.public_url,
        };
        Object.entries(candidates).forEach(([key, value]) => {
            if (value !== undefined && value !== null && value !== "") metadata[key] = value;
        });
        return metadata;
    }

    function metadataBadge(label, value) {
        if (value === undefined || value === null || value === "") return "";
        return `<span class="plot-metadata-badge"><strong>${escapeHtml(label)}</strong>${escapeHtml(value)}</span>`;
    }

    function renderPlotMetadataBadges(metadata = {}) {
        const badges = [
            metadataBadge("Type", metadata.plot_type || metadata.kind),
            metadataBadge("Source", metadata.domain),
            metadataBadge("Job", metadata.job_id),
            metadataBadge("Model", metadata.model_id),
            metadataBadge("Dataset", metadata.dataset_id),
            metadataBadge("Inference", metadata.inference_id),
            metadataBadge("File", metadata.filename),
        ].filter(Boolean);
        return badges.length ? `<div class="plot-metadata-badges">${badges.join("")}</div>` : "";
    }

    function renderPlotMetadataDetails(metadata = {}) {
        if (!Object.keys(metadata).length) return "";
        return renderCollapsibleJson(metadata, {
            title: "Plot Metadata",
            collapsed: true,
            maxPlotlyRows: 40,
            maxPlotlyColumns: 10,
        });
    }

    function encodedMetadataAttribute(metadata = {}) {
        try {
            return escapeHtml(JSON.stringify(metadata));
        } catch (_error) {
            return "{}";
        }
    }

    function renderLegacyImage(plot = {}) {
        const artifact = plot.artifact || {};
        const src = artifact.url || artifact.public_url || "";
        if (!src) return '<div class="helper-text">Plot artifact is missing a renderable URL.</div>';
        const metadata = plotIdentityMetadata(plot);
        return `
            <button
                type="button"
                class="plot-image-trigger"
                data-plot-lightbox="true"
                data-image-src="${escapeHtml(src)}"
                data-image-title="${escapeHtml(plot.title || artifact.filename || "Plot artifact")}"
                data-image-description="${escapeHtml(plot.description || "")}"
                data-image-metadata="${encodedMetadataAttribute(metadata)}"
            >
                <img class="plot-image" src="${escapeHtml(src)}" alt="${escapeHtml(plot.title || artifact.filename || "Plot artifact")}">
            </button>
        `;
    }

    function renderTablePlot(plot = {}) {
        const rows = Array.isArray(plot.rows) ? plot.rows : [];
        const columns = Array.isArray(plot.columns) && plot.columns.length
            ? plot.columns
            : (rows[0] ? Object.keys(rows[0]) : []);
        if (!rows.length || !columns.length) return '<div class="helper-text">No table rows available.</div>';
        return `
            <div class="table-scroll">
                <table class="full-width-table">
                    <thead>
                        <tr>${columns.map((column) => `<th>${escapeHtml(column)}</th>`).join("")}</tr>
                    </thead>
                    <tbody>
                        ${rows.map((row) => `
                            <tr>
                                ${columns.map((column) => {
                                    const value = row?.[column];
                                    const rendered = typeof value === "object" ? JSON.stringify(value) : value;
                                    return `<td>${escapeHtml(rendered)}</td>`;
                                }).join("")}
                            </tr>
                        `).join("")}
                    </tbody>
                </table>
            </div>
            ${plot.truncated ? `<div class="helper-text">Showing ${escapeHtml(rows.length)} of ${escapeHtml(plot.row_count || rows.length)} rows.</div>` : ""}
        `;
    }

    function renderPlotBody(plot = {}) {
        if (plot.kind === "plotly_figure" && plot.figure) return renderPlotlyFrame(plot);
        if (plot.kind === "legacy_image") return renderLegacyImage(plot);
        if (plot.kind === "table") return renderTablePlot(plot);
        return renderSeriesBars(plot);
    }

    function renderPlotDeck(plots = []) {
        const usablePlots = Array.isArray(plots)
            ? plots.filter((plot) => plot && (plot.figure || plot.artifact || plot.kind === "table" || (Array.isArray(plot.series) && plot.series.length)))
            : [];
        if (!usablePlots.length) return '<div class="helper-text">No plots available for this result yet.</div>';
        return `
            <div class="plot-grid">
                ${usablePlots.map((plot) => {
                    const metadata = plotIdentityMetadata(plot);
                    return `
                        <article class="card plot-spec-card">
                            <h4>${escapeHtml(plot.title || "Generated Plot")}</h4>
                            ${plot.description ? `<div class="helper-text plot-description">${escapeHtml(plot.description)}</div>` : ""}
                            ${renderPlotMetadataBadges(metadata)}
                            ${renderPlotBody(plot)}
                            ${renderPlotMetadataDetails(metadata)}
                        </article>
                    `;
                }).join("")}
            </div>
        `;
    }

    function ensurePlotLightbox() {
        let modal = document.getElementById("amanajePlotLightbox");
        if (modal) return modal;
        document.body.insertAdjacentHTML("beforeend", `
            <div class="plot-lightbox" id="amanajePlotLightbox" hidden data-plot-lightbox-overlay="true">
                <div class="plot-lightbox-dialog" role="dialog" aria-modal="true" aria-labelledby="plotLightboxTitle">
                    <button type="button" class="plot-lightbox-close" data-plot-lightbox-close="true" aria-label="Close expanded plot">&times;</button>
                    <div class="plot-lightbox-content">
                        <img id="plotLightboxImage" class="plot-lightbox-image" alt="">
                        <div class="plot-lightbox-side">
                            <h3 id="plotLightboxTitle"></h3>
                            <p id="plotLightboxDescription" class="helper-text"></p>
                            <div id="plotLightboxMetadata"></div>
                        </div>
                    </div>
                </div>
            </div>
        `);
        return document.getElementById("amanajePlotLightbox");
    }

    function openPlotImageLightbox({ src, title, description, metadata }) {
        const modal = ensurePlotLightbox();
        const image = modal.querySelector("#plotLightboxImage");
        const titleElement = modal.querySelector("#plotLightboxTitle");
        const descriptionElement = modal.querySelector("#plotLightboxDescription");
        const metadataElement = modal.querySelector("#plotLightboxMetadata");
        image.src = src;
        image.alt = title || "Expanded plot";
        titleElement.textContent = title || "Plot artifact";
        descriptionElement.textContent = description || "";
        metadataElement.innerHTML = metadata && Object.keys(metadata).length
            ? renderJsonExplorer(metadata, { title: "Plot Metadata", maxPlotlyRows: 40, maxPlotlyColumns: 10 })
            : "";
        modal.hidden = false;
        document.body.classList.add("plot-lightbox-open");
        activateTabs(modal);
        enhanceCollapsibles(modal);
        modal.querySelector(".plot-lightbox-close")?.focus();
    }

    function closePlotImageLightbox() {
        const modal = document.getElementById("amanajePlotLightbox");
        if (!modal) return;
        modal.hidden = true;
        document.body.classList.remove("plot-lightbox-open");
        const image = modal.querySelector("#plotLightboxImage");
        if (image) image.removeAttribute("src");
    }

    function initPlotInteractions() {
        if (window.AmanajePlotInteractionsBound) return;
        window.AmanajePlotInteractionsBound = true;
        document.addEventListener("click", (event) => {
            const target = event.target instanceof Element ? event.target : null;
            if (!target) return;
            const lightboxTrigger = target.closest("[data-plot-lightbox='true']");
            if (lightboxTrigger) {
                let metadata = {};
                try {
                    metadata = JSON.parse(lightboxTrigger.getAttribute("data-image-metadata") || "{}");
                } catch (_error) {
                    metadata = {};
                }
                openPlotImageLightbox({
                    src: lightboxTrigger.getAttribute("data-image-src") || "",
                    title: lightboxTrigger.getAttribute("data-image-title") || "Plot artifact",
                    description: lightboxTrigger.getAttribute("data-image-description") || "",
                    metadata,
                });
                return;
            }
            if (
                target.closest("[data-plot-lightbox-close='true']")
                || target.getAttribute?.("data-plot-lightbox-overlay") === "true"
            ) {
                closePlotImageLightbox();
            }
        });
        document.addEventListener("keydown", (event) => {
            if (event.key === "Escape") closePlotImageLightbox();
        });
    }

    function renderResultTabs(tabs = [], options = {}) {
        const usableTabs = Array.isArray(tabs) ? tabs.filter((tab) => tab && tab.key) : [];
        if (!usableTabs.length) return "";
        const prefix = options.idPrefix || `tabs_${Date.now()}`;
        const activeKey = options.activeKey || usableTabs[0].key;
        return `
            <div class="result-tabs" data-result-tabs="${escapeHtml(prefix)}">
                <div class="result-tab-strip">
                    ${usableTabs.map((tab) => `
                        <button
                            type="button"
                            class="result-tab-btn ${tab.key === activeKey ? "active" : ""}"
                            data-result-tab-button="${escapeHtml(prefix)}"
                            data-tab-key="${escapeHtml(tab.key)}"
                        >
                            ${escapeHtml(tab.label || tab.key)}
                        </button>
                    `).join("")}
                </div>
                ${usableTabs.map((tab) => `
                    <section
                        class="result-tab-panel ${tab.key === activeKey ? "active" : ""}"
                        data-result-tab-panel="${escapeHtml(prefix)}"
                        data-tab-key="${escapeHtml(tab.key)}"
                    >
                        ${tab.content || ""}
                    </section>
                `).join("")}
            </div>
        `;
    }

    function activateTabs(root = document) {
        root.querySelectorAll("[data-result-tab-button]").forEach((button) => {
            if (button.dataset.resultTabBound === "true") return;
            button.dataset.resultTabBound = "true";
            button.addEventListener("click", () => {
                const group = button.dataset.resultTabButton;
                const tabKey = button.dataset.tabKey;
                root.querySelectorAll(`[data-result-tab-button="${group}"]`).forEach((item) => {
                    item.classList.toggle("active", item.dataset.tabKey === tabKey);
                });
                root.querySelectorAll(`[data-result-tab-panel="${group}"]`).forEach((panel) => {
                    panel.classList.toggle("active", panel.dataset.tabKey === tabKey);
                });
                enhanceCollapsibles(root);
            });
        });
    }

    function shouldCollapse(text) {
        const normalized = String(text || "").trim();
        return normalized.length > CHAR_LIMIT || normalized.split(/\r?\n/).length > LINE_LIMIT;
    }

    function looksLikeJson(text) {
        const normalized = String(text || "").trim();
        return normalized.startsWith("{") || normalized.startsWith("[");
    }

    function prepareJsonDisplays(root = document) {
        root.querySelectorAll(".json-view, .summary-table code, .full-width-table code, .table-scroll code, code[data-auto-json='true']").forEach((element) => {
            const text = element.textContent || "";
            if (!text.trim()) return;
            if (!looksLikeJson(text) && !shouldCollapse(text)) return;

            element.classList.add("json-block", "inline-json-block");
            if (!element.dataset.collapseTitle) {
                element.dataset.collapseTitle = "JSON Details";
            }
        });
    }

    function enhanceCollapsibles(root = document) {
        prepareJsonDisplays(root);

        root.querySelectorAll(".json-view, .json-block, .log-block, .inline-json-block, [data-auto-collapsible='true']").forEach((block, index) => {
            if (block.closest("[data-collapsible-wrapper='true']")) return;
            if (block.closest("[data-json-explorer='true']")) return;
            if (!shouldCollapse(block.textContent || "")) return;
            const wrapper = document.createElement("div");
            const title = block.dataset.collapseTitle || `Details ${index + 1}`;
            const text = block.textContent || "";
            if (looksLikeJson(text)) {
                try {
                    wrapper.innerHTML = renderCollapsibleJson(JSON.parse(text), { title, collapsed: true });
                } catch (_error) {
                    wrapper.innerHTML = renderCollapsibleContent(block.outerHTML, { title, collapsed: true });
                }
            } else {
                wrapper.innerHTML = renderCollapsibleContent(block.outerHTML, { title, collapsed: true });
            }
            const replacement = wrapper.firstElementChild;
            block.replaceWith(replacement);
        });

        root.querySelectorAll("[data-collapsible-toggle='true']").forEach((button) => {
            if (button.dataset.collapsibleBound === "true") return;
            button.dataset.collapsibleBound = "true";
            button.addEventListener("click", () => {
                const wrapper = button.closest("[data-collapsible-wrapper='true']");
                const body = wrapper ? wrapper.querySelector(".amanaje-collapse-body") : null;
                if (!body) return;
                const isHidden = body.hasAttribute("hidden");
                if (isHidden) {
                    body.removeAttribute("hidden");
                    button.textContent = "Collapse";
                    button.setAttribute("aria-expanded", "true");
                } else {
                    body.setAttribute("hidden", "");
                    button.textContent = "Expand";
                    button.setAttribute("aria-expanded", "false");
                }
            });
        });
    }

    function initAssistantCollapsibles(root = document) {
        root.querySelectorAll("[data-assistant-toggle]").forEach((button) => {
            if (button.dataset.assistantToggleBound === "true") return;
            button.dataset.assistantToggleBound = "true";
            const wrapper = button.closest(".assistant-collapsible");
            const panel = wrapper ? wrapper.querySelector("[data-assistant-panel]") : null;
            const expanded = button.getAttribute("aria-expanded") === "true";
            if (panel && !panel.id) {
                panel.id = `assistant-panel-${button.dataset.assistantToggle || "global"}`;
            }
            if (panel) {
                button.setAttribute("aria-controls", panel.id);
            }
            if (panel && !expanded) {
                panel.setAttribute("hidden", "");
            }
            button.addEventListener("click", () => {
                if (!panel) return;
                const isHidden = panel.hasAttribute("hidden");
                if (isHidden) {
                    panel.removeAttribute("hidden");
                    button.setAttribute("aria-expanded", "true");
                } else {
                    panel.setAttribute("hidden", "");
                    button.setAttribute("aria-expanded", "false");
                }
            });
        });
    }

    function setAssistantStatus(key, label = "Ready", tone = "idle") {
        const badge = document.querySelector(`[data-assistant-status="${key}"]`);
        if (!badge) return;
        badge.textContent = label;
        badge.className = `assistant-status-badge ${tone || "idle"}`;
    }

    function isDebugModeEnabled() {
        return Boolean(window.AmanajeDebug || window.AmanajeSettings?.feature_flags?.debug_mode);
    }

    function captureFrontendError(error, context = {}) {
        const normalizedError = error instanceof Error ? error : new Error(String(error ?? "Unknown frontend error"));
        const debugEnabled = isDebugModeEnabled();
        const safeContext = { ...(context || {}) };
        if (!debugEnabled) {
            delete safeContext.backend_debug;
            delete safeContext.frontend_stack;
            delete safeContext.stack;
            delete safeContext.traceback;
        }
        const payload = {
            message: normalizedError.message,
            name: normalizedError.name || "Error",
            page: {
                path: window.location?.pathname || "",
                search: window.location?.search || "",
                title: document.title || "",
            },
            context: safeContext,
            captured_at: new Date().toISOString(),
        };
        if (debugEnabled) {
            payload.stack = normalizedError.stack || null;
            payload.backend_debug = safeContext.backend_debug || normalizedError.backendDebug || null;
        }
        window.AmanajeLastFrontendError = payload;
        window.dispatchEvent(new CustomEvent("amanaje:frontend-error", { detail: payload }));
        if (debugEnabled) {
            console.debug("[Amanaje] Frontend debug capture", payload);
        }
        return payload;
    }

    function decorateApiError(error, payload, context = {}) {
        error.payload = payload;
        if (payload?.debug) {
            error.backendDebug = payload.debug;
        }
        error.fetchContext = context;
        return error;
    }

    function shouldSetJsonContentType(body) {
        return Boolean(body) && !(typeof FormData !== "undefined" && body instanceof FormData);
    }

    async function fetchJson(url, options = {}, settings = {}) {
        const method = String(options.method || "GET").toUpperCase();
        const context = {
            url: String(url),
            method,
            source: settings.source || "AmanajeUI.fetchJson",
        };
        const headers = {
            ...(shouldSetJsonContentType(options.body) ? { "Content-Type": "application/json" } : {}),
            ...(options.headers || {}),
        };

        try {
            window.__amanajeFetchJsonDepth = (window.__amanajeFetchJsonDepth || 0) + 1;
            let response;
            try {
                response = await window.fetch(url, { ...options, headers });
            } finally {
                window.__amanajeFetchJsonDepth = Math.max((window.__amanajeFetchJsonDepth || 1) - 1, 0);
            }
            let payload = {};
            try {
                payload = await response.json();
            } catch (parseError) {
                payload = { detail: `Unable to parse response JSON: ${parseError.message}` };
            }
            if ((!response.ok || payload?.status === "error") && !settings.allowError) {
                const error = decorateApiError(
                    new Error(payload.detail || payload.error || `HTTP ${response.status}`),
                    payload,
                    { ...context, status: response.status, backend_debug: payload.debug || null }
                );
                if (isDebugModeEnabled()) {
                    error.__amanajeCaptured = true;
                    captureFrontendError(error, error.fetchContext);
                }
                throw error;
            }
            return payload;
        } catch (error) {
            if (isDebugModeEnabled() && !error.__amanajeCaptured) {
                error.__amanajeCaptured = true;
                captureFrontendError(error, context);
            }
            throw error;
        }
    }

    function installFetchGuard() {
        if (window.__amanajeFetchGuardInstalled || typeof window.fetch !== "function") return;
        window.__amanajeFetchGuardInstalled = true;
        const nativeFetch = window.fetch.bind(window);
        window.fetch = async (...args) => {
            const [resource, options = {}] = args;
            const method = String(options?.method || "GET").toUpperCase();
            const url = typeof resource === "string" ? resource : (resource?.url || String(resource));
            try {
                const response = await nativeFetch(...args);
                if (isDebugModeEnabled() && !window.__amanajeFetchJsonDepth && response && response.ok === false) {
                    let payload = {};
                    try {
                        payload = await response.clone().json();
                    } catch (_error) {
                        payload = {};
                    }
                    captureFrontendError(
                        new Error(payload.detail || payload.error || `HTTP ${response.status}`),
                        {
                            url,
                            method,
                            status: response.status,
                            source: "window.fetch",
                            backend_debug: payload.debug || null,
                        }
                    );
                }
                return response;
            } catch (error) {
                if (isDebugModeEnabled()) {
                    error.__amanajeCaptured = true;
                    captureFrontendError(error, { url, method, source: "window.fetch" });
                }
                throw error;
            }
        };
    }

    async function loadAssistantModelOptions(select, options = {}) {
        const element = typeof select === "string" ? document.getElementById(select) : select;
        if (!element) return [];
        const storageKey = options.storageKey || "amanajeAssistantModelId";
        const storedValue = window.localStorage?.getItem(storageKey) || "";
        const currentValue = element.value || storedValue;
        const defaultLabel = options.defaultLabel || "Configured active provider";
        element.innerHTML = `<option value="">${escapeHtml(defaultLabel)}</option>`;
        if (!element.dataset.assistantSelectionBound) {
            element.addEventListener("change", () => {
                const value = String(element.value || "").trim();
                if (value) {
                    window.localStorage?.setItem(storageKey, value);
                } else {
                    window.localStorage?.removeItem(storageKey);
                }
            });
            element.dataset.assistantSelectionBound = "true";
        }
        try {
            const payload = await fetchJson("/assistant/models/list", {}, { source: "loadAssistantModelOptions" });
            const models = Array.isArray(payload.models) ? payload.models : [];
            element.insertAdjacentHTML(
                "beforeend",
                models.map((model) => {
                    const id = model.id;
                    const name = model.name || `Assistant Model #${id}`;
                    const version = model.model_version || model.assistant_config?.model_version || model.version || "unversioned";
                    return `<option value="${escapeHtml(id)}">${escapeHtml(name)} (${escapeHtml(version)})</option>`;
                }).join("")
            );
            if (currentValue && Array.from(element.options).some((option) => option.value === currentValue)) {
                element.value = currentValue;
            }
            return models;
        } catch (error) {
            element.insertAdjacentHTML("beforeend", `<option value="" disabled>Registry models unavailable</option>`);
            return [];
        }
    }

    function getAssistantModelRequest(select) {
        const element = typeof select === "string" ? document.getElementById(select) : select;
        const selected = element?.value ? String(element.value).trim() : "";
        if (selected) {
            window.localStorage?.setItem("amanajeAssistantModelId", selected);
        }
        return selected ? { assistant_model_id: selected, provider: "auto" } : { provider: "auto" };
    }

    function applyGlobalSettings(settings = window.AmanajeSettings || {}) {
        const currentSettings = window.AmanajeSettings || {};
        const currentFlags = currentSettings.feature_flags || {};
        const nextFlags = {
            ...currentFlags,
            ...(settings.feature_flags || {}),
        };
        window.AmanajeSettings = {
            ...currentSettings,
            ...settings,
            feature_flags: nextFlags,
        };

        const debugEnabled = Boolean(nextFlags.debug_mode);
        const assistantVisible = nextFlags.assistant_visible !== false;
        if (document.body) {
            document.body.dataset.debugMode = debugEnabled ? "true" : "false";
            document.body.dataset.assistantVisible = assistantVisible ? "true" : "false";
        }
        window.AmanajeDebug = debugEnabled;
        if (debugEnabled) {
            console.debug("[Amanaje] Debug mode enabled.", {
                assistantVisible,
                updatedAt: window.AmanajeSettings.updated_at || null,
            });
        }
    }

    const OPERATION_STORAGE_KEY = "amanaje:watched-run-ids";
    let operationPollTimer = null;
    let operationStackBusy = false;

    function getWatchedRunIds() {
        try {
            const parsed = JSON.parse(window.localStorage.getItem(OPERATION_STORAGE_KEY) || "[]");
            return Array.isArray(parsed) ? parsed.map(String).filter(Boolean).slice(0, 30) : [];
        } catch (_error) {
            return [];
        }
    }

    function saveWatchedRunIds(ids = []) {
        const unique = Array.from(new Set((ids || []).map(String).filter(Boolean))).slice(0, 30);
        try {
            window.localStorage.setItem(OPERATION_STORAGE_KEY, JSON.stringify(unique));
        } catch (_error) {
            return unique;
        }
        return unique;
    }

    function watchRun(runId, options = {}) {
        const normalized = String(runId || "").trim();
        if (!normalized) return [];
        const watched = saveWatchedRunIds([normalized, ...getWatchedRunIds()]);
        window.dispatchEvent(new CustomEvent("amanaje:operation-watch", {
            detail: { run_id: normalized, source: options.source || "unknown" },
        }));
        loadOperationStack({ immediate: true });
        return watched;
    }

    function unwatchRun(runId) {
        const normalized = String(runId || "").trim();
        const watched = saveWatchedRunIds(getWatchedRunIds().filter((id) => id !== normalized));
        loadOperationStack({ immediate: true });
        return watched;
    }

    function mergeRuns(...groups) {
        const byId = new Map();
        groups.flat().filter(Boolean).forEach((run) => {
            const id = String(run.run_id || "");
            if (!id) return;
            byId.set(id, run);
        });
        return Array.from(byId.values()).sort((left, right) => {
            const leftTime = left.updated_at || left.created_at || "";
            const rightTime = right.updated_at || right.created_at || "";
            return rightTime.localeCompare(leftTime);
        });
    }

    function statusTone(status) {
        if (status === "completed") return "success";
        if (status === "failed" || status === "cancelled") return "error";
        if (status === "running" || status === "cancel_requested") return "running";
        if (status === "paused") return "queued";
        return "queued";
    }

    function formatRunTitle(run = {}) {
        const type = String(run.run_type || "operation").replace(/_/g, " ");
        const context = run.context || {};
        const parts = [];
        if (context.operation) parts.push(String(context.operation).replace(/_/g, " "));
        if (context.model_id) parts.push(`model ${context.model_id}`);
        if (context.dataset_id) parts.push(`dataset ${context.dataset_id}`);
        if (context.study_id) parts.push(`study ${context.study_id}`);
        return `${type}${parts.length ? ` - ${parts.join(" - ")}` : ""}`;
    }

    function runQueueLabel(run = {}) {
        const diagnostics = run.queue_diagnostics || {};
        const queue = diagnostics.queue || run.queue?.queue || run.queue?.name || "";
        const position = diagnostics.position ? ` #${diagnostics.position}` : "";
        return queue ? `${queue}${position}` : "";
    }

    function renderWorkerSummary(workerRuntime = {}) {
        const target = document.getElementById("globalOperationWorkerSummary");
        if (!target) return;
        const status = workerRuntime.status || "unknown";
        const summary = workerRuntime.summary || {};
        const queues = Array.isArray(workerRuntime.queues) ? workerRuntime.queues : [];
        const queueText = queues
            .map((queue) => `${queue.name}: ${queue.queued_jobs || 0}`)
            .join(" | ");
        target.dataset.status = status;
        target.textContent = status === "ok"
            ? `Workers: ${summary.live_workers || summary.active_workers || 0} live, ${summary.stale_workers || 0} stale | queued ${summary.queued_jobs || 0}${queueText ? ` | ${queueText}` : ""}`
            : `Workers: ${status}${workerRuntime.error ? ` | ${workerRuntime.error}` : ""}`;
    }

    function renderOperationStack(runs = []) {
        const list = document.getElementById("globalOperationList");
        const count = document.getElementById("globalOperationCount");
        if (!list || !count) return;
        const activeCount = runs.filter((run) => ["queued", "running", "paused", "cancel_requested"].includes(String(run.status || ""))).length;
        count.textContent = String(activeCount || runs.length || 0);
        count.dataset.tone = activeCount ? "active" : "idle";

        if (!runs.length) {
            list.innerHTML = '<div class="global-operation-empty">No operations are running.</div>';
            return;
        }

        list.innerHTML = runs.slice(0, 12).map((run) => {
            const tone = statusTone(String(run.status || "queued"));
            const latestEvent = run.progress?.latest_event || run.error || "";
            const runId = String(run.run_id || "");
            const status = String(run.status || "queued");
            const failure = run.failure?.message || "";
            const queueLabel = runQueueLabel(run);
            const canPause = ["queued", "running"].includes(status);
            const canResume = status === "paused";
            const canCancel = ["queued", "running", "paused", "cancel_requested"].includes(status);
            return `
                <article class="global-operation-item ${tone}">
                    <div class="global-operation-main">
                        <strong>${escapeHtml(formatRunTitle(run))}</strong>
                        <span>${escapeHtml(run.stage || run.status || "queued")}</span>
                    </div>
                    <div class="global-operation-meta">
                        <code>${escapeHtml(runId)}</code>
                        <span>${escapeHtml(status)}</span>
                        ${queueLabel ? `<span>${escapeHtml(queueLabel)}</span>` : ""}
                    </div>
                    ${failure ? `<div class="global-operation-event">${escapeHtml(failure)}</div>` : (latestEvent ? `<div class="global-operation-event">${escapeHtml(latestEvent)}</div>` : "")}
                    <div class="global-operation-actions">
                        ${canPause ? `<button type="button" data-run-control="pause" data-run-id="${escapeHtml(runId)}">Pause</button>` : ""}
                        ${canResume ? `<button type="button" data-run-control="resume" data-run-id="${escapeHtml(runId)}">Resume</button>` : ""}
                        ${canCancel ? `<button type="button" data-run-control="cancel" data-run-id="${escapeHtml(runId)}">Cancel</button>` : ""}
                        <a href="/operations" class="global-operation-dismiss">Open</a>
                        <button type="button" class="global-operation-dismiss" data-run-dismiss="${escapeHtml(runId)}" aria-label="Dismiss operation">Dismiss</button>
                    </div>
                </article>
            `;
        }).join("");

        list.querySelectorAll("[data-run-dismiss]").forEach((button) => {
            button.addEventListener("click", () => unwatchRun(button.dataset.runDismiss));
        });
        list.querySelectorAll("[data-run-control]").forEach((button) => {
            button.addEventListener("click", () => controlRun(button.dataset.runId, button.dataset.runControl));
        });
    }

    async function controlRun(runId, action) {
        const normalizedRunId = String(runId || "").trim();
        const normalizedAction = String(action || "").trim().toLowerCase();
        if (!normalizedRunId || !["pause", "resume", "cancel"].includes(normalizedAction)) return null;
        const result = await fetchJson(`/runs/${encodeURIComponent(normalizedRunId)}/${normalizedAction}`, {
            method: "POST",
        }, { source: "operation-stack" });
        watchRun(normalizedRunId, { source: `operation-${normalizedAction}` });
        await loadOperationStack({ immediate: true });
        return result;
    }

    async function loadOperationStack(options = {}) {
        const panel = document.getElementById("globalOperationStack");
        if (!panel || operationStackBusy) return [];
        operationStackBusy = true;
        try {
            const watchedIds = getWatchedRunIds();
            const summaryUrl = `/operations/summary?limit=12${watchedIds.length ? `&run_ids=${encodeURIComponent(watchedIds.join(","))}` : ""}`;
            const recent = await fetchJson(summaryUrl, {}, { source: "operation-stack", allowError: true });
            const workerRuntime = recent.worker_runtime || {};
            renderWorkerSummary(workerRuntime);
            const runs = mergeRuns(recent.runs || [], recent.active_runs || [], recent.recent_failures || []);
            renderOperationStack(runs);
            window.dispatchEvent(new CustomEvent("amanaje:operation-stack-update", { detail: { runs } }));
            const hasActive = runs.some((run) => ["queued", "running", "paused", "cancel_requested"].includes(String(run.status || "")));
            const body = document.getElementById("globalOperationBody");
            const panelOpen = Boolean(body && !body.hasAttribute("hidden"));
            if (operationPollTimer) {
                window.clearTimeout(operationPollTimer);
                operationPollTimer = null;
            }
            if (hasActive || panelOpen) {
                operationPollTimer = window.setTimeout(loadOperationStack, hasActive ? 2500 : 10000);
            }
            return runs;
        } catch (error) {
            const list = document.getElementById("globalOperationList");
            if (list) list.innerHTML = `<div class="global-operation-empty">${escapeHtml(error.message)}</div>`;
            return [];
        } finally {
            operationStackBusy = false;
        }
    }

    function initOperationStack() {
        const panel = document.getElementById("globalOperationStack");
        if (!panel || panel.dataset.bound === "true") return;
        panel.dataset.bound = "true";
        const toggle = document.getElementById("globalOperationToggle");
        const body = document.getElementById("globalOperationBody");
        toggle?.addEventListener("click", () => {
            const expanded = toggle.getAttribute("aria-expanded") === "true";
            toggle.setAttribute("aria-expanded", expanded ? "false" : "true");
            body?.toggleAttribute("hidden", expanded);
            if (!expanded) {
                loadOperationStack({ immediate: true });
            }
        });
        document.getElementById("globalOperationRefresh")?.addEventListener("click", () => loadOperationStack({ immediate: true }));
        loadOperationStack({ immediate: true });
    }

    window.AmanajeUI = {
        escapeHtml,
        isDebugModeEnabled,
        captureFrontendError,
        fetchJson,
        renderMetricCards,
        renderCollapsibleContent,
        renderCollapsibleJson,
        normalizeJsonTable,
        renderJsonExplorer,
        renderPlotlyTableSpec,
        applyPlotTheme,
        dashboardBaseUrl,
        plotIdentityMetadata,
        renderPlotDeck,
        renderResultTabs,
        activateTabs,
        enhanceCollapsibles,
        initAssistantCollapsibles,
        applyGlobalSettings,
        loadAssistantModelOptions,
        getAssistantModelRequest,
        setAssistantStatus,
        watchRun,
        unwatchRun,
        controlRun,
        loadOperationStack,
    };

    document.addEventListener("DOMContentLoaded", () => {
        installFetchGuard();
        applyGlobalSettings(window.AmanajeSettings || {});
        activateTabs(document);
        enhanceCollapsibles(document);
        initAssistantCollapsibles(document);
        initPlotInteractions();
        initOperationStack();
        const observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                mutation.addedNodes.forEach((node) => {
                    if (!(node instanceof HTMLElement)) return;
                    activateTabs(node);
                    enhanceCollapsibles(node);
                    initAssistantCollapsibles(node);
                });
            });
        });
        observer.observe(document.body, { childList: true, subtree: true });
    });

    window.addEventListener("error", (event) => {
        if (!isDebugModeEnabled()) return;
        captureFrontendError(event.error || event.message, {
            source: "window.onerror",
            file: event.filename,
            line: event.lineno,
            column: event.colno,
        });
    });

    window.addEventListener("unhandledrejection", (event) => {
        if (!isDebugModeEnabled()) return;
        captureFrontendError(event.reason, { source: "unhandledrejection" });
    });
})();
