(function () {
    const state = {
        payload: null,
        logs: {
            intervalId: null,
            sourcesRendered: false,
            loading: false,
        },
    };

    function escapeHtml(value) {
        return window.AmanajeUI?.escapeHtml
            ? window.AmanajeUI.escapeHtml(value)
            : String(value ?? "")
                .replace(/&/g, "&amp;")
                .replace(/</g, "&lt;")
                .replace(/>/g, "&gt;")
                .replace(/"/g, "&quot;")
                .replace(/'/g, "&#39;");
    }

    function setStatus(message, tone = "") {
        const element = document.getElementById("settingsStatus");
        if (!element) return;
        element.textContent = message;
        element.className = `settings-status ${tone}`.trim();
    }

    async function fetchJson(url, options = {}) {
        if (window.AmanajeUI?.fetchJson) {
            return window.AmanajeUI.fetchJson(url, options, { source: "settings-page" });
        }
        const response = await fetch(url, {
            headers: { "Content-Type": "application/json", ...(options.headers || {}) },
            ...options,
        });
        const payload = await response.json().catch(() => ({}));
        if (!response.ok || payload?.status === "error") {
            const error = new Error(payload?.detail || `HTTP ${response.status}`);
            error.payload = payload;
            throw error;
        }
        return payload;
    }

    function sourceLabel(row) {
        if (row.source === "saved_override") return "Saved";
        if (row.source === "process") return "Process";
        return "Default";
    }

    function renderWarnings(payload) {
        const container = document.getElementById("settingsWarnings");
        if (!container) return;
        const warnings = Array.isArray(payload?.warnings) ? payload.warnings : [];
        if (!warnings.length) {
            container.innerHTML = "";
            return;
        }
        container.innerHTML = warnings
            .map((warning) => `<div class="settings-warning">${escapeHtml(warning)}</div>`)
            .join("");
    }

    function renderFeatureFlags(payload) {
        const flags = payload?.feature_flags || {};
        const debugMode = document.getElementById("settingsDebugMode");
        const assistantVisible = document.getElementById("settingsAssistantVisible");
        const dashExtensionsEnabled = document.getElementById("settingsDashExtensionsEnabled");
        if (debugMode) debugMode.checked = Boolean(flags.debug_mode);
        if (assistantVisible) assistantVisible.checked = flags.assistant_visible !== false;
        if (dashExtensionsEnabled) {
            const enabled = Boolean(payload?.services?.dashboard_extensions_enabled);
            dashExtensionsEnabled.checked = enabled;
            dashExtensionsEnabled.dataset.originalValue = enabled ? "true" : "false";
        }
    }

    function renderEnvironmentTable(payload) {
        const container = document.getElementById("settingsEnvironmentTable");
        if (!container) return;
        const rows = (Array.isArray(payload?.environment_variables) ? payload.environment_variables : [])
            .filter((row) => row.key !== "AMANAJE_DASH_EXTENSIONS_ENABLED");
        if (!rows.length) {
            container.innerHTML = "No editable environment variables are configured.";
            return;
        }

        container.innerHTML = `
            <div class="settings-table-wrap">
                <table class="settings-table">
                    <thead>
                        <tr>
                            <th>Variable</th>
                            <th>Saved Override</th>
                            <th>Effective Value</th>
                            <th>Source</th>
                            <th>Restart</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${rows.map((row) => {
                            const savedValue = row.saved_override ?? "";
                            const effectiveValue = row.value ?? "";
                            const inputType = row.redacted ? "password" : "text";
                            return `
                                <tr>
                                    <td>
                                        <div class="settings-key">${escapeHtml(row.key)}</div>
                                        <div class="helper-text">${escapeHtml(row.label || row.category || "")}</div>
                                    </td>
                                    <td>
                                        <input
                                            type="${inputType}"
                                            data-settings-env-key="${escapeHtml(row.key)}"
                                            data-original-value="${escapeHtml(savedValue)}"
                                            value="${escapeHtml(savedValue)}"
                                            placeholder="Use process/default value"
                                            autocomplete="off"
                                        >
                                    </td>
                                    <td><code>${escapeHtml(effectiveValue)}</code></td>
                                    <td><span class="settings-badge">${escapeHtml(sourceLabel(row))}</span></td>
                                    <td>${row.restart_required ? '<span class="settings-badge">Required</span>' : ""}</td>
                                </tr>
                            `;
                        }).join("")}
                    </tbody>
                </table>
            </div>
        `;
    }

    function renderRuntimeSnapshot(payload) {
        const container = document.getElementById("settingsRuntimeSnapshot");
        if (!container) return;
        if (window.AmanajeUI?.renderCollapsibleJson) {
            container.innerHTML = window.AmanajeUI.renderCollapsibleJson(payload?.runtime_config || {}, {
                title: "Runtime Config",
                collapsed: false,
            });
            window.AmanajeUI.enhanceCollapsibles(container);
            return;
        }
        container.innerHTML = `<pre class="json-block">${escapeHtml(JSON.stringify(payload?.runtime_config || {}, null, 2))}</pre>`;
    }

    function renderDebugState(payload) {
        const badge = document.getElementById("settingsDebugState");
        if (!badge) return;
        const enabled = Boolean(payload?.feature_flags?.debug_mode);
        badge.textContent = enabled ? "Debug active" : "Debug inactive";
        badge.className = `settings-badge ${enabled ? "success" : ""}`.trim();
    }

    function renderFrontendDebug(detail = window.AmanajeLastFrontendError || null) {
        const container = document.getElementById("settingsFrontendDebug");
        if (!container) return;
        if (!detail) {
            container.innerHTML = "No frontend error captured in this browser session.";
            return;
        }
        if (window.AmanajeUI?.renderCollapsibleJson) {
            container.innerHTML = window.AmanajeUI.renderCollapsibleJson(detail, {
                title: "Latest Frontend Error",
                collapsed: false,
            });
            window.AmanajeUI.enhanceCollapsibles(container);
            return;
        }
        container.innerHTML = `<pre class="json-block">${escapeHtml(JSON.stringify(detail, null, 2))}</pre>`;
    }

    function setLogsStatus(message, tone = "") {
        const element = document.getElementById("settingsLogsStatus");
        if (!element) return;
        element.textContent = message;
        element.className = `settings-badge ${tone}`.trim();
    }

    function renderLogSources(sources = []) {
        const select = document.getElementById("settingsLogSource");
        if (!select || state.logs.sourcesRendered) return;
        const usableSources = Array.isArray(sources) ? sources : [];
        select.innerHTML = usableSources
            .map((source) => {
                const suffix = source.available === false ? " (unavailable)" : "";
                return `<option value="${escapeHtml(source.id)}">${escapeHtml(source.label || source.id)}${escapeHtml(suffix)}</option>`;
            })
            .join("");
        select.value = "all";
        state.logs.sourcesRendered = true;
    }

    function renderLogWarnings(warnings = []) {
        const container = document.getElementById("settingsLogsWarnings");
        if (!container) return;
        const usableWarnings = Array.isArray(warnings) ? warnings.filter(Boolean) : [];
        container.innerHTML = usableWarnings
            .map((warning) => `<div class="settings-log-warning">${escapeHtml(warning)}</div>`)
            .join("");
    }

    function levelClass(level) {
        return String(level || "info").toLowerCase().replace(/[^a-z0-9_-]/g, "");
    }

    function renderLogEntries(entries = []) {
        const terminal = document.getElementById("settingsLogsTerminal");
        if (!terminal) return;
        const usableEntries = Array.isArray(entries) ? entries : [];
        if (!usableEntries.length) {
            terminal.innerHTML = `<div class="helper-text">No log entries available for this source.</div>`;
            return;
        }
        const shouldStickToBottom = terminal.scrollTop + terminal.clientHeight >= terminal.scrollHeight - 48;
        terminal.innerHTML = usableEntries
            .map((entry) => {
                const timestamp = entry.timestamp || "";
                const source = entry.source_label || entry.source || "Log";
                const level = entry.level || "";
                const message = entry.message || entry.raw || "";
                return `
                    <div class="settings-log-line">
                        <span class="settings-log-time">${escapeHtml(timestamp)}</span>
                        <span class="settings-log-source">${escapeHtml(source)}</span>
                        <span class="settings-log-level ${escapeHtml(levelClass(level))}">${escapeHtml(level || "-")}</span>
                        <span class="settings-log-message">${escapeHtml(message)}</span>
                    </div>
                `;
            })
            .join("");
        if (shouldStickToBottom) {
            terminal.scrollTop = terminal.scrollHeight;
        }
    }

    function renderLogs(payload) {
        renderLogSources(payload?.sources || []);
        renderLogWarnings(payload?.warnings || []);
        renderLogEntries(payload?.entries || []);
        const updated = document.getElementById("settingsLogsUpdated");
        if (updated) {
            updated.textContent = payload?.generated_at
                ? `Updated ${new Date(payload.generated_at).toLocaleTimeString()}`
                : "Updated";
        }
        const warningCount = Array.isArray(payload?.warnings) ? payload.warnings.length : 0;
        setLogsStatus(warningCount ? `${warningCount} warning${warningCount === 1 ? "" : "s"}` : "Live", warningCount ? "warning" : "success");
    }

    function buildLogsUrl() {
        const params = new URLSearchParams();
        params.set("source", document.getElementById("settingsLogSource")?.value || "all");
        params.set("tail", document.getElementById("settingsLogTail")?.value || "160");
        params.set("include_docker", document.getElementById("settingsLogsIncludeDocker")?.checked ? "true" : "false");
        return `/settings/logs?${params.toString()}`;
    }

    async function loadLogs() {
        if (state.logs.loading) return;
        state.logs.loading = true;
        setLogsStatus("Loading");
        try {
            renderLogs(await fetchJson(buildLogsUrl()));
        } catch (error) {
            renderLogWarnings([error.message]);
            setLogsStatus("Error", "error");
        } finally {
            state.logs.loading = false;
        }
    }

    function syncLogPolling() {
        if (state.logs.intervalId) {
            window.clearInterval(state.logs.intervalId);
            state.logs.intervalId = null;
        }
        if (document.getElementById("settingsLogsAutoRefresh")?.checked) {
            state.logs.intervalId = window.setInterval(loadLogs, 5000);
        }
    }

    function render(payload) {
        state.payload = payload;
        renderFeatureFlags(payload);
        renderEnvironmentTable(payload);
        renderRuntimeSnapshot(payload);
        renderWarnings(payload);
        renderDebugState(payload);
        renderFrontendDebug();
        window.AmanajeUI?.applyGlobalSettings?.(payload);
        const restartRequired = Boolean(payload?.restart_required?.required);
        setStatus(restartRequired ? "Saved overrides need restart" : "Settings ready", restartRequired ? "warning" : "success");
    }

    async function loadSettings() {
        setStatus("Loading settings");
        try {
            render(await fetchJson("/settings/config"));
        } catch (error) {
            setStatus(error.message, "error");
        }
    }

    function buildSavePayload() {
        const envOverrides = {};
        document.querySelectorAll("[data-settings-env-key]").forEach((input) => {
            const key = input.dataset.settingsEnvKey;
            const nextValue = input.value.trim();
            const originalValue = input.dataset.originalValue || "";
            if (nextValue !== originalValue) {
                envOverrides[key] = nextValue || null;
            }
        });
        document.querySelectorAll("[data-settings-env-toggle]").forEach((input) => {
            const key = input.dataset.settingsEnvToggle;
            const nextValue = input.checked ? "true" : "false";
            const originalValue = input.dataset.originalValue || "false";
            if (nextValue !== originalValue) {
                envOverrides[key] = nextValue;
            }
        });

        return {
            feature_flags: {
                debug_mode: Boolean(document.getElementById("settingsDebugMode")?.checked),
                assistant_visible: Boolean(document.getElementById("settingsAssistantVisible")?.checked),
            },
            env_overrides: envOverrides,
        };
    }

    function previewFeatureFlags() {
        const payload = state.payload || {};
        const featureFlags = {
            ...(payload.feature_flags || {}),
            debug_mode: Boolean(document.getElementById("settingsDebugMode")?.checked),
            assistant_visible: Boolean(document.getElementById("settingsAssistantVisible")?.checked),
        };
        const previewPayload = {
            ...payload,
            feature_flags: featureFlags,
            services: {
                ...(payload.services || {}),
                dashboard_extensions_enabled: Boolean(document.getElementById("settingsDashExtensionsEnabled")?.checked),
            },
        };
        renderDebugState(previewPayload);
        window.AmanajeUI?.applyGlobalSettings?.(previewPayload);
    }

    async function saveSettings() {
        setStatus("Saving settings");
        try {
            const payload = await fetchJson("/settings/config", {
                method: "POST",
                body: JSON.stringify(buildSavePayload()),
            });
            render(payload);
            setStatus(payload?.restart_required?.required ? "Saved, restart required" : "Saved", "success");
        } catch (error) {
            const errors = Array.isArray(error.payload?.errors)
                ? ` ${error.payload.errors.map((item) => `${item.field}: ${item.reason}`).join("; ")}`
                : "";
            setStatus(`${error.message}${errors}`, "error");
        }
    }

    document.addEventListener("DOMContentLoaded", () => {
        document.getElementById("settingsSave")?.addEventListener("click", saveSettings);
        document.getElementById("settingsReload")?.addEventListener("click", loadSettings);
        document.getElementById("settingsDebugMode")?.addEventListener("change", previewFeatureFlags);
        document.getElementById("settingsAssistantVisible")?.addEventListener("change", previewFeatureFlags);
        document.getElementById("settingsDashExtensionsEnabled")?.addEventListener("change", previewFeatureFlags);
        document.getElementById("settingsLogsRefresh")?.addEventListener("click", loadLogs);
        document.getElementById("settingsLogSource")?.addEventListener("change", loadLogs);
        document.getElementById("settingsLogTail")?.addEventListener("change", loadLogs);
        document.getElementById("settingsLogsIncludeDocker")?.addEventListener("change", loadLogs);
        document.getElementById("settingsLogsAutoRefresh")?.addEventListener("change", syncLogPolling);
        window.addEventListener("amanaje:frontend-error", (event) => {
            renderFrontendDebug(event.detail);
        });
        loadSettings();
        loadLogs();
        syncLogPolling();
    });
})();
