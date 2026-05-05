(function () {
    const state = {
        payload: null,
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
        if (debugMode) debugMode.checked = Boolean(flags.debug_mode);
        if (assistantVisible) assistantVisible.checked = flags.assistant_visible !== false;
    }

    function renderEnvironmentTable(payload) {
        const container = document.getElementById("settingsEnvironmentTable");
        if (!container) return;
        const rows = Array.isArray(payload?.environment_variables) ? payload.environment_variables : [];
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
                            return `
                                <tr>
                                    <td>
                                        <div class="settings-key">${escapeHtml(row.key)}</div>
                                        <div class="helper-text">${escapeHtml(row.label || row.category || "")}</div>
                                    </td>
                                    <td>
                                        <input
                                            type="text"
                                            data-settings-env-key="${escapeHtml(row.key)}"
                                            data-original-value="${escapeHtml(savedValue)}"
                                            value="${escapeHtml(savedValue)}"
                                            placeholder="Use process/default value"
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
        window.addEventListener("amanaje:frontend-error", (event) => {
            renderFrontendDebug(event.detail);
        });
        loadSettings();
    });
})();
