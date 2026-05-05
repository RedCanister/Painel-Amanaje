(function () {
    const LINE_LIMIT = 12;
    const CHAR_LIMIT = 500;

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

    function renderCollapsibleJson(data, options = {}) {
        const title = options.title || "JSON";
        const preClass = options.preClass || "json-block";
        const jsonString = typeof data === "string" ? data : JSON.stringify(data ?? {}, null, 2);
        return renderCollapsibleContent(
            `<pre class="${preClass}">${escapeHtml(jsonString)}</pre>`,
            { title, collapsed: options.collapsed !== false, containerClass: options.containerClass || "" }
        );
    }
     // TODO - Be sure to extend the plotly implementation to this function
    function renderPlotDeck(plots = []) {
        const usablePlots = Array.isArray(plots) ? plots.filter((plot) => Array.isArray(plot.series) && plot.series.length) : [];
        if (!usablePlots.length) return '<div class="helper-text">No plots available for this result yet.</div>';
        return `
            <div class="plot-grid">
                ${usablePlots.map((plot) => {
                    const maxValue = Math.max(...plot.series.map((item) => Number(item.value) || 0), 1);
                    return `
                        <div class="card">
                            <h4>${escapeHtml(plot.title || "Generated Plot")}</h4>
                            ${plot.description ? `<div class="helper-text" style="margin-bottom:0.7rem;">${escapeHtml(plot.description)}</div>` : ""}
                            <div class="plot-bars">
                                ${plot.series.map((item) => {
                                    const value = Number(item.value) || 0;
                                    const width = Math.max((value / maxValue) * 100, value > 0 ? 6 : 0);
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
                        </div>
                    `;
                }).join("")}
            </div>
        `;
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
            if (!shouldCollapse(block.textContent || "")) return;
            const wrapper = document.createElement("div");
            wrapper.innerHTML = renderCollapsibleContent(block.outerHTML, {
                title: block.dataset.collapseTitle || `Details ${index + 1}`,
                collapsed: true,
            });
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
        const currentValue = element.value;
        const defaultLabel = options.defaultLabel || "Configured active provider";
        element.innerHTML = `<option value="">${escapeHtml(defaultLabel)}</option>`;
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

    window.AmanajeUI = {
        escapeHtml,
        isDebugModeEnabled,
        captureFrontendError,
        fetchJson,
        renderMetricCards,
        renderCollapsibleContent,
        renderCollapsibleJson,
        renderPlotDeck,
        renderResultTabs,
        activateTabs,
        enhanceCollapsibles,
        initAssistantCollapsibles,
        applyGlobalSettings,
        loadAssistantModelOptions,
        getAssistantModelRequest,
        setAssistantStatus,
    };

    document.addEventListener("DOMContentLoaded", () => {
        installFetchGuard();
        applyGlobalSettings(window.AmanajeSettings || {});
        activateTabs(document);
        enhanceCollapsibles(document);
        initAssistantCollapsibles(document);
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
