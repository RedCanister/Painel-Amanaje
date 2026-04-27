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

    window.AmanajeUI = {
        escapeHtml,
        renderMetricCards,
        renderCollapsibleContent,
        renderCollapsibleJson,
        renderPlotDeck,
        renderResultTabs,
        activateTabs,
        enhanceCollapsibles,
    };

    document.addEventListener("DOMContentLoaded", () => {
        activateTabs(document);
        enhanceCollapsibles(document);
        const observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                mutation.addedNodes.forEach((node) => {
                    if (!(node instanceof HTMLElement)) return;
                    activateTabs(node);
                    enhanceCollapsibles(node);
                });
            });
        });
        observer.observe(document.body, { childList: true, subtree: true });
    });
})();
