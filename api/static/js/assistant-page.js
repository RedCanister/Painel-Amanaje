(function () {
    const state = {
        overview: null,
        currentDraft: null,
        currentRunId: null,
        currentReview: null,
        executionApproved: false,
    };

    function escapeHtml(value) {
        if (window.AmanajeUI?.escapeHtml) {
            return window.AmanajeUI.escapeHtml(value);
        }
        return String(value ?? "")
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#39;");
    }

    function pretty(value) {
        return JSON.stringify(value ?? {}, null, 2);
    }

    function showAlert(message, tone = "info") {
        const target = document.getElementById("assistantAlert");
        if (!target) return;
        target.textContent = message;
        target.className = `assistant-alert visible ${tone === "error" ? "error" : ""}`;
    }

    function clearAlert() {
        const target = document.getElementById("assistantAlert");
        if (!target) return;
        target.textContent = "";
        target.className = "assistant-alert";
    }

    function setOutput(id, value) {
        const target = document.getElementById(id);
        if (!target) return;
        target.textContent = typeof value === "string" ? value : pretty(value);
        window.AmanajeUI?.enhanceCollapsibles?.(document);
    }

    async function fetchJson(url, options = {}, settings = {}) {
        const response = await fetch(url, {
            headers: options.body ? { "Content-Type": "application/json", ...(options.headers || {}) } : (options.headers || {}),
            ...options,
        });
        let payload = {};
        try {
            payload = await response.json();
        } catch (error) {
            payload = { detail: `Unable to parse response JSON: ${error.message}` };
        }
        if (!response.ok && !settings.allowError) {
            throw new Error(payload.detail || payload.error || `HTTP ${response.status}`);
        }
        return payload;
    }

    function selectedModelId(selectId = "assistantModelSelect") {
        return String(document.getElementById(selectId)?.value || "").trim();
    }

    function assistantModelRequest(selectId) {
        return window.AmanajeUI?.getAssistantModelRequest?.(selectId) || { provider: "auto" };
    }

    function metric(label, value) {
        return `
            <div class="assistant-metric">
                <span>${escapeHtml(label)}</span>
                <strong>${escapeHtml(value ?? "-")}</strong>
            </div>
        `;
    }

    function renderKeyValueTable(entries) {
        const rows = Object.entries(entries || {});
        if (!rows.length) return '<div class="assistant-empty">No data available.</div>';
        return `
            <div class="assistant-table-wrapper">
                <table class="assistant-table">
                    <tbody>
                        ${rows.map(([key, value]) => `
                            <tr>
                                <th>${escapeHtml(key)}</th>
                                <td>${escapeHtml(typeof value === "object" ? JSON.stringify(value) : value)}</td>
                            </tr>
                        `).join("")}
                    </tbody>
                </table>
            </div>
        `;
    }

    function renderTable(rows, columns, emptyMessage = "No records available.") {
        const safeRows = Array.isArray(rows) ? rows : [];
        if (!safeRows.length) return `<div class="assistant-empty">${escapeHtml(emptyMessage)}</div>`;
        return `
            <table class="assistant-table">
                <thead>
                    <tr>${columns.map((column) => `<th>${escapeHtml(column.label)}</th>`).join("")}</tr>
                </thead>
                <tbody>
                    ${safeRows.map((row) => `
                        <tr>
                            ${columns.map((column) => {
                                const value = typeof column.value === "function" ? column.value(row) : row[column.value];
                                return `<td>${column.html ? value : escapeHtml(value ?? "-")}</td>`;
                            }).join("")}
                        </tr>
                    `).join("")}
                </tbody>
            </table>
        `;
    }

    function renderHeaderPills(overview) {
        const provider = overview?.provider_status || {};
        const active = provider.active_provider || provider.provider || "unknown";
        const activeProvider = provider.providers?.[active] || provider[active] || {};
        const dataset = overview?.assistant_dataset?.latest_dataset || {};
        const references = overview?.references?.repository || {};
        const target = document.getElementById("assistantHeaderPills");
        if (!target) return;
        target.innerHTML = `
            <span class="assistant-pill">Provider: ${escapeHtml(active)}</span>
            <span class="assistant-pill">Fallback: ${escapeHtml(activeProvider.fallback_active ?? activeProvider.fallback_provider ?? "n/a")}</span>
            <span class="assistant-pill">Model: ${escapeHtml(activeProvider.model_version || provider.model_version || "unversioned")}</span>
            <span class="assistant-pill">AssistantModels: ${escapeHtml(overview?.assistant_models?.count ?? 0)}</span>
            <span class="assistant-pill">Dataset: ${escapeHtml(dataset.dataset_hash ? "ready" : overview?.assistant_dataset?.status || "missing")}</span>
            <span class="assistant-pill">References: ${escapeHtml(references.stored_reference_count ?? 0)}</span>
        `;
    }

    function renderOverview(overview) {
        state.overview = overview;
        renderHeaderPills(overview);
        const provider = overview.provider_status || {};
        const active = provider.active_provider || "unknown";
        const dataset = overview.assistant_dataset?.latest_dataset || {};
        const refs = overview.references?.repository || {};
        const mlflow = overview.mlflow || {};

        document.getElementById("assistantOverviewMetrics").innerHTML = [
            metric("Active Provider", active),
            metric("Provider Count", Object.keys(provider.providers || {}).length),
            metric("Assistant Models", overview.assistant_models?.count ?? 0),
            metric("Dataset Records", dataset.record_count ?? 0),
            metric("Reference Count", refs.stored_reference_count ?? 0),
            metric("MLflow Status", mlflow.status || "unknown"),
        ].join("");

        setOutput("assistantProviderOutput", provider);
        document.getElementById("assistantContractsOutput").innerHTML = renderKeyValueTable(overview.alignment_contracts || {});
        renderModelTable(overview.assistant_models?.models || []);
        setOutput("assistantDatasetOutput", overview.assistant_dataset || {});
        setOutput("assistantReferenceOutput", {
            repository: overview.references?.repository || {},
            latest: overview.references?.latest || [],
        });
        renderRouteCatalog(overview.route_catalog || []);
        renderArtifactLocations(overview.artifact_locations || []);
        setOutput("assistantRawOverview", overview);
        renderEvalProviderOptions(provider);
    }

    function renderEvalProviderOptions(providerStatus) {
        const select = document.getElementById("assistantEvalProvider");
        if (!select) return;
        const current = select.value || "amanaje_slm";
        const providers = Object.keys(providerStatus.providers || {});
        if (!providers.length) return;
        select.innerHTML = providers.map((provider) => (
            `<option value="${escapeHtml(provider)}">${escapeHtml(provider)}</option>`
        )).join("") + '<option value="auto">auto</option>';
        select.value = providers.includes(current) || current === "auto" ? current : providers[0];
    }

    function renderModelTable(models) {
        const target = document.getElementById("assistantModelsOutput");
        if (!target) return;
        target.innerHTML = renderTable(
            models,
            [
                { label: "ID", value: "id" },
                { label: "Name", value: "name" },
                { label: "Version", value: (row) => row.model_version || row.version || row.assistant_config?.model_version },
                { label: "Provider", value: (row) => row.provider_config?.type || row.assistant_config?.runtime_type || row.provider },
                { label: "Dataset Hash", value: (row) => row.assistant_config?.training_dataset_hash || row.training_dataset_hash },
                { label: "Reference Data", value: "reference_data" },
            ],
            "No AssistantModel entries are registered yet."
        );
    }

    function renderRouteCatalog(routes) {
        const target = document.getElementById("assistantRouteCatalog");
        if (!target) return;
        target.innerHTML = renderTable(routes, [
            { label: "Group", value: "group" },
            { label: "Method", value: "method" },
            { label: "Path", value: "path" },
            { label: "Operation", value: "operation" },
        ]);
    }

    function renderArtifactLocations(locations) {
        const target = document.getElementById("assistantArtifactLocations");
        if (!target) return;
        target.innerHTML = renderTable(locations, [
            { label: "Key", value: "key" },
            { label: "Path", value: "path" },
            { label: "Exists", value: (row) => row.exists ? "yes" : "no" },
            { label: "Directory", value: (row) => row.is_dir ? "yes" : "no" },
        ]);
    }

    async function refreshOverview() {
        clearAlert();
        const overview = await fetchJson("/assistant/management/overview");
        renderOverview(overview);
        await refreshAssistantSelectors();
        showAlert("Assistant management overview refreshed.");
        return overview;
    }

    async function refreshAssistantSelectors() {
        await Promise.all([
            window.AmanajeUI?.loadAssistantModelOptions?.("assistantModelSelect"),
            window.AmanajeUI?.loadAssistantModelOptions?.("assistantDraftModelSelect"),
        ]);
    }

    async function testProvider() {
        setOutput("assistantProviderOutput", "Testing active provider...");
        const result = await fetchJson("/assistant/provider/test?provider=auto", { method: "POST" });
        setOutput("assistantProviderOutput", result);
        showAlert("Provider diagnostic completed.");
    }

    async function modelAction(action) {
        const id = selectedModelId();
        if (!id) {
            showAlert("Select an AssistantModel first.", "error");
            return;
        }

        const endpoints = {
            test: { url: `/assistant/models/${encodeURIComponent(id)}/test`, method: "POST" },
            status: { url: `/assistant/models/${encodeURIComponent(id)}/status`, method: "GET" },
            attach: { url: `/assistant/models/${encodeURIComponent(id)}/training/dataset/attach`, method: "POST" },
            curate: { url: `/assistant/models/${encodeURIComponent(id)}/training/curate`, method: "POST" },
        };
        const endpoint = endpoints[action];
        setOutput("assistantModelActionOutput", `Running ${action}...`);
        const result = await fetchJson(endpoint.url, { method: endpoint.method });
        setOutput("assistantModelActionOutput", result);
        if (action === "attach") {
            await refreshOverview();
        }
        showAlert(`AssistantModel ${action} completed.`);
    }

    async function rebuildDataset() {
        setOutput("assistantDatasetOutput", "Rebuilding assistant training dataset...");
        const result = await fetchJson("/assistant/datasets/rebuild", { method: "POST" });
        setOutput("assistantDatasetOutput", result);
        await refreshOverview();
        showAlert("Assistant training dataset rebuilt.");
    }

    async function loadLatestDataset() {
        const result = await fetchJson("/assistant/datasets/latest", {}, { allowError: true });
        setOutput("assistantDatasetOutput", result);
        showAlert(result.status === "error" ? "No assistant dataset has been built yet." : "Latest assistant dataset loaded.", result.status === "error" ? "error" : "info");
    }

    async function syncReferences() {
        const maxFiles = Number(document.getElementById("assistantReferenceMaxFiles")?.value || 100);
        setOutput("assistantReferenceOutput", "Syncing internal reference repository...");
        const result = await fetchJson(`/assistant/references/sync?max_files=${encodeURIComponent(Math.max(1, Math.min(maxFiles, 1000)))}`, { method: "POST" });
        setOutput("assistantReferenceOutput", result);
        await refreshOverview();
        showAlert("Internal reference repository sync completed.");
    }

    async function searchReferences() {
        const query = document.getElementById("assistantReferenceQuery")?.value || "";
        const result = await fetchJson("/assistant/references/search", {
            method: "POST",
            body: JSON.stringify({ query, limit: 20 }),
        });
        setOutput("assistantReferenceOutput", result);
        showAlert("Reference metadata search completed.");
    }

    async function draftWorkflow() {
        const prompt = document.getElementById("assistantDraftPrompt")?.value.trim();
        if (!prompt) {
            showAlert("Write an assistant request first.", "error");
            return;
        }
        const targetType = document.getElementById("assistantDraftTarget")?.value || "dataset_generation";
        setOutput("assistantDraftOutput", "Drafting workflow...");
        const result = await fetchJson("/assistant/draft", {
            method: "POST",
            body: JSON.stringify({
                prompt,
                target_type: targetType,
                ...assistantModelRequest("assistantDraftModelSelect"),
                context: {
                    source: "assistant_management",
                    management_test: true,
                },
                constraints: {
                    max_references: 8,
                },
            }),
        });
        state.currentDraft = result.draft || null;
        state.currentRunId = result.run_id || null;
        state.currentReview = result.review || null;
        setOutput("assistantDraftOutput", result);
        if (result.draft?.code) {
            document.getElementById("assistantExecutionCode").value = result.draft.code;
            document.getElementById("assistantExecutionProfile").value = result.draft.execution_profile || targetType;
            state.executionApproved = false;
        }
        showAlert("Assistant draft created and reviewed.");
    }

    async function reviewWorkflow() {
        if (!state.currentDraft) {
            showAlert("Create a draft first.", "error");
            return;
        }
        const result = await fetchJson("/assistant/review", {
            method: "POST",
            body: JSON.stringify({ draft: state.currentDraft }),
        });
        state.currentReview = result.review || null;
        setOutput("assistantDraftOutput", result);
        showAlert(result.approved ? "Draft review passed." : "Draft needs revision.", result.approved ? "info" : "error");
    }

    async function approveWorkflow() {
        if (!state.currentDraft) {
            showAlert("Create a draft first.", "error");
            return;
        }
        const result = await fetchJson("/assistant/approve", {
            method: "POST",
            body: JSON.stringify({
                draft: state.currentDraft,
                run_id: state.currentRunId,
                reviewer: "assistant-management",
                notes: "Approved from Assistant Management.",
            }),
        });
        state.currentRunId = result.run_id || state.currentRunId;
        setOutput("assistantDraftOutput", result);
        showAlert("Assistant draft approved.");
    }

    async function feedbackWorkflow() {
        if (!state.currentDraft) {
            showAlert("Create a draft first.", "error");
            return;
        }
        const result = await fetchJson("/assistant/feedback", {
            method: "POST",
            body: JSON.stringify({
                draft: state.currentDraft,
                run_id: state.currentRunId,
                label: "corrected",
                reason: "management_review",
                notes: "Feedback recorded from Assistant Management.",
                user_edits: {},
            }),
        });
        setOutput("assistantDraftOutput", result);
        showAlert("Assistant feedback recorded.");
    }

    async function submitWorkflow() {
        if (!state.currentRunId) {
            showAlert("Approve a draft before submitting.", "error");
            return;
        }
        const result = await fetchJson("/assistant/submit", {
            method: "POST",
            body: JSON.stringify({
                run_id: state.currentRunId,
                action: "prepare",
                payload: { source: "assistant_management" },
            }),
        });
        setOutput("assistantDraftOutput", result);
        showAlert("Approved assistant draft submitted for preparation.");
    }

    async function reviewExecutionCode() {
        const code = document.getElementById("assistantExecutionCode")?.value || "";
        const profile = document.getElementById("assistantExecutionProfile")?.value || "dataset_generation";
        if (!code.trim()) {
            showAlert("Paste or draft code before running a safety review.", "error");
            return;
        }
        const result = await fetchJson("/execution/review", {
            method: "POST",
            body: JSON.stringify({ code, profile }),
        }, { allowError: true });
        state.executionApproved = Boolean(result.approved);
        setOutput("assistantExecutionOutput", result);
        showAlert(result.approved ? "Code passed guarded execution review." : "Code needs revision before execution.", result.approved ? "info" : "error");
    }

    async function runExecutionCode() {
        const code = document.getElementById("assistantExecutionCode")?.value || "";
        const profile = document.getElementById("assistantExecutionProfile")?.value || "dataset_generation";
        if (!state.executionApproved) {
            showAlert("Review and approve the code before guarded execution.", "error");
            return;
        }
        const result = await fetchJson("/execution/run", {
            method: "POST",
            body: JSON.stringify({
                code,
                profile,
                draft_id: state.currentDraft?.draft_id || null,
                approved: true,
                context: { source: "assistant_management" },
            }),
        }, { allowError: true });
        setOutput("assistantExecutionOutput", result);
        showAlert(result.status === "success" ? "Guarded execution completed." : "Guarded execution returned an error.", result.status === "success" ? "info" : "error");
    }

    async function runEvals() {
        const provider = document.getElementById("assistantEvalProvider")?.value || "amanaje_slm";
        setOutput("assistantEvalOutput", `Running golden evals for ${provider}...`);
        const result = await fetchJson(`/assistant/evals/run?provider=${encodeURIComponent(provider)}`, { method: "POST" });
        setOutput("assistantEvalOutput", result);
        showAlert("Golden eval run completed.");
    }

    async function curateGlobalTraining() {
        setOutput("assistantEvalOutput", "Curating global assistant training examples...");
        const result = await fetchJson("/assistant/training/curate", { method: "POST" });
        setOutput("assistantEvalOutput", result);
        showAlert("Assistant training curation completed.");
    }

    function bindTabs() {
        document.querySelectorAll("[data-assistant-tab]").forEach((button) => {
            button.addEventListener("click", () => {
                const key = button.dataset.assistantTab;
                document.querySelectorAll("[data-assistant-tab]").forEach((item) => {
                    item.classList.toggle("active", item.dataset.assistantTab === key);
                });
                document.querySelectorAll(".assistant-tab-panel").forEach((panel) => {
                    panel.classList.toggle("active", panel.id === `assistantTab${key.charAt(0).toUpperCase()}${key.slice(1)}`);
                });
            });
        });
    }

    function bindActions() {
        document.getElementById("btnAssistantRefresh")?.addEventListener("click", () => runAction(refreshOverview));
        document.getElementById("btnAssistantProviderTest")?.addEventListener("click", () => runAction(testProvider));
        document.getElementById("btnAssistantModelTest")?.addEventListener("click", () => runAction(() => modelAction("test")));
        document.getElementById("btnAssistantModelStatus")?.addEventListener("click", () => runAction(() => modelAction("status")));
        document.getElementById("btnAssistantAttachDataset")?.addEventListener("click", () => runAction(() => modelAction("attach")));
        document.getElementById("btnAssistantCurateModel")?.addEventListener("click", () => runAction(() => modelAction("curate")));
        document.getElementById("btnAssistantDatasetRebuild")?.addEventListener("click", () => runAction(rebuildDataset));
        document.getElementById("btnAssistantDatasetLatest")?.addEventListener("click", () => runAction(loadLatestDataset));
        document.getElementById("btnAssistantReferenceSync")?.addEventListener("click", () => runAction(syncReferences));
        document.getElementById("btnAssistantReferenceSearch")?.addEventListener("click", () => runAction(searchReferences));
        document.getElementById("btnAssistantDraft")?.addEventListener("click", () => runAction(draftWorkflow));
        document.getElementById("btnAssistantReview")?.addEventListener("click", () => runAction(reviewWorkflow));
        document.getElementById("btnAssistantApprove")?.addEventListener("click", () => runAction(approveWorkflow));
        document.getElementById("btnAssistantFeedback")?.addEventListener("click", () => runAction(feedbackWorkflow));
        document.getElementById("btnAssistantSubmit")?.addEventListener("click", () => runAction(submitWorkflow));
        document.getElementById("btnAssistantExecutionReview")?.addEventListener("click", () => runAction(reviewExecutionCode));
        document.getElementById("btnAssistantExecutionRun")?.addEventListener("click", () => runAction(runExecutionCode));
        document.getElementById("btnAssistantEvalRun")?.addEventListener("click", () => runAction(runEvals));
        document.getElementById("btnAssistantTrainingCurate")?.addEventListener("click", () => runAction(curateGlobalTraining));
    }

    async function runAction(action) {
        try {
            clearAlert();
            await action();
        } catch (error) {
            console.error(error);
            showAlert(error.message || String(error), "error");
        }
    }

    document.addEventListener("DOMContentLoaded", () => {
        bindTabs();
        bindActions();
        runAction(refreshOverview);
    });
})();
