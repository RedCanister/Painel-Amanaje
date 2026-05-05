function initializeEditor(template, elementId) {
    return new Promise(resolve => {
        require.config({ paths: { vs: 'https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.34.1/min/vs' } });
        require(['vs/editor/editor.main'], function () {
            window.editor = monaco.editor.create(document.getElementById(elementId), {
                value: template || '',
                language: 'python',
                theme: 'vs-dark',
                automaticLayout: true,
                tabSize: 4,
                minimap: { enabled: false },
                lineNumbers: 'on',
                scrollBeyondLastLine: true,
                wordWrap: 'on'
            });

            resolve(window.editor);
        });
    });
}

function setUploadStatus(message, type) {
    const container = document.getElementById('uploadStatus');
    if (!container) return;

    container.innerHTML = `<div class="status-message ${type}" style="display:block;">${message}</div>`;
}

async function createFetchJson(url, options = {}) {
    if (window.AmanajeUI?.fetchJson) {
        return window.AmanajeUI.fetchJson(url, options, { source: 'create-page' });
    }
    const shouldSetJsonContentType = options.body && !(typeof FormData !== 'undefined' && options.body instanceof FormData);
    const response = await fetch(url, {
        headers: shouldSetJsonContentType ? { 'Content-Type': 'application/json', ...(options.headers || {}) } : (options.headers || {}),
        ...options
    });
    const text = await response.text();
    let payload = {};
    if (text) {
        try {
            payload = JSON.parse(text);
        } catch (_error) {
            payload = { detail: text };
        }
    }
    if (!response.ok || payload?.status === 'error') {
        throw new Error(payload?.detail || payload?.error || `HTTP ${response.status}`);
    }
    return payload;
}

function appendIfPresent(formData, key, value) {
    if (value === undefined || value === null || value === '') return;

    if (typeof value === 'object') {
        formData.append(key, JSON.stringify(value));
        return;
    }

    formData.append(key, String(value));
}

function decodeBase64Bytes(payload) {
    const binary = atob(payload);
    return Uint8Array.from(binary, char => char.charCodeAt(0));
}

function inferFilenameFromMetadata(metadata, fallbackName, fallbackExtension) {
    const path = typeof metadata.path === 'string' ? metadata.path.trim() : '';
    const pathFilename = path ? path.split(/[\\/]/).pop() : '';
    if (pathFilename) return pathFilename;
    return `${metadata.name || fallbackName}${fallbackExtension}`;
}

function buildDatasetFile(metadata) {
    if (typeof metadata.csv_text !== 'string' || metadata.csv_text.trim() === '') {
        throw new Error("Datasets require a materialized 'csv_text' value. Execute the editor code first or assign a literal CSV string.");
    }

    return new File([metadata.csv_text], `${metadata.name || 'dataset'}.csv`, { type: 'text/csv' });
}

function buildModelFile(metadata) {
    const supportedPayloads = [
        { key: 'joblib_bytes', extension: '.joblib', mime: 'application/octet-stream' },
        { key: 'torchscript_bytes', extension: '.pt', mime: 'application/octet-stream' },
        { key: 'pytorch_bytes', extension: '.pt', mime: 'application/octet-stream' },
        { key: 'torch_bytes', extension: '.pt', mime: 'application/octet-stream' },
        { key: 'model_bytes', extension: '.bin', mime: 'application/octet-stream' },
        { key: 'pickle_bytes', extension: '.pkl', mime: 'application/octet-stream' },
        { key: 'onnx_bytes', extension: '.onnx', mime: 'application/octet-stream' }
    ];

    const artifact = supportedPayloads.find(({ key }) => typeof metadata[key] === 'string' && metadata[key].trim() !== '');
    if (!artifact) {
        throw new Error(
            "Models require a materialized artifact payload. Supported editor fields are 'joblib_bytes', 'torchscript_bytes', 'pytorch_bytes', 'pickle_bytes', or 'onnx_bytes'."
        );
    }

    const bytes = decodeBase64Bytes(metadata[artifact.key]);
    const fileName = inferFilenameFromMetadata(metadata, 'model', artifact.extension);
    return new File([bytes], fileName, { type: artifact.mime });
}

function renderSupportMatrixSection(title, entries) {
    const cards = Object.entries(entries || {}).map(([extension, details]) => `
        <div class="support-card">
            <h4>${escapeHtml(`${details.label || 'Artifact'} ${extension}`)}</h4>
            <div class="small-text">${escapeHtml((details.notes || []).join(' ') || 'Capability-based support.')}</div>
            <div class="support-badges">
                ${['register', 'inspect', 'train', 'predict', 'simulate', 'monitor'].map((capability) => `
                    <span class="support-badge">${escapeHtml(`${capability}: ${details[capability]}`)}</span>
                `).join('')}
            </div>
        </div>
    `).join('');

    return `
        <div class="support-card" style="grid-column: 1 / -1;">
            <h4>${escapeHtml(title)}</h4>
            <div class="support-grid">${cards}</div>
        </div>
    `;
}

async function loadSupportMatrix() {
    const target = document.getElementById('supportMatrix');
    if (!target) return;

    try {
        const result = await createFetchJson('/upload/support');

        target.innerHTML = `
            ${renderSupportMatrixSection('Datasets', result.datasets || {})}
            ${renderSupportMatrixSection('Models', result.models || {})}
        `;
    } catch (error) {
        target.innerHTML = `<div class="status-message error" style="display:block;">Failed to load support matrix: ${escapeHtml(error.message)}</div>`;
    }
}

function renderWarnings(warnings = []) {
    if (!Array.isArray(warnings) || !warnings.length) return '';
    return warnings.slice(0, 6).map((warning) => `
        <div class="analysis-warning">${escapeHtml(warning)}</div>
    `).join('');
}

function renderDatasetAnalysis(summary = {}) {
    const explorer = Array.isArray(summary.column_explorer) ? summary.column_explorer.slice(0, 12) : [];
    return `
        <div class="analysis-card" style="grid-column: 1 / -1;">
            <h4>Column Explorer</h4>
            <div class="analysis-kpis">
                <div class="analysis-kpi"><span>Rows</span><strong>${escapeHtml(summary.rows ?? 0)}</strong></div>
                <div class="analysis-kpi"><span>Columns</span><strong>${escapeHtml(summary.columns ?? 0)}</strong></div>
                <div class="analysis-kpi"><span>Missing Values</span><strong>${escapeHtml(summary.missing_values ?? 0)}</strong></div>
                <div class="analysis-kpi"><span>Recommended Target</span><strong>${escapeHtml(summary.profile?.recommended_target || '-')}</strong></div>
            </div>
            ${renderWarnings(summary.top_warnings || summary.parser_report?.warnings || [])}
            <table class="analysis-table">
                <thead>
                    <tr>
                        <th>Column</th>
                        <th>Family</th>
                        <th>Nulls</th>
                        <th>Unique</th>
                        <th>Flags</th>
                        <th>Samples</th>
                    </tr>
                </thead>
                <tbody>
                    ${explorer.map((column) => `
                        <tr>
                            <td><strong>${escapeHtml(column.name)}</strong><br><span class="small-text">${escapeHtml(column.dtype)}</span></td>
                            <td>${escapeHtml(column.family)}</td>
                            <td>${escapeHtml(column.null_count)}</td>
                            <td>${escapeHtml(column.unique_count)}</td>
                            <td>${escapeHtml([
                                column.mixed_type ? 'mixed' : '',
                                column.high_cardinality ? 'high-cardinality' : '',
                                column.datetime_confidence >= 0.8 ? `date:${column.datetime_confidence}` : ''
                            ].filter(Boolean).join(', ') || 'stable')}</td>
                            <td>${escapeHtml((column.sample_values || []).join(', '))}</td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        </div>
    `;
}

function renderModelAnalysis(summary = {}) {
    const manifest = summary.artifact_manifest || summary.artifact_metadata || {};
    const metrics = summary.metrics || {};
    return `
        <div class="analysis-card" style="grid-column: 1 / -1;">
            <h4>Artifact Manifest</h4>
            <div class="analysis-kpis">
                <div class="analysis-kpi"><span>Format</span><strong>${escapeHtml(manifest.artifact_format || '-')}</strong></div>
                <div class="analysis-kpi"><span>Framework</span><strong>${escapeHtml(manifest.framework || summary.parameters?.framework || '-')}</strong></div>
                <div class="analysis-kpi"><span>Loader</span><strong>${escapeHtml(manifest.loader || '-')}</strong></div>
                <div class="analysis-kpi"><span>Metrics</span><strong>${escapeHtml(Object.keys(metrics).length)}</strong></div>
            </div>
            ${renderWarnings(manifest.warnings || [])}
            <table class="analysis-table">
                <thead>
                    <tr>
                        <th>Capability</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    ${Object.entries(manifest.runtime_capabilities || {}).map(([key, value]) => `
                        <tr>
                            <td>${escapeHtml(key)}</td>
                            <td>${escapeHtml(String(value))}</td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        </div>
    `;
}

async function submitUpload(event) {
    event.preventDefault();

    const operationId = document.getElementById('operationId').value;
    const metadata = getUploadMetadata();
    const name = metadata.name || metadata.objectName || metadata.object_name;

    if (!name) {
        setUploadStatus("Define 'name' in the editor before creating the object.", 'error');
        return;
    }

    if (!window.editor || !window.editor.getValue().trim()) {
        setUploadStatus('Write a script in the editor before creating the object.', 'error');
        return;
    }

    let file;
    try {
        file = operationId === 'models' ? buildModelFile(metadata) : buildDatasetFile(metadata);
    } catch (error) {
        setUploadStatus(error.message, 'error');
        return;
    }

    const formData = new FormData();
    formData.append('operationId', operationId);
    formData.append('objectName', name);
    formData.append('file', file, file.name);

    appendIfPresent(formData, 'description', metadata.description);
    appendIfPresent(formData, 'path', metadata.path);
    appendIfPresent(formData, 'version', metadata.version);
    appendIfPresent(formData, 'history', metadata.history);

    if (operationId === 'datasets') {
        appendIfPresent(formData, 'datasetType', metadata.dataset_type || metadata.type);
        appendIfPresent(formData, 'connectionString', metadata.connection_string);
    } else {
        appendIfPresent(formData, 'modelType', metadata.model_type || metadata.type);
        appendIfPresent(formData, 'parameters', metadata.parameters || {});
        appendIfPresent(formData, 'metrics', metadata.metrics || {});
        appendIfPresent(formData, 'referenceData', metadata.reference_data || metadata.referenceData);
        appendIfPresent(formData, 'inputFeatures', metadata.input_features || []);
        appendIfPresent(formData, 'outputFeatures', metadata.output_features || []);
        appendIfPresent(formData, 'isTrained', metadata.is_trained);
        appendIfPresent(formData, 'isTested', metadata.is_tested);
        appendIfPresent(formData, 'isDeployed', metadata.is_deployed);
    }

    try {
        setUploadStatus('Sending object to backend...', 'info');

        const result = await createFetchJson(`/upload/${operationId}`, {
            method: 'POST',
            body: formData
        });

        const detailMessage = operationId === 'datasets'
            ? `Created dataset <code>${result.id || 'n/a'}</code> with ${escapeHtml(result.shape?.[1] ?? 0)} columns.`
            : `Created model <code>${result.id || 'n/a'}</code> with runtime support: <code>${escapeHtml(JSON.stringify(result.runtime_capabilities || {}))}</code>.`;
        setUploadStatus(detailMessage, 'success');
        await refreshSelectors();
        await refreshFeatures();
        await refreshModels();
    } catch (error) {
        console.error('Upload error:', error);
        setUploadStatus(`Upload failed: ${error.message}`, 'error');
    }
}

async function refreshSelectors() {
    await PopulateListOptions({
        selectElement: '#datasetSelect',
        endpoint: '/features',
        labelField: 'name',
        valueField: 'id',
        placeholderText: 'Select a Dataset'
    });

    await PopulateListOptions({
        selectElement: '#modelSelect',
        endpoint: '/list/model',
        labelField: 'name',
        valueField: 'id',
        placeholderText: 'Select a Model'
    });
}

async function refreshFeatures() {
    const featureList = document.getElementById('featureList');

    try {
        const features = await createFetchJson('/features');
        if (!Array.isArray(features) || features.length === 0) {
            featureList.innerHTML = '<div class="feature-item">No datasets uploaded yet.</div>';
            return;
        }

        featureList.innerHTML = features.map(feature => `
            <div class="feature-item">
                <h4 style="margin:0.25rem 0;">${feature.name || 'Unnamed dataset'}</h4>
                <p style="margin:0.5rem 0; font-size:0.9em; color:#666;">${feature.description || 'No description provided.'}</p>
                <div style="display:flex; gap:1rem; font-size:0.9em; color:#666; margin-top:6px;">
                    <small>Type: <b>${feature.dataset_type || 'N/A'}</b></small>
                    <small>Size: <b>${Number(feature.size || 0).toFixed(2)} MB</b></small>
                    <small>ID: <b>${feature.id}</b></small>
                </div>
            </div>
        `).join('');
    } catch (error) {
        console.error('Error fetching datasets:', error);
        featureList.innerHTML = '<div class="status-message error" style="display:block;">Failed to load dataset history.</div>';
    }
}

async function refreshModels() {
    const modelList = document.getElementById('modelList');

    try {
        const models = await createFetchJson('/list/model');
        if (!Array.isArray(models) || models.length === 0) {
            modelList.innerHTML = '<div class="model-item">No models uploaded yet.</div>';
            return;
        }

        modelList.innerHTML = models.map(model => `
            <div class="model-item">
                <h4 style="margin:0.25rem 0;">${model.name || 'Unnamed model'}</h4>
                <p style="margin:0.5rem 0; font-size:0.9em; color:#666;">${model.description || 'No description provided.'}</p>
                <div style="display:flex; gap:1rem; font-size:0.9em; color:#666; margin-top:6px;">
                    <small>Type: <b>${model.model_type || 'N/A'}</b></small>
                    <small>Size: <b>${Number(model.size || 0).toFixed(2)} MB</b></small>
                    <small>ID: <b>${model.id}</b></small>
                </div>
            </div>
        `).join('');
    } catch (error) {
        console.error('Error fetching models:', error);
        modelList.innerHTML = '<div class="status-message error" style="display:block;">Failed to load model history.</div>';
    }
}

async function analyzeData() {
    const datasetId = document.getElementById('datasetSelect').value;
    const target = document.getElementById('datasetAnalysisResults');

    if (!datasetId) {
        alert('Please select a dataset first.');
        return;
    }

    try {
        target.innerHTML = '<div class="status-message info" style="display:block;">Analyzing dataset...</div>';

        const results = await createFetchJson(`/analysis/data?dataset_id=${encodeURIComponent(datasetId)}`);
        target.innerHTML = renderDatasetAnalysis(results.summary || {});
    } catch (error) {
        console.error('Dataset analysis error:', error);
        target.innerHTML = `<div class="status-message error" style="display:block;">Analysis failed: ${error.message}</div>`;
    }
}

async function analyzeModel() {
    const modelId = document.getElementById('modelSelect').value;
    const target = document.getElementById('modelAnalysisResults');

    if (!modelId) {
        alert('Please select a model first.');
        return;
    }

    try {
        target.innerHTML = '<div class="status-message info" style="display:block;">Analyzing model...</div>';

        const results = await createFetchJson(`/analysis/model?model_id=${encodeURIComponent(modelId)}`);
        target.innerHTML = renderModelAnalysis(results.summary || {});
    } catch (error) {
        console.error('Model analysis error:', error);
        target.innerHTML = `<div class="status-message error" style="display:block;">Analysis failed: ${error.message}</div>`;
    }
}

async function extractFeatures() {
    const datasetId = document.getElementById('datasetSelect').value;
    const target = document.getElementById('datasetAnalysisResults');

    if (!datasetId) {
        alert('Please select a dataset first.');
        return;
    }

    try {
        target.innerHTML = '<div class="status-message info" style="display:block;">Extracting feature candidates...</div>';
        const result = await createFetchJson(`/features/extract?dataset_id=${encodeURIComponent(datasetId)}`);
        const summary = result.summary || {};
        target.innerHTML = `
            ${renderDatasetAnalysis(result.manifest?.analysis || {})}
            <div class="analysis-card" style="grid-column: 1 / -1;">
                <h4>Feature Suggestions</h4>
                <table class="analysis-table">
                    <tbody>
                        ${Object.entries(summary).map(([key, value]) => `
                            <tr>
                                <th>${escapeHtml(key.replace(/_/g, ' '))}</th>
                                <td>${escapeHtml(typeof value === 'object' ? JSON.stringify(value) : String(value))}</td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
        `;
    } catch (error) {
        console.error('Feature extraction error:', error);
        target.innerHTML = `<div class="status-message error" style="display:block;">Feature extraction failed: ${error.message}</div>`;
    }
}

async function generateScript() {
    const userPrompt = window.prompt(
        'Describe the script you want to generate:',
        'Example: Load a CSV file and calculate summary statistics.'
    );

    if (!userPrompt || !userPrompt.trim()) {
        showStatus('No description provided.', 'info');
        return;
    }

    document.getElementById('assistantCreatePrompt').value = userPrompt.trim();
    await draftCreateAssistantScript();
}

function getCreateAssistantProfile() {
    return document.getElementById('operationId').value === 'models'
        ? 'model_generation'
        : 'dataset_generation';
}

function getCreateAssistantTargetType() {
    return document.getElementById('operationId').value === 'models'
        ? 'model_generation'
        : 'dataset_generation';
}

function renderCreateAssistantOutput(payload = {}) {
    const target = document.getElementById('assistantCreateOutput');
    if (!target) return;

    const draft = payload.draft || window.currentCreateAssistantDraft || {};
    const review = payload.review || window.currentCreateAssistantReview || {};
    const safety = payload.safety || review.safety || {};
    const execution = payload.execution || window.currentCreateAssistantExecution || null;
    const actions = review.required_actions || [];
    const messages = review.messages || [];
    const outputKeys = Object.keys(execution?.metadata || {});
    const statusLabel = payload.status || review.status || execution?.status || 'draft';
    const statusTone = actions.length ? 'needs_attention' : (execution?.status === 'success' || review.approved ? 'success' : 'idle');
    window.AmanajeUI?.setAssistantStatus?.('create', statusLabel, statusTone);

    target.innerHTML = `
        <h4>${escapeHtml(draft.title || 'Assistant Review')}</h4>
        <div class="assistant-pill-row">
            <span class="assistant-pill">${escapeHtml(statusLabel)}</span>
            <span class="assistant-pill">Profile: ${escapeHtml(safety.profile || getCreateAssistantProfile())}</span>
            <span class="assistant-pill">Risk: ${escapeHtml(safety.risk_level || 'none')}</span>
        </div>
        <p class="small-text">${escapeHtml(draft.summary || 'Assistant output will appear here.')}</p>
        ${messages.length ? `<div class="small-text">${messages.map((item) => escapeHtml(item)).join('<br>')}</div>` : ''}
        ${actions.length ? `<div class="status-message error" style="display:block;">${actions.map((item) => escapeHtml(item)).join('<br>')}</div>` : ''}
        ${outputKeys.length ? `<div class="small-text"><strong>Materialized:</strong> ${escapeHtml(outputKeys.join(', '))}</div>` : ''}
    `;
}

async function draftCreateAssistantScript() {
    const prompt = (document.getElementById('assistantCreatePrompt')?.value || '').trim();
    if (!prompt) {
        showStatus('Write an assistant request first.', 'error');
        return;
    }

    try {
        showStatus('Drafting assistant script...', 'info');
        const result = await createFetchJson('/assistant/draft', {
            method: 'POST',
            body: JSON.stringify({
                prompt,
                target_type: getCreateAssistantTargetType(),
                ...window.AmanajeUI?.getAssistantModelRequest?.('assistantCreateModel'),
                context: {
                    operationId: document.getElementById('operationId').value,
                    existing_metadata: getUploadMetadata()
                }
            })
        });

        await window.editorReady;

        window.currentCreateAssistantDraft = result.draft || null;
        window.currentCreateAssistantRunId = result.run_id || null;
        window.currentCreateAssistantReview = result.review || null;
        window.currentCreateAssistantExecution = null;

        if (window.editor && result.draft?.code) {
            window.editor.setValue(result.draft.code);
            window.executedVariables = {};
            displayMetadataPanel(getUploadMetadata());
            renderCreateAssistantOutput(result);
            showStatus('Assistant draft loaded into the editor.', result.review?.approved ? 'success' : 'info');
        } else {
            renderCreateAssistantOutput(result);
            showStatus('No script was generated.', 'error');
        }
    } catch (error) {
        console.error('Assistant draft error:', error);
        showStatus(`Assistant draft failed: ${error.message}`, 'error');
    }
}

async function reviewCreateAssistantScript() {
    if (!window.editor || !window.editor.getValue().trim()) {
        showStatus('No code to review.', 'error');
        return null;
    }

    try {
        showStatus('Reviewing assistant script...', 'info');
        const result = await createFetchJson('/execution/review', {
            method: 'POST',
            body: JSON.stringify({
                code: window.editor.getValue(),
                profile: getCreateAssistantProfile()
            })
        });
        window.currentCreateAssistantReview = {
            status: result.status,
            approved: result.approved,
            safety: result.safety,
            messages: result.approved ? ['Script passed assistant execution review.'] : [],
            required_actions: result.safety?.violations?.map((violation) => violation.message) || []
        };
        renderCreateAssistantOutput({
            status: result.status,
            draft: window.currentCreateAssistantDraft,
            review: window.currentCreateAssistantReview,
            safety: result.safety
        });
        showStatus(
            result.approved ? 'Assistant script review passed.' : 'Assistant script needs revision.',
            result.approved ? 'success' : 'error'
        );
        return result;
    } catch (error) {
        console.error('Assistant review error:', error);
        showStatus(`Assistant review failed: ${error.message}`, 'error');
        return null;
    }
}

async function runCreateAssistantScript() {
    const review = await reviewCreateAssistantScript();
    if (!review?.approved) return;

    try {
        showStatus('Running assistant script with guarded execution...', 'info');
        const result = await createFetchJson('/execution/run', {
            method: 'POST',
            body: JSON.stringify({
                code: window.editor.getValue(),
                profile: getCreateAssistantProfile(),
                draft_id: window.currentCreateAssistantDraft?.draft_id,
                approved: true,
                context: { operationId: document.getElementById('operationId').value }
            })
        });

        window.currentCreateAssistantExecution = result;
        window.executedVariables = result.variables || {};
        displayMetadataPanel(result.metadata || getUploadMetadata());
        displayExecutionOutput(result.stdout || '', result.stderr || '');
        renderCreateAssistantOutput({
            status: result.status,
            draft: window.currentCreateAssistantDraft,
            review: window.currentCreateAssistantReview,
            execution: result
        });
        showStatus('Assistant script ran successfully. Review the metadata, then create the object.', 'success');
    } catch (error) {
        console.error('Assistant execution error:', error);
        showStatus(`Assistant execution failed: ${error.message}`, 'error');
    }
}

function useCreateAssistantResult() {
    if (!window.currentCreateAssistantExecution) {
        showStatus('Run an assistant draft before using the result.', 'error');
        return;
    }
    window.executedVariables = window.currentCreateAssistantExecution.variables || {};
    displayMetadataPanel(window.currentCreateAssistantExecution.metadata || getUploadMetadata());
    showStatus('Assistant result is now the active create metadata.', 'success');
}

function waitForDependencies(timeout = 5000) {
    const startTime = Date.now();
    const requiredFunctions = [
        'executeCode',
        'parseAndDisplayMetadata',
        'loadMetadataTemplate',
        'loadFileList',
        'saveVariablesToFile',
        'extractMetadataFromEditor',
        'getUploadMetadata',
        'ToggleFormOptions',
        'PopulateListOptions',
        'exportVariablesAsJSON',
        'clearEditor'
    ];

    return new Promise(resolve => {
        const checkInterval = setInterval(() => {
            const allLoaded = requiredFunctions.every(fn => typeof window[fn] === 'function');
            if (allLoaded) {
                clearInterval(checkInterval);
                resolve();
            } else if (Date.now() - startTime > timeout) {
                clearInterval(checkInterval);
                resolve();
            }
        }, 50);
    });
}

document.addEventListener('DOMContentLoaded', async () => {
    await waitForDependencies();

    window.editor = null;
    window.executedVariables = {};
    window.editorReady = (async () => {
        window.editor = await initializeEditor('# Insert your creation code here', 'editorContainer');
    })();
    await window.editorReady;

    ToggleFormOptions({
        triggerElement: '#operationId',
        mapping: {
            datasets: ['featuresSection', 'datasetAnalysis'],
            models: ['modelsSection', 'modelAnalysis']
        },
        allFields: ['featuresSection', 'modelsSection', 'datasetAnalysis', 'modelAnalysis']
    });

    document.getElementById('uploadForm')?.addEventListener('submit', submitUpload);
    document.getElementById('btnCreateObject')?.addEventListener('click', () => {
        document.getElementById('uploadForm')?.dispatchEvent(new Event('submit', { cancelable: true }));
    });
    document.getElementById('btnExecuteScript')?.addEventListener('click', executeCode);
    document.getElementById('btnExtractVariables')?.addEventListener('click', () => {
        const result = parseAndDisplayEditorState();
        const assignments = result?.assignments || [];
        const metadata = result?.metadata || {};

        if (assignments.length > 0 || Object.keys(metadata).length > 0) {
            showStatus(
                `Found ${assignments.length} variable assignments and ${Object.keys(metadata).length} metadata fields.`,
                'success'
            );
        } else {
            showStatus('No variables found in the editor.', 'info');
        }
    });
    document.getElementById('btnSaveScript')?.addEventListener('click', saveVariablesToFile);
    document.getElementById('btnLoadScript')?.addEventListener('click', loadFileList);
    document.getElementById('btnLoadTemplate')?.addEventListener('click', () => {
        loadMetadataTemplate(document.getElementById('operationId').value);
    });
    document.getElementById('btnGenerateScript')?.addEventListener('click', generateScript);
    document.getElementById('btnAssistantDraftScript')?.addEventListener('click', draftCreateAssistantScript);
    document.getElementById('btnAssistantReviewScript')?.addEventListener('click', reviewCreateAssistantScript);
    document.getElementById('btnAssistantRunScript')?.addEventListener('click', runCreateAssistantScript);
    document.getElementById('btnAssistantUseResult')?.addEventListener('click', useCreateAssistantResult);
    document.getElementById('btnExportJSON')?.addEventListener('click', exportVariablesAsJSON);
    document.getElementById('btnResetScript')?.addEventListener('click', clearEditor);
    document.getElementById('btnRefreshFeatures')?.addEventListener('click', refreshFeatures);
    document.getElementById('btnExtractFeatures')?.addEventListener('click', extractFeatures);
    document.getElementById('btnRefreshModels')?.addEventListener('click', refreshModels);
    document.getElementById('btnOptimizeModels')?.addEventListener('click', () => {
        window.location.href = '/optimization';
    });
    document.getElementById('btnAnalyzeDataset')?.addEventListener('click', analyzeData);
    document.getElementById('btnAnalyzeModel')?.addEventListener('click', analyzeModel);
    document.getElementById('operationId')?.addEventListener('change', () => {
        displayMetadataPanel(getUploadMetadata());
    });

    await window.AmanajeUI?.loadAssistantModelOptions?.('assistantCreateModel');
    await refreshSelectors();
    await refreshFeatures();
    await refreshModels();
    await loadSupportMatrix();
    loadFileList();
    displayMetadataPanel(getUploadMetadata());
});
