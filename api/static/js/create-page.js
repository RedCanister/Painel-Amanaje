function initializeEditor(template, elementId) {
    return new Promise(resolve => {
        const container = document.getElementById(elementId);
        const createFallbackEditor = () => {
            if (!container) {
                const memoryEditor = {
                    value: template || '',
                    getValue() { return this.value; },
                    setValue(value) { this.value = value || ''; },
                    focus() {}
                };
                window.editor = memoryEditor;
                resolve(memoryEditor);
                return;
            }
            container.innerHTML = '';
            const textarea = document.createElement('textarea');
            textarea.className = 'amanaje-editor-fallback';
            textarea.value = template || '';
            textarea.spellcheck = false;
            textarea.style.width = '100%';
            textarea.style.height = '100%';
            textarea.style.minHeight = '520px';
            textarea.style.border = '0';
            textarea.style.padding = '1rem';
            textarea.style.boxSizing = 'border-box';
            textarea.style.resize = 'vertical';
            textarea.style.background = '#111827';
            textarea.style.color = '#e5e7eb';
            textarea.style.fontFamily = 'Consolas, Monaco, monospace';
            textarea.style.fontSize = '0.95rem';
            textarea.style.lineHeight = '1.5';
            container.appendChild(textarea);
            const fallbackEditor = {
                getValue() { return textarea.value; },
                setValue(value) { textarea.value = value || ''; },
                focus() { textarea.focus(); },
                layout() {}
            };
            window.editor = fallbackEditor;
            resolve(fallbackEditor);
        };

        if (typeof require !== 'function') {
            createFallbackEditor();
            return;
        }
        require.config({ paths: { vs: 'https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.34.1/min/vs' } });
        require(['vs/editor/editor.main'], function () {
            if (!window.monaco || !container) {
                createFallbackEditor();
                return;
            }
            window.editor = monaco.editor.create(container, {
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
        }, createFallbackEditor);
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

function renderJsonCell(value, title = 'JSON') {
    if (value && typeof value === 'object') {
        return window.AmanajeUI?.renderJsonExplorer
            ? window.AmanajeUI.renderJsonExplorer(value, { title, maxPlotlyRows: 40, maxPlotlyColumns: 12 })
            : `<code data-auto-json="true">${escapeHtml(JSON.stringify(value))}</code>`;
    }
    return escapeHtml(String(value ?? ''));
}

function appendIfPresent(formData, key, value) {
    if (value === undefined || value === null || value === '') return;

    if (typeof value === 'object') {
        formData.append(key, JSON.stringify(value));
        return;
    }

    formData.append(key, String(value));
}

let datasetRegistryCache = [];
let modelRegistryCache = [];

function updateOperationToggleUI(operationId) {
    document.querySelectorAll('[data-operation-choice]').forEach((button) => {
        button.classList.toggle('active', button.dataset.operationChoice === operationId);
        button.setAttribute('aria-pressed', button.dataset.operationChoice === operationId ? 'true' : 'false');
    });
}

function setOperationChoice(operationId) {
    const select = document.getElementById('operationId');
    if (!select) return;
    if (select.value !== operationId) {
        select.value = operationId;
        select.dispatchEvent(new Event('change', { bubbles: true }));
    }
    updateOperationToggleUI(operationId);
}

function bindOperationToggle() {
    document.querySelectorAll('[data-operation-choice]').forEach((button) => {
        button.addEventListener('click', () => setOperationChoice(button.dataset.operationChoice));
    });
    updateOperationToggleUI(document.getElementById('operationId')?.value || 'datasets');
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

function buildAssistantModelFile(metadata) {
    const supportedPayloads = [
        { key: 'assistant_bundle_bytes', extension: '.zip', mime: 'application/zip' },
        { key: 'bundle_bytes', extension: '.zip', mime: 'application/zip' },
        { key: 'zip_bytes', extension: '.zip', mime: 'application/zip' }
    ];
    const artifact = supportedPayloads.find(({ key }) => typeof metadata[key] === 'string' && metadata[key].trim() !== '');
    if (!artifact) {
        throw new Error("Assistant Models require a .zip bundle file with assistant_model_manifest.json, tokenizer assets, model weights, and prompt template metadata.");
    }
    const bytes = decodeBase64Bytes(metadata[artifact.key]);
    const fileName = inferFilenameFromMetadata(metadata, 'assistant_model_bundle', artifact.extension);
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
            ${renderSupportMatrixSection('Assistant Models', result.assistant_models || {})}
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

function renderDatasetAnalysis(summary = {}, plots = []) {
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
            ${window.AmanajeUI?.renderPlotDeck ? AmanajeUI.renderPlotDeck(plots || []) : ''}
        </div>
    `;
}

function renderModelAnalysis(summary = {}, plots = []) {
    const manifest = summary.artifact_manifest || summary.artifact_metadata || {};
    const metrics = summary.metrics || {};
    const parameters = summary.parameters || {};
    const inputFeatures = Array.isArray(summary.input_features) ? summary.input_features : [];
    const outputFeatures = Array.isArray(summary.output_features) ? summary.output_features : [];
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
            <div class="analysis-kpis" style="margin-top:0.9rem;">
                <div class="analysis-kpi">
                    <span>Input Features</span>
                    <strong>${escapeHtml(inputFeatures.length || 0)}</strong>
                    <div class="small-text">${escapeHtml(inputFeatures.join(', ') || 'No input features registered')}</div>
                </div>
                <div class="analysis-kpi">
                    <span>Output Features</span>
                    <strong>${escapeHtml(outputFeatures.length || 0)}</strong>
                    <div class="small-text">${escapeHtml(outputFeatures.join(', ') || 'No output features registered')}</div>
                </div>
                <div class="analysis-kpi">
                    <span>Saved Parameters</span>
                    <strong>${escapeHtml(Object.keys(parameters).length || 0)}</strong>
                    <div class="small-text">${escapeHtml(summary.path || 'No artifact path registered')}</div>
                </div>
            </div>
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
            ${Object.keys(parameters).length ? `
                <table class="analysis-table">
                    <thead>
                        <tr>
                            <th>Parameter</th>
                            <th>Value</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${Object.entries(parameters).map(([key, value]) => `
                            <tr>
                                <td>${escapeHtml(key)}</td>
                                <td>${renderJsonCell(value, key)}</td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            ` : ''}
            ${window.AmanajeUI?.renderPlotDeck ? AmanajeUI.renderPlotDeck(plots || []) : ''}
        </div>
    `;
}

async function submitUpload(event) {
    event.preventDefault();

    const operationId = document.getElementById('operationId').value;
    const metadata = getUploadMetadata();
    const selectedFile = document.getElementById('artifactFile')?.files?.[0] || null;
    const selectedFileStem = selectedFile?.name ? selectedFile.name.replace(/\.[^.]+$/, '') : '';
    const name = metadata.name || metadata.objectName || metadata.object_name || selectedFileStem;

    if (!name) {
        setUploadStatus("Define 'name' in the editor or choose an artifact file before creating the object.", 'error');
        return;
    }

    if ((!window.editor || !window.editor.getValue().trim()) && !selectedFile) {
        setUploadStatus('Write a script in the editor or choose an artifact file before creating the object.', 'error');
        return;
    }

    let file;
    try {
        if (selectedFile) {
            file = selectedFile;
        } else if (operationId === 'models') {
            file = buildModelFile(metadata);
        } else if (operationId === 'assistant-models') {
            file = buildAssistantModelFile(metadata);
        } else {
            file = buildDatasetFile(metadata);
        }
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
        appendIfPresent(formData, 'modelType', operationId === 'assistant-models' ? 'assistant_model' : (metadata.model_type || metadata.type));
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
            : operationId === 'assistant-models'
                ? `Registered AssistantModel <code>${result.id || 'n/a'}</code> with bundle status: <code>${escapeHtml(result.bundle_status || 'unknown')}</code>.`
                : `Created model <code>${result.id || 'n/a'}</code> with runtime support: ${renderJsonCell(result.runtime_capabilities || {}, 'Runtime Support')}.`;
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
        datasetRegistryCache = Array.isArray(features) ? features : [];
        if (!datasetRegistryCache.length) {
            featureList.innerHTML = '<div class="feature-item">No datasets uploaded yet.</div>';
            return;
        }

        featureList.innerHTML = datasetRegistryCache.map(feature => `
            <div class="feature-item">
                <h4 style="margin:0.25rem 0;">${feature.name || 'Unnamed dataset'}</h4>
                <p style="margin:0.5rem 0; font-size:0.9em; color:#666;">${feature.description || 'No description provided.'}</p>
                <div style="display:flex; gap:1rem; font-size:0.9em; color:#666; margin-top:6px;">
                    <small>Type: <b>${feature.dataset_type || 'N/A'}</b></small>
                    <small>Size: <b>${Number(feature.size || 0).toFixed(2)} MB</b></small>
                    <small>ID: <b>${feature.id}</b></small>
                </div>
                <div class="editor-controls" style="margin:0.85rem 0 0;">
                    <button type="button" class="btn" data-dataset-analyze="${feature.id}">Analyze</button>
                    <button type="button" class="btn" data-dataset-extract="${feature.id}">Extract Features</button>
                </div>
            </div>
        `).join('');

        featureList.querySelectorAll('[data-dataset-analyze]').forEach((button) => {
            button.addEventListener('click', () => analyzeData(button.dataset.datasetAnalyze));
        });
        featureList.querySelectorAll('[data-dataset-extract]').forEach((button) => {
            button.addEventListener('click', () => extractFeatures(button.dataset.datasetExtract));
        });
    } catch (error) {
        console.error('Error fetching datasets:', error);
        featureList.innerHTML = '<div class="status-message error" style="display:block;">Failed to load dataset history.</div>';
    }
}

async function refreshModels() {
    const modelList = document.getElementById('modelList');

    try {
        const models = await createFetchJson('/list/model');
        modelRegistryCache = Array.isArray(models) ? models : [];
        if (!modelRegistryCache.length) {
            modelList.innerHTML = '<div class="model-item">No models uploaded yet.</div>';
            return;
        }

        modelList.innerHTML = modelRegistryCache.map(model => `
            <div class="model-item">
                <h4 style="margin:0.25rem 0;">${model.name || 'Unnamed model'}</h4>
                <p style="margin:0.5rem 0; font-size:0.9em; color:#666;">${model.description || 'No description provided.'}</p>
                <div style="display:flex; gap:1rem; font-size:0.9em; color:#666; margin-top:6px;">
                    <small>Type: <b>${model.model_type || 'N/A'}</b></small>
                    <small>Size: <b>${Number(model.size || 0).toFixed(2)} MB</b></small>
                    <small>ID: <b>${model.id}</b></small>
                </div>
                <div style="display:flex; gap:1rem; font-size:0.9em; color:#666; margin-top:6px;">
                    <small>Inputs: <b>${Array.isArray(model.input_features) ? model.input_features.length : 0}</b></small>
                    <small>Outputs: <b>${Array.isArray(model.output_features) ? model.output_features.length : 0}</b></small>
                    <small>Trained: <b>${model.is_trained ? 'yes' : 'no'}</b></small>
                </div>
                <div class="editor-controls" style="margin:0.85rem 0 0;">
                    <button type="button" class="btn" data-model-analyze="${model.id}">Analyze</button>
                </div>
            </div>
        `).join('');

        modelList.querySelectorAll('[data-model-analyze]').forEach((button) => {
            button.addEventListener('click', () => analyzeModel(button.dataset.modelAnalyze));
        });
    } catch (error) {
        console.error('Error fetching models:', error);
        modelList.innerHTML = '<div class="status-message error" style="display:block;">Failed to load model history.</div>';
    }
}

async function analyzeData(datasetIdOverride = null) {
    const datasetId = datasetIdOverride || document.getElementById('datasetSelect').value;
    const target = document.getElementById('datasetAnalysisResults');

    if (!datasetId) {
        alert('Please select a dataset first.');
        return;
    }

    try {
        target.innerHTML = '<div class="status-message info" style="display:block;">Analyzing dataset...</div>';
        document.getElementById('datasetSelect').value = String(datasetId);
        const results = await createFetchJson(`/analysis/object?registry_type=datasetmodel&item_id=${encodeURIComponent(datasetId)}`);
        target.innerHTML = renderDatasetAnalysis(results.summary || {}, results.plots || []);
    } catch (error) {
        console.error('Dataset analysis error:', error);
        target.innerHTML = `<div class="status-message error" style="display:block;">Analysis failed: ${error.message}</div>`;
    }
}

async function analyzeModel(modelIdOverride = null) {
    const modelId = modelIdOverride || document.getElementById('modelSelect').value;
    const target = document.getElementById('modelAnalysisResults');

    if (!modelId) {
        alert('Please select a model first.');
        return;
    }

    try {
        target.innerHTML = '<div class="status-message info" style="display:block;">Analyzing model...</div>';
        document.getElementById('modelSelect').value = String(modelId);
        const results = await createFetchJson(`/analysis/object?registry_type=learningmodel&item_id=${encodeURIComponent(modelId)}`);
        target.innerHTML = renderModelAnalysis(results.summary || {}, results.plots || []);
    } catch (error) {
        console.error('Model analysis error:', error);
        target.innerHTML = `<div class="status-message error" style="display:block;">Analysis failed: ${error.message}</div>`;
    }
}

async function extractFeatures(datasetIdOverride = null) {
    const datasetId = datasetIdOverride || document.getElementById('datasetSelect').value;
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
                                <td>${renderJsonCell(value, key.replace(/_/g, ' '))}</td>
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

function pickRandomReferences(references, count = 3) {
    const pool = [...references];
    const picks = [];
    while (pool.length && picks.length < count) {
        const index = Math.floor(Math.random() * pool.length);
        picks.push(pool.splice(index, 1)[0]);
    }
    return picks;
}

function buildReferenceSeedPrompt(references, operationId) {
    const modeLabel = operationId === 'assistant-models' ? 'assistant model bundle' : (operationId === 'models' ? 'model' : 'dataset');
    const lines = references.map((reference) => `- ${reference.name || reference.reference_id} [${reference.source_type}] ${reference.summary || ''}`.trim());
    return `Create a ${modeLabel} generation script inspired by this random project reference sample:\n${lines.join('\n')}\n\nKeep the output aligned with Painel Amanaje registry metadata.`;
}

async function generateScript() {
    const promptField = document.getElementById('assistantCreatePrompt');
    let userPrompt = (promptField?.value || '').trim();
    if (!userPrompt) {
        try {
            const referencesPayload = await createFetchJson('/assistant/references?limit=24');
            const sampledReferences = pickRandomReferences(referencesPayload.references || [], 3);
            if (sampledReferences.length) {
                userPrompt = buildReferenceSeedPrompt(sampledReferences, document.getElementById('operationId').value);
                showStatus('Seeded the prompt from a random assistant reference sample.', 'info');
            }
        } catch (error) {
            console.warn('Unable to seed create prompt from assistant references:', error);
        }
    }

    if (!userPrompt) {
        userPrompt = window.prompt(
            'Describe the script you want to generate:',
            'Example: Load a CSV file and calculate summary statistics.'
        );
    }

    if (!userPrompt || !userPrompt.trim()) {
        showStatus('No description provided.', 'info');
        return;
    }

    promptField.value = userPrompt.trim();
    await draftCreateAssistantScript();
}

function getCreateAssistantProfile() {
    return ['models', 'assistant-models'].includes(document.getElementById('operationId').value)
        ? 'model_generation'
        : 'dataset_generation';
}

function getCreateAssistantTargetType() {
    return ['models', 'assistant-models'].includes(document.getElementById('operationId').value)
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

    bindOperationToggle();
    ToggleFormOptions({
        triggerElement: '#operationId',
        mapping: {
            datasets: ['featuresSection'],
            models: ['modelsSection'],
            'assistant-models': ['modelsSection']
        },
        allFields: ['featuresSection', 'modelsSection']
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
        updateOperationToggleUI(document.getElementById('operationId').value);
        displayMetadataPanel(getUploadMetadata());
    });

    await window.AmanajeUI?.loadAssistantModelOptions?.('assistantCreateModel');
    document.getElementById('datasetSelect')?.addEventListener('focus', refreshSelectors, { once: true });
    document.getElementById('modelSelect')?.addEventListener('focus', refreshSelectors, { once: true });
    await loadSupportMatrix();
    loadFileList();
    displayMetadataPanel(getUploadMetadata());
});
