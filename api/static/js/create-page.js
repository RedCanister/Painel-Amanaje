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
        const response = await fetch('/upload/support');
        const result = await response.json();
        if (!response.ok) throw new Error(result.detail || `HTTP ${response.status}`);

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

        const response = await fetch(`/upload/${operationId}`, {
            method: 'POST',
            body: formData
        });

        const result = await response.json();
        if (!response.ok) {
            throw new Error(result.detail || result.error || `HTTP ${response.status}`);
        }

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
        const response = await fetch('/features');
        if (!response.ok) throw new Error(`HTTP ${response.status}`);

        const features = await response.json();
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
        const response = await fetch('/list/model');
        if (!response.ok) throw new Error(`HTTP ${response.status}`);

        const models = await response.json();
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

        const response = await fetch(`/analysis/data?dataset_id=${encodeURIComponent(datasetId)}`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);

        const results = await response.json();
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

        const response = await fetch(`/analysis/model?model_id=${encodeURIComponent(modelId)}`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);

        const results = await response.json();
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
        const response = await fetch(`/features/extract?dataset_id=${encodeURIComponent(datasetId)}`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);

        const result = await response.json();
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

    try {
        showStatus('Generating script...', 'info');
        const response = await fetch('/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                prompt: userPrompt,
                operationId: document.getElementById('operationId').value
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const result = await response.json();
        await window.editorReady;

        if (window.editor && result.script) {
            window.editor.setValue(result.script);
            displayMetadataPanel(getUploadMetadata());
            showStatus('Script generated successfully.', 'success');
        } else {
            showStatus('No script was generated.', 'error');
        }
    } catch (error) {
        console.error('Generate error:', error);
        showStatus(`Generation failed: ${error.message}`, 'error');
    }
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
        const assignments = parseVariableAssignmentsFromText();
        const metadata = parseAndDisplayMetadata();

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

    await refreshSelectors();
    await refreshFeatures();
    await refreshModels();
    await loadSupportMatrix();
    loadFileList();
    displayMetadataPanel(getUploadMetadata());
});
