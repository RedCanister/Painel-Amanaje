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

function buildDatasetFile(metadata) {
    if (typeof metadata.csv_text !== 'string' || metadata.csv_text.trim() === '') {
        throw new Error("Datasets require a materialized 'csv_text' value. Execute the editor code first or assign a literal CSV string.");
    }

    return new File([metadata.csv_text], `${metadata.name || 'dataset'}.csv`, { type: 'text/csv' });
}

function buildModelFile(metadata) {
    const payload = metadata.pytorch_bytes || metadata.torch_bytes || metadata.model_bytes;
    if (typeof payload !== 'string' || payload.trim() === '') {
        throw new Error("Models require a materialized base64 PyTorch payload such as 'pytorch_bytes'. Execute the editor code first or assign a literal payload.");
    }

    const binary = atob(payload);
    const bytes = Uint8Array.from(binary, char => char.charCodeAt(0));
    const path = typeof metadata.path === 'string' ? metadata.path.trim() : '';
    const pathFilename = path ? path.split(/[\\/]/).pop() : '';
    const fileName = pathFilename || `${metadata.name || 'model'}.pt`;
    return new File([bytes], fileName, { type: 'application/octet-stream' });
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

        setUploadStatus(`Created object with id <code>${result.id || 'n/a'}</code>.`, 'success');
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
        target.innerHTML = `
            <div class="analysis-result" style="grid-column: 1 / -1;">
                <h4>Analysis Results for Dataset #${datasetId}</h4>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr style="border-bottom: 2px solid #ddd; background: #f5f5f5;">
                        <th style="text-align: left; padding: 0.5rem;">Metric</th>
                        <th style="text-align: right; padding: 0.5rem;">Value</th>
                    </tr>
                    ${Object.entries(results.summary || {}).map(([key, value]) => `
                        <tr style="border-bottom: 1px solid #eee;">
                            <td style="padding: 0.5rem;">${key.replace(/_/g, ' ').toUpperCase()}</td>
                            <td style="text-align: right; padding: 0.5rem;"><b>${escapeHtml(typeof value === 'object' ? JSON.stringify(value) : String(value))}</b></td>
                        </tr>
                    `).join('')}
                </table>
            </div>
        `;
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
        target.innerHTML = `
            <div class="analysis-result" style="grid-column: 1 / -1;">
                <h4>Analysis Results for Model #${modelId}</h4>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr style="border-bottom: 2px solid #ddd; background: #f5f5f5;">
                        <th style="text-align: left; padding: 0.75rem;">Metric</th>
                        <th style="text-align: right; padding: 0.75rem;">Value</th>
                    </tr>
                    ${Object.entries(results.summary || results.metrics || {}).map(([key, value]) => `
                        <tr style="border-bottom: 1px solid #eee;">
                            <td style="padding: 0.75rem;">${key.replace(/_/g, ' ').toUpperCase()}</td>
                            <td style="text-align: right; padding: 0.75rem;"><b>${escapeHtml(typeof value === 'object' ? JSON.stringify(value) : String(value))}</b></td>
                        </tr>
                    `).join('')}
                </table>
            </div>
        `;
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
            <div class="analysis-result" style="grid-column: 1 / -1;">
                <h4>Feature Suggestions for Dataset #${datasetId}</h4>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr style="border-bottom: 2px solid #ddd; background: #f5f5f5;">
                        <th style="text-align: left; padding: 0.5rem;">Suggestion</th>
                        <th style="text-align: right; padding: 0.5rem;">Value</th>
                    </tr>
                    ${Object.entries(summary).map(([key, value]) => `
                        <tr style="border-bottom: 1px solid #eee;">
                            <td style="padding: 0.5rem;">${key.replace(/_/g, ' ').toUpperCase()}</td>
                            <td style="text-align: right; padding: 0.5rem;"><b>${escapeHtml(typeof value === 'object' ? JSON.stringify(value) : String(value))}</b></td>
                        </tr>
                    `).join('')}
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
    loadFileList();
    displayMetadataPanel(getUploadMetadata());
});
