let extractedVariables = {};

async function editorFetchJson(url, options = {}) {
    if (window.AmanajeUI?.fetchJson) {
        return window.AmanajeUI.fetchJson(url, options, { source: 'editor-utilities' });
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

function buildEditorDocumentPayload(filename) {
    const metadata = getUploadMetadata();
    const script = window.editor ? window.editor.getValue() : '';
    const normalizedName = filename || metadata.name || `script_${new Date().toISOString().split('T')[0]}`;
    const version = Number(metadata.version || 1);

    return {
        name: normalizedName,
        description: metadata.description || 'Saved from the Create workspace',
        object_type: 'code_model',
        size: 0.0,
        path: metadata.path || 'editor',
        date: new Date().toISOString(),
        version: Number.isFinite(version) ? version : 1,
        history: Array.isArray(metadata.history) ? metadata.history : [{ operation: 'save_script', when: new Date().toISOString() }],
        variables: extractedVariables || {},
        code: {
            script,
            language: 'python',
            metadata
        }
    };
}

async function executeCode() {
    if (!window.editor) {
        showStatus('Please wait for Monaco Editor to finish loading.', 'error');
        return;
    }

    const code = window.editor.getValue();
    if (!code.trim()) {
        showStatus('No code to execute.', 'error');
        return;
    }

    try {
        showStatus('Executing code on backend...', 'info');
        clearExecutionOutput();

        const result = await editorFetchJson('/execute', {
            method: 'POST',
            body: JSON.stringify({ code })
        });

        window.executedVariables = result.variables || {};

        if (result.stdout || result.stderr) {
            displayExecutionOutput(result.stdout, result.stderr);
        }

        if (result.variables && Object.keys(result.variables).length > 0) {
            displayExtractedVariables(result.variables);
            displayMetadataPanel(getUploadMetadata());
            showStatus('Code executed successfully. Variables extracted.', 'success');
        } else if (result.error) {
            showStatus(`Execution error: ${result.error}`, 'error');
        } else {
            displayMetadataPanel(getUploadMetadata());
            showStatus('Code executed successfully.', 'success');
        }

        loadFileList();
    } catch (error) {
        console.error('Execution error:', error);
        showStatus(`Execution failed: ${error.message}`, 'error');
    }
}

function displayExecutionOutput(stdout, stderr) {
    const outputContainer = document.getElementById('executionOutput');
    const outputContent = document.getElementById('outputContent');
    if (!outputContainer || !outputContent) return;

    outputContent.innerHTML = '';

    if (stdout && stdout.trim()) {
        const stdoutLine = document.createElement('div');
        stdoutLine.className = 'output-line success';
        stdoutLine.textContent = `Output:\n${stdout}`;
        outputContent.appendChild(stdoutLine);
    }

    if (stderr && stderr.trim()) {
        const stderrLine = document.createElement('div');
        stderrLine.className = 'output-line error';
        stderrLine.textContent = `Errors:\n${stderr}`;
        outputContent.appendChild(stderrLine);
    }

    outputContainer.classList.add('visible');
}

function clearExecutionOutput() {
    const outputContainer = document.getElementById('executionOutput');
    const outputContent = document.getElementById('outputContent');
    if (!outputContainer || !outputContent) return;

    outputContent.innerHTML = '';
    outputContainer.classList.remove('visible');
}

function parseVariableAssignmentsFromText() {
    if (!window.editor) return [];

    const code = window.editor.getValue();
    const assignments = [];
    const assignmentRegex = /^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.+?)(?=\n|$)/gm;
    let match;

    while ((match = assignmentRegex.exec(code)) !== null) {
        assignments.push({
            name: match[1],
            expression: match[2].trim(),
            line: code.substring(0, match.index).split('\n').length
        });
    }

    return assignments;
}

function buildVariablePayloadFromAssignment(assignment) {
    const parsedValue = parsePythonLiteral(assignment.expression);
    const kind = inferVariableKind(parsedValue, assignment.expression);
    const preview = serializeVariableForDisplay(parsedValue, kind);

    return {
        expression: assignment.expression,
        kind,
        type: kind,
        line: assignment.line,
        value: parsedValue,
        raw_value: parsedValue,
        preview
    };
}

function parseAndDisplayEditorState() {
    const assignments = parseVariableAssignmentsFromText();
    const metadata = extractMetadataFromEditor();
    const parsedVariables = assignments.reduce((collection, assignment) => {
        collection[assignment.name] = buildVariablePayloadFromAssignment(assignment);
        return collection;
    }, {});

    window.editorMetadata = metadata || {};
    displayExtractedVariables(parsedVariables);
    displayMetadataPanel(window.editorMetadata);

    return {
        assignments,
        metadata,
        parsedVariables
    };
}

function displayExtractedVariables(variables) {
    const panel = document.getElementById('variablePanel');
    const list = document.getElementById('variableList');
    if (!list || !panel) return;

    list.innerHTML = '';
    extractedVariables = variables || {};

    if (!variables || Object.keys(variables).length === 0) {
        list.innerHTML = '<p style="color: #777;">No variables extracted.</p>';
        panel.style.display = 'block';
        return;
    }

    Object.entries(variables).forEach(([name, info]) => {
        const item = document.createElement('div');
        item.className = 'variable-item';
        item.innerHTML = `
            <span class="var-name">${escapeHtml(name)}</span>
            <span class="var-type">${escapeHtml(info.kind || info.type || '')}</span>
            <span class="var-value">${escapeHtml(stringifyVariableValue(info.preview ?? info.raw_value ?? info.value ?? ''))}</span>
        `;
        list.appendChild(item);
    });

    panel.style.display = 'block';
}

function stringifyVariableValue(value) {
    if (typeof value === 'string') return value;

    try {
        return JSON.stringify(value);
    } catch (error) {
        return String(value);
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text ?? '';
    return div.innerHTML;
}

async function saveVariablesToFile() {
    const filename = prompt(
        'Enter filename to save the script:',
        `script_${new Date().toISOString().split('T')[0]}`
    );

    if (!filename) return;

    if (!window.editor) await window.editorReady;
    const script = window.editor ? window.editor.getValue() : '';
    if (!script.trim()) {
        showStatus('No script to save.', 'error');
        return;
    }

    try {
        await editorFetchJson('/codemodel/create', {
            method: 'POST',
            body: JSON.stringify(buildEditorDocumentPayload(filename))
        });

        showStatus(
            Object.keys(extractedVariables || {}).length
                ? `Script and extracted variables saved as ${filename}.`
                : `Script saved as ${filename}.`,
            'success'
        );
        loadFileList();
    } catch (error) {
        console.error('Save error:', error);
        showStatus(`Save error: ${error.message || String(error)}`, 'error');
    }
}

async function loadFileList() {
    try {
        const files = await editorFetchJson('/codemodel/list');
        const fileList = document.getElementById('fileList');
        const filePanel = document.getElementById('filePanel');
        if (!fileList || !filePanel) return;

        fileList.innerHTML = '';

        if (!files || files.length === 0) {
            fileList.innerHTML = '<li>No saved files.</li>';
            filePanel.style.display = 'block';
            return;
        }

        files.forEach(file => {
            const li = document.createElement('li');
            li.innerHTML = `
                <span>
                    ${escapeHtml(file.name)}
                    <small style="color: #999; margin-left: 10px;">${new Date(file.date).toLocaleString()}</small>
                </span>
                <div>
                    <button class="load-btn" data-filename="${escapeHtml(file.name)}">Load</button>
                    <button class="delete-btn" data-filename="${escapeHtml(file.name)}">Delete</button>
                </div>
            `;
            fileList.appendChild(li);
        });

        filePanel.style.display = 'block';

        document.querySelectorAll('.load-btn').forEach(btn => {
            btn.addEventListener('click', async event => {
                const filename = event.target.getAttribute('data-filename');
                await loadVariableFileFromList(filename);
            });
        });

        document.querySelectorAll('.delete-btn').forEach(btn => {
            btn.addEventListener('click', async event => {
                const filename = event.target.getAttribute('data-filename');
                await deleteVariableFile(filename);
            });
        });
    } catch (error) {
        console.error('File list failed:', error);
    }
}

async function loadVariableFileFromList(filename) {
    try {
        const data = await editorFetchJson(`/codemodel/get/${encodeURIComponent(filename)}`);
        await window.editorReady;

        if (!window.editor) {
            throw new Error('Monaco editor failed to initialize. Reload the page.');
        }

        const codeCandidate = (data && (data.script ?? data.code)) || null;
        let codeText = null;

        if (typeof codeCandidate === 'string') {
            codeText = codeCandidate;
        } else if (codeCandidate && typeof codeCandidate === 'object') {
            codeText = codeCandidate.script ?? codeCandidate.text ?? JSON.stringify(codeCandidate, null, 2);
        }

        if (codeText) {
            window.editor.setValue(codeText);
        }

        window.executedVariables = data.variables || {};
        displayExtractedVariables(data.variables || {});
        displayMetadataPanel(getUploadMetadata());
        showStatus(`Loaded ${filename}.`, 'success');
    } catch (error) {
        console.error('Load error details:', error);
        showStatus(`Load failed: ${error.message || String(error)}`, 'error');
    }
}

async function deleteVariableFile(filename) {
    if (!confirm(`Delete ${filename}?`)) return;

    try {
        await editorFetchJson(`/codemodel/delete/${encodeURIComponent(filename)}`, {
            method: 'DELETE'
        });

        showStatus(`Deleted ${filename}.`, 'success');
        loadFileList();
    } catch (error) {
        showStatus(`Delete failed: ${error.message}`, 'error');
    }
}

function exportVariablesAsJSON() {
    if (!window.editor || !window.editor.getValue().trim()) {
        showStatus('No script to export.', 'error');
        return;
    }

    const exportData = {
        exportedAt: new Date().toISOString(),
        document: buildEditorDocumentPayload(getUploadMetadata().name || `script_export_${Date.now()}`)
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = `variables_${Date.now()}.json`;
    document.body.appendChild(anchor);
    anchor.click();
    document.body.removeChild(anchor);
    URL.revokeObjectURL(url);

    showStatus('Variables exported.', 'success');
}

function parsePythonLiteral(rawValue) {
    const value = (rawValue || '').trim();
    if (!value) return '';

    if (
        (value.startsWith('"') && value.endsWith('"')) ||
        (value.startsWith("'") && value.endsWith("'"))
    ) {
        return value.slice(1, -1);
    }

    if (value === 'True') return true;
    if (value === 'False') return false;
    if (value === 'None') return null;

    if (/^-?\d+(\.\d+)?$/.test(value)) {
        return Number(value);
    }

    if (
        (value.startsWith('[') && value.endsWith(']')) ||
        (value.startsWith('{') && value.endsWith('}'))
    ) {
        try {
            const normalized = value
                .replace(/\bTrue\b/g, 'true')
                .replace(/\bFalse\b/g, 'false')
                .replace(/\bNone\b/g, 'null')
                .replace(/'/g, '"');
            return JSON.parse(normalized);
        } catch (error) {
            return value;
        }
    }

    return value;
}

function inferVariableKind(value, expression = '') {
    if (value === null) return 'null';
    if (Array.isArray(value)) return 'sequence';
    if (typeof value === 'boolean') return 'boolean';
    if (typeof value === 'number') return 'number';
    if (typeof value === 'string') {
        const trimmed = expression.trim();
        if (trimmed.startsWith('{') && trimmed.endsWith('}')) return 'mapping';
        return 'string';
    }
    if (typeof value === 'object') return 'mapping';
    return 'object';
}

function serializeVariableForDisplay(value, kind) {
    if (typeof value === 'string') return value;
    if (kind === 'mapping' || kind === 'sequence') {
        try {
            return JSON.stringify(value);
        } catch (error) {
            return String(value);
        }
    }
    return String(value);
}

function isMaterializedEditorField(fieldName) {
    return [
        'csv_text',
        'pytorch_bytes',
        'torch_bytes',
        'model_bytes',
        'torchscript_bytes',
        'joblib_bytes',
        'pickle_bytes',
        'onnx_bytes'
    ].includes(fieldName);
}

function isQuotedPythonString(rawValue) {
    const value = (rawValue || '').trim();
    return (
        (value.startsWith('"') && value.endsWith('"')) ||
        (value.startsWith("'") && value.endsWith("'"))
    );
}

function extractMetadataFromEditor(allowedFields = null) {
    if (!window.editor) return {};

    const code = window.editor.getValue();
    const metadata = {};
    const defaultFields = [
        'name',
        'description',
        'object_type',
        'dataset_type',
        'connection_string',
        'model_type',
        'type',
        'objectName',
        'object_name',
        'path',
        'date',
        'version',
        'history',
        'csv_text',
        'pytorch_bytes',
        'torch_bytes',
        'model_bytes',
        'torchscript_bytes',
        'joblib_bytes',
        'pickle_bytes',
        'onnx_bytes',
        'parameters',
        'metrics',
        'reference_data',
        'referenceData',
        'input_features',
        'output_features',
        'is_trained',
        'is_tested',
        'is_deployed'
    ];

    const fieldsToExtract = allowedFields || defaultFields;
    const assignmentRegex = /^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.+?)(?=\n|$)/gm;
    let match;

    while ((match = assignmentRegex.exec(code)) !== null) {
        const varName = match[1];
        const varValue = match[2].trim();
        const parsedValue = parsePythonLiteral(varValue);

        if (!fieldsToExtract.includes(varName)) continue;
        if (isMaterializedEditorField(varName) && parsedValue === varValue && !isQuotedPythonString(varValue)) {
            continue;
        }
        metadata[varName] = parsedValue;
    }

    return metadata;
}

function getExecutedMetadata() {
    const metadata = {};
    const variables = window.executedVariables || {};

    Object.entries(variables).forEach(([key, info]) => {
        if (info && Object.prototype.hasOwnProperty.call(info, 'raw_value')) {
            metadata[key] = info.raw_value;
        }
    });

    return metadata;
}

function getUploadMetadata() {
    return {
        ...extractMetadataFromEditor(),
        ...getExecutedMetadata()
    };
}

function displayMetadataPanel(metadata) {
    const container = document.getElementById('metadataPanel');
    const content = document.getElementById('metadataContent');
    const hint = document.getElementById('metadataHint');

    if (!container || !content) return;

    if (hint) {
        hint.style.display = metadata && Object.keys(metadata).length > 0 ? 'none' : 'block';
    }

    if (!metadata || Object.keys(metadata).length === 0) {
        content.innerHTML = '<p style="color: #999;">No metadata variables defined in the editor.</p>';
    } else {
        const rows = Object.entries(metadata).map(([key, value]) => `
            <tr>
                <td class="metadata-key">${escapeHtml(key)}</td>
                <td class="metadata-value">${escapeHtml(stringifyVariableValue(value))}</td>
            </tr>
        `).join('');

        content.innerHTML = `
            <table class="metadata-table">
                <tr><th>Field</th><th>Value</th></tr>
                ${rows}
            </table>
        `;
    }

    container.style.display = 'block';
    window.editorMetadata = metadata || {};
}

function loadMetadataTemplate(operationId) {
    const dataTemplate = `# Data creation template
import pandas as pd

df = pd.DataFrame({
    "value": [10, 20, 30],
    "category": ["a", "b", "c"]
})

csv_text = df.to_csv(index=False)

name = "sample_dataset"
description = "Dataset created from the editor"
object_type = "dataset"
path = "generated/sample_dataset.csv"
version = 1
dataset_type = "dataset"
connection_string = ""
`;

    const modelTemplate = `# Model creation template
# Default path: create a scikit-learn artifact with joblib_bytes.
# Alternative PyTorch TorchScript example is included below.

import base64
import io
import joblib
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=50, random_state=42)
buffer = io.BytesIO()
joblib.dump(model, buffer)
buffer.seek(0)

name = "sample_model"
description = "Model created from the editor"
object_type = "learning_model"
path = "generated/sample_model.joblib"
version = 1
model_type = "supervised_model"
parameters = {"framework": "sklearn", "estimator_class": "RandomForestRegressor"}
metrics = {"task": "regression"}
reference_data = "sample_dataset.csv"
input_features = ["input"]
output_features = ["output"]
is_trained = False
is_tested = False
is_deployed = False
joblib_bytes = base64.b64encode(buffer.read()).decode("utf-8")

# PyTorch TorchScript alternative:
# import torch
# import torch.nn as nn
# class MyModel(nn.Module):
#     def __init__(self, input_size=1, hidden_size=8, output_size=1):
#         super().__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(hidden_size, output_size)
#     def forward(self, x):
#         return self.fc2(self.relu(self.fc1(x)))
# model = MyModel()
# scripted = torch.jit.script(model)
# torch_buffer = io.BytesIO()
# torch.jit.save(scripted, torch_buffer)
# torch_buffer.seek(0)
# path = "generated/sample_model.pt"
# parameters = {"framework": "pytorch", "input_size": 1, "output_size": 1}
# torchscript_bytes = base64.b64encode(torch_buffer.read()).decode("utf-8")
`;

    if (!window.editor) {
        alert('Editor not initialized. Please refresh the page.');
        return;
    }

    window.editor.setValue(operationId === 'models' ? modelTemplate : dataTemplate);
    window.executedVariables = {};
    displayMetadataPanel(getUploadMetadata());
    showStatus('Template loaded. Update the metadata values before creating the object.', 'success');
}

function parseAndDisplayMetadata() {
    const metadata = getUploadMetadata();
    displayMetadataPanel(metadata);

    if (Object.keys(metadata).length > 0) {
        showStatus(`Found ${Object.keys(metadata).length} metadata fields.`, 'success');
    } else {
        showStatus('No metadata variables found. Define values such as name = "my_object".', 'info');
    }

    return metadata;
}

function showStatus(message, type = 'success') {
    const statusEl = document.getElementById('statusMessage');
    if (!statusEl) return;

    statusEl.textContent = message;
    statusEl.className = `status-message ${type}`;
    statusEl.style.display = 'block';
}

function clearEditor() {
    if (!window.editor) return;

    if (confirm('Clear editor content?')) {
        window.editor.setValue('');
        window.executedVariables = {};
        extractedVariables = {};

        const variablePanel = document.getElementById('variablePanel');
        if (variablePanel) variablePanel.style.display = 'none';

        const metadataPanel = document.getElementById('metadataPanel');
        if (metadataPanel) metadataPanel.style.display = 'none';

        clearExecutionOutput();
        showStatus('Editor cleared.', 'success');
    }
}
