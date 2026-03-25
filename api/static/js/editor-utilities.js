let extractedVariables = {};

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

        const response = await fetch('/execute', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ code })
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`HTTP ${response.status}: ${errorText}`);
        }

        const result = await response.json();
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
            <span class="var-type">${escapeHtml(info.type || '')}</span>
            <span class="var-value">${escapeHtml(stringifyVariableValue(info.raw_value ?? info.value ?? ''))}</span>
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
        'Enter filename to save variables:',
        `variables_${new Date().toISOString().split('T')[0]}`
    );

    if (!filename) return;

    if (!extractedVariables || Object.keys(extractedVariables).length === 0) {
        showStatus('No variables to save. Execute code first.', 'error');
        return;
    }

    if (!window.editor) await window.editorReady;
    const script = window.editor ? window.editor.getValue() : '';

    try {
        const saveData = {
            name: filename,
            description: 'uploaded code',
            object_type: 'code',
            size: 0.0,
            path: 'editor',
            date: new Date().toISOString(),
            version: 0,
            history: [{ uploaded: 'now' }],
            variables: extractedVariables,
            code: { script }
        };

        const response = await fetch('/codemodel/create', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(saveData)
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        showStatus(`Variables saved as ${filename}.`, 'success');
        loadFileList();
    } catch (error) {
        console.error('Save error:', error);
        showStatus(`Save error: ${error.message || String(error)}`, 'error');
    }
}

async function loadFileList() {
    try {
        const response = await fetch('/codemodel/list');
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const files = await response.json();
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
        const response = await fetch(`/codemodel/get/${encodeURIComponent(filename)}`);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const data = await response.json();
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
        const response = await fetch(`/codemodel/delete/${encodeURIComponent(filename)}`, {
            method: 'DELETE'
        });

        if (!response.ok) {
            throw new Error('Failed to delete file.');
        }

        showStatus(`Deleted ${filename}.`, 'success');
        loadFileList();
    } catch (error) {
        showStatus(`Delete failed: ${error.message}`, 'error');
    }
}

function exportVariablesAsJSON() {
    if (!extractedVariables || Object.keys(extractedVariables).length === 0) {
        showStatus('No variables to export.', 'error');
        return;
    }

    const exportData = {
        exportedAt: new Date().toISOString(),
        variables: extractedVariables,
        code: window.editor ? window.editor.getValue() : ''
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

        if (!fieldsToExtract.includes(varName)) continue;
        metadata[varName] = parsePythonLiteral(varValue);
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
dataset_type = "csv"
connection_string = ""
`;

    const modelTemplate = `# Model creation template
# Paste a base64-encoded ONNX model into onnx_bytes before creating the object.

name = "sample_model"
description = "Model created from the editor"
object_type = "learning_model"
path = "generated/sample_model.onnx"
version = 1
model_type = "supervised"
parameters = {"framework": "onnx", "input_size": 1, "output_size": 1}
metrics = {"task": "regression"}
reference_data = "sample_dataset.csv"
input_features = ["input"]
output_features = ["output"]
is_trained = True
is_tested = False
is_deployed = False
onnx_bytes = ""
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
