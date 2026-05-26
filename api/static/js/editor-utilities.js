let extractedVariables = {};
let markerState = [];

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

function dom(id) {
    return document.getElementById(id);
}

function selectedEditorRegistryContext() {
    const selectedValues = (selectId) => Array.from(dom(selectId)?.selectedOptions || [])
        .map((option) => Number(option.value))
        .filter((value) => Number.isFinite(value));
    return {
        dataset_ids: selectedValues('editorDatasetContext'),
        model_ids: selectedValues('editorModelContext')
    };
}

function populateRegistrySelect(selectId, items = []) {
    const select = dom(selectId);
    if (!select) return;
    select.innerHTML = items.map((item) => `
        <option value="${escapeHtml(item.id)}">${escapeHtml(item.name || `${item.id}`)}</option>
    `).join('');
}

async function loadEditorRegistryContext() {
    if (!dom('editorDatasetContext') && !dom('editorModelContext')) return;
    const payload = await editorFetchJson('/editor/context');
    populateRegistrySelect('editorDatasetContext', payload.datasets || []);
    populateRegistrySelect('editorModelContext', payload.models || []);
    const summary = dom('editorRegistryContextSummary');
    if (summary) {
        summary.textContent = `${(payload.datasets || []).length} datasets and ${(payload.models || []).length} models available to editor helpers.`;
    }
}

function insertTextAtCursor(text) {
    if (!window.editor) return;
    if (typeof window.editor.executeEdits === 'function' && typeof window.editor.getSelection === 'function') {
        const selection = window.editor.getSelection();
        window.editor.executeEdits('registry-context', [{ range: selection, text, forceMoveMarkers: true }]);
        window.editor.focus?.();
        refreshCreateWorkspaceInsights();
        return;
    }

    window.editor.setValue(`${window.editor.getValue() || ''}${text}`);
    window.editor.focus?.();
    refreshCreateWorkspaceInsights();
}

function insertRegistrySnippet() {
    const context = selectedEditorRegistryContext();
    const datasetId = context.dataset_ids[0] || '';
    const modelId = context.model_ids[0] || '';
    insertTextAtCursor(`

# Registry context helpers are injected by Painel Amanaje when this script runs.
print("Datasets:", list_datasets())
print("Models:", list_models())
df = load_dataset(${datasetId ? datasetId : ''})
model = load_model(${modelId ? modelId : ''}) if list_models() else None
print(df.head())
`);
}

function clearRegistryContextSelection() {
    ['editorDatasetContext', 'editorModelContext'].forEach((selectId) => {
        Array.from(dom(selectId)?.options || []).forEach((option) => { option.selected = false; });
    });
    const summary = dom('editorRegistryContextSummary');
    if (summary) summary.textContent = 'Registry context selection cleared.';
}

function setRuntimeState(label) {
    const target = dom('runtimeStateLabel');
    if (target) target.textContent = label;
}

function updateCursorStatus() {
    if (!window.editor || typeof window.editor.getPosition !== 'function') return;
    const position = window.editor.getPosition();
    if (!position) return;
    const target = dom('cursorPositionLabel');
    if (target) target.textContent = `Ln ${position.lineNumber}, Col ${position.column}`;
}

function updateDocumentStats() {
    const code = window.editor ? window.editor.getValue() : '';
    const lineCount = code ? code.split('\n').length : 0;
    const target = dom('documentStatsLabel');
    if (target) target.textContent = `${lineCount} lines | ${code.length} chars`;
}

function updateHeaderMetadata() {
    const metadata = extractMetadataFromEditor();
    const fileLabel = dom('editorFileLabel');
    const subtitle = dom('editorSubtitle');
    if (fileLabel) fileLabel.textContent = metadata.path || 'generated/create_workspace.py';
    if (subtitle) subtitle.textContent = metadata.description || 'Ready for dataset and model creation scripts.';
}

function buildOutlineEntries() {
    if (!window.editor) return [];
    const lines = window.editor.getValue().split('\n');
    const outline = [];

    lines.forEach((line, index) => {
        const functionMatch = line.match(/^def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(/);
        const classMatch = line.match(/^class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[:(]/);
        const assignmentMatch = line.match(/^([a-zA-Z_][a-zA-Z0-9_]*)\s*=/);

        if (functionMatch) outline.push({ name: functionMatch[1], detail: 'Function', line: index + 1 });
        else if (classMatch) outline.push({ name: classMatch[1], detail: 'Class', line: index + 1 });
        else if (assignmentMatch && !/^\s/.test(line)) outline.push({ name: assignmentMatch[1], detail: 'Assignment', line: index + 1 });
    });

    return outline.slice(0, 18);
}

function revealEditorLine(lineNumber) {
    if (!window.editor) return;
    if (typeof window.editor.revealLineInCenter === 'function') {
        window.editor.revealLineInCenter(lineNumber);
    }
    if (typeof window.editor.setPosition === 'function') {
        window.editor.setPosition({ lineNumber, column: 1 });
    }
    window.editor.focus?.();
}

function renderOutline() {
    const outlineList = dom('outlineList');
    if (!outlineList) return;
    const outline = buildOutlineEntries();

    if (!outline.length) {
        outlineList.innerHTML = '<div class="empty-state">Outline appears here.</div>';
        return;
    }

    outlineList.innerHTML = outline.map(item => `
        <button type="button" class="outline-item" data-line="${item.line}">
            <strong>${escapeHtml(item.name)}</strong>
            <span>${escapeHtml(item.detail)} - line ${item.line}</span>
        </button>
    `).join('');

    outlineList.querySelectorAll('[data-line]').forEach(button => {
        button.addEventListener('click', () => revealEditorLine(Number(button.getAttribute('data-line')) || 1));
    });
}

function renderProblems(problemEntries) {
    const problemsList = dom('problemsList');
    if (!problemsList) return;
    const issues = Array.isArray(problemEntries) ? problemEntries : [];
    markerState = issues;

    if (!issues.length) {
        problemsList.innerHTML = '<div class="empty-state">No diagnostics yet.</div>';
        return;
    }

    problemsList.innerHTML = issues.map((issue, index) => `
        <button type="button" class="problem-item ${escapeHtml(issue.level || 'info')}" data-problem-index="${index}">
            <strong>${escapeHtml(issue.title || 'Issue')}</strong>
            <span>${escapeHtml(issue.message || '')}</span>
        </button>
    `).join('');

    problemsList.querySelectorAll('[data-problem-index]').forEach(button => {
        button.addEventListener('click', () => {
            const issue = markerState[Number(button.getAttribute('data-problem-index')) || 0];
            if (issue) revealEditorLine(issue.line || 1);
        });
    });
}

function pushMonacoMarkers(problemEntries) {
    if (window.editor && window.monaco && typeof window.editor.getModel === 'function') {
        const severityMap = {
            error: monaco.MarkerSeverity.Error,
            warning: monaco.MarkerSeverity.Warning,
            info: monaco.MarkerSeverity.Info
        };
        const markers = (problemEntries || []).map(issue => ({
            startLineNumber: issue.line || 1,
            startColumn: issue.column || 1,
            endLineNumber: issue.endLine || issue.line || 1,
            endColumn: issue.endColumn || Math.max((issue.column || 1) + 1, 2),
            message: issue.message || 'Issue',
            severity: severityMap[issue.level || 'info'] || monaco.MarkerSeverity.Info
        }));
        monaco.editor.setModelMarkers(window.editor.getModel(), 'amanaje-create-editor', markers);
    }
    renderProblems(problemEntries);
}

function extractProblemsFromExecution(result) {
    const issues = [];
    const traceback = result?.error || result?.stderr || '';
    const lineMatch = traceback.match(/line (\d+)/);
    const lineNumber = lineMatch ? Number(lineMatch[1]) : 1;

    if (result?.error) {
        issues.push({
            level: 'error',
            title: 'Execution error',
            message: result.error.trim().split('\n').slice(-1)[0] || 'Execution error',
            line: lineNumber,
            column: 1
        });
    }

    return issues;
}

function refreshCreateWorkspaceInsights() {
    updateCursorStatus();
    updateDocumentStats();
    updateHeaderMetadata();
    renderOutline();
}

function focusPanel(panelId) {
    const panel = dom(panelId);
    if (panel) panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function openCommandPalette() {
    if (!window.editor) return;
    window.editor.focus?.();
    if (typeof window.editor.trigger === 'function') {
        window.editor.trigger('keyboard', 'editor.action.quickCommand', {});
    }
}

function toggleCollapse(targetId, toggleButton) {
    const body = dom(targetId);
    if (!body || !toggleButton) return;

    const expanded = toggleButton.getAttribute('aria-expanded') === 'true';
    body.hidden = expanded;
    toggleButton.setAttribute('aria-expanded', expanded ? 'false' : 'true');

    const icon = toggleButton.querySelector('.collapse-icon');
    if (icon) icon.textContent = expanded ? '+' : '-';
}

function wireCreateEditorShellControls() {
    dom('btnCommandPalette')?.addEventListener('click', openCommandPalette);
    dom('btnRefreshEditorContext')?.addEventListener('click', () => loadEditorRegistryContext().catch(error => showStatus(error.message, 'error')));
    dom('btnInsertRegistrySnippet')?.addEventListener('click', insertRegistrySnippet);
    dom('btnClearRegistryContext')?.addEventListener('click', clearRegistryContextSelection);
    dom('btnClearOutput')?.addEventListener('click', clearExecutionOutput);
    dom('btnClearProblems')?.addEventListener('click', () => pushMonacoMarkers([]));
    dom('btnSyncVariables')?.addEventListener('click', parseAndDisplayEditorState);
    dom('btnSyncMetadata')?.addEventListener('click', () => displayMetadataPanel(getUploadMetadata()));
    dom('btnFocusOutput')?.addEventListener('click', () => focusPanel('outputPanel'));
    dom('btnFocusVariables')?.addEventListener('click', () => focusPanel('variablesPanel'));
    dom('btnFocusMetadata')?.addEventListener('click', () => focusPanel('metadataPanel'));
    document.querySelectorAll('[data-collapse-target]').forEach(button => {
        button.addEventListener('click', () => toggleCollapse(button.getAttribute('data-collapse-target'), button));
    });
    refreshCreateWorkspaceInsights();
    pushMonacoMarkers([]);
    clearExecutionOutput();
    loadEditorRegistryContext().catch(error => showStatus(error.message, 'warning'));
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
        metadata,
        code: {
            script,
            language: 'python',
            metadata,
            line_count: script ? script.split('\n').length : 0
        }
    };
}

async function executeCode() {
    await window.editorReady;

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
        setRuntimeState('Queued');
        showStatus('Queueing code execution...', 'info');
        clearExecutionOutput();

        const accepted = await editorFetchJson('/execute/jobs', {
            method: 'POST',
            body: JSON.stringify({ code, registryContext: selectedEditorRegistryContext() })
        });
        window.AmanajeUI?.watchRun?.(accepted.run_id, { source: 'create-editor' });
        setRuntimeState('Running');
        const result = await waitForEditorExecutionRun(accepted.run_id);

        window.editorExecution = result || null;
        window.executedVariables = result.variables || {};
        window.editorDocument = result.document || buildEditorDocumentPayload(getUploadMetadata().name);
        window.editorMetadata = result.metadata || getUploadMetadata();
        extractedVariables = result.variables || {};

        displayExecutionOutput(
            result.stdout,
            result.stderr,
            result.status === 'success' ? 'Execution finished.' : 'Execution completed with errors.'
        );
        displayExtractedVariables(result.variables || {});
        displayMetadataPanel(window.editorMetadata);
        pushMonacoMarkers(extractProblemsFromExecution(result));

        if (result.variables && Object.keys(result.variables).length > 0) {
            setRuntimeState(result.error ? 'Execution failed' : 'Last run succeeded');
            showStatus(`Execution finished. ${Object.keys(result.variables || {}).length} variables are available.`, result.error ? 'error' : 'success');
        } else if (result.error) {
            setRuntimeState('Execution failed');
            showStatus(`Execution error: ${result.error}`, 'error');
        } else {
            setRuntimeState('Last run succeeded');
            showStatus('Code executed successfully.', 'success');
        }

        loadFileList();
        refreshCreateWorkspaceInsights();
    } catch (error) {
        console.error('Execution error:', error);
        setRuntimeState('Execution failed');
        pushMonacoMarkers([{ level: 'error', title: 'Request failed', message: error.message || String(error), line: 1, column: 1 }]);
        displayExecutionOutput('', String(error), 'Backend request failed.');
        showStatus(`Execution failed: ${error.message}`, 'error');
    }
}

async function waitForEditorExecutionRun(runId) {
    const startedAt = Date.now();
    while (Date.now() - startedAt < 30 * 60 * 1000) {
        const run = await editorFetchJson(`/runs/get/${encodeURIComponent(runId)}`);
        const latestEvent = run.progress?.latest_event || run.stage || run.status;
        showStatus(`Execution ${run.status || 'queued'}: ${latestEvent || runId}`, 'info');
        if (run.status === 'completed') {
            return run.result || {};
        }
        if (run.status === 'failed') {
            if (run.result && Object.keys(run.result).length) {
                return run.result;
            }
            throw new Error(run.error || 'The background execution failed.');
        }
        await new Promise(resolve => setTimeout(resolve, 1500));
    }
    throw new Error('The background execution did not finish within the polling window.');
}

function displayExecutionOutput(stdout, stderr, infoMessage = '') {
    const outputContainer = document.getElementById('executionOutput');
    if (!outputContainer) return;

    const chunks = [];
    if (infoMessage) chunks.push(`<div class="output-line"><span class="output-title">Status</span>${escapeHtml(infoMessage)}</div>`);
    if (stdout && stdout.trim()) chunks.push(`<div class="output-line"><span class="output-title">Stdout</span>${escapeHtml(stdout)}</div>`);
    if (stderr && stderr.trim()) chunks.push(`<div class="output-line error"><span class="output-title">Stderr</span>${escapeHtml(stderr)}</div>`);

    if (!chunks.length) {
        outputContainer.classList.add('empty');
        outputContainer.classList.remove('visible');
        outputContainer.textContent = 'Run the current script to inspect standard output and backend execution errors.';
        return;
    }

    outputContainer.classList.remove('empty');
    outputContainer.classList.add('visible');
    outputContainer.innerHTML = chunks.join('');
}

function clearExecutionOutput() {
    const outputContainer = document.getElementById('executionOutput');
    if (!outputContainer) return;

    outputContainer.textContent = 'Run the current script to inspect standard output and backend execution errors.';
    outputContainer.classList.add('empty');
    outputContainer.classList.remove('visible');
}

function isEscaped(text, index) {
    let backslashes = 0;
    for (let cursor = index - 1; cursor >= 0 && text[cursor] === '\\'; cursor -= 1) backslashes += 1;
    return backslashes % 2 === 1;
}

function countBracketDelta(line) {
    let delta = 0;
    let quote = null;

    for (let index = 0; index < line.length; index += 1) {
        const current = line[index];
        const nextThree = line.slice(index, index + 3);

        if (!quote && current === '#') break;
        if (!quote && (nextThree === '"""' || nextThree === "'''")) {
            quote = nextThree;
            index += 2;
            continue;
        }
        if (quote && nextThree === quote) {
            quote = null;
            index += 2;
            continue;
        }
        if (quote === '"' || quote === "'") {
            if (current === quote && !isEscaped(line, index)) quote = null;
            continue;
        }
        if (!quote && (current === '"' || current === "'")) {
            quote = current;
            continue;
        }
        if (quote) continue;
        if ('([{'.includes(current)) delta += 1;
        if (')]}'.includes(current)) delta -= 1;
    }

    return delta;
}

function parseVariableAssignmentsFromText() {
    if (!window.editor) return [];

    const code = window.editor.getValue();
    const lines = code.split('\n');
    const assignments = [];
    let currentAssignment = null;
    let bracketDepth = 0;

    lines.forEach((line, index) => {
        const trimmed = line.trim();

        if (!currentAssignment) {
            if (!trimmed || trimmed.startsWith('#')) return;
            if (/^\s+/.test(line)) return;

            const match = line.match(/^([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.+?)\s*$/);
            if (!match) return;

            currentAssignment = { name: match[1], expressionLines: [match[2]], line: index + 1 };
            bracketDepth = countBracketDelta(match[2]);

            if (bracketDepth <= 0 && !match[2].trim().endsWith('\\')) {
                assignments.push({
                    name: currentAssignment.name,
                    expression: currentAssignment.expressionLines.join('\n').trim(),
                    line: currentAssignment.line
                });
                currentAssignment = null;
                bracketDepth = 0;
            }
            return;
        }

        currentAssignment.expressionLines.push(line);
        bracketDepth += countBracketDelta(line);

        if (bracketDepth <= 0 && !trimmed.endsWith('\\')) {
            assignments.push({
                name: currentAssignment.name,
                expression: currentAssignment.expressionLines.join('\n').trim(),
                line: currentAssignment.line
            });
            currentAssignment = null;
            bracketDepth = 0;
        }
    });

    if (currentAssignment) {
        assignments.push({
            name: currentAssignment.name,
            expression: currentAssignment.expressionLines.join('\n').trim(),
            line: currentAssignment.line
        });
    }

    return assignments;
}

function buildVariablePayloadFromAssignment(assignment) {
    const expression = String(assignment.expression ?? '');
    const parsedValue = parsePythonLiteral(expression);
    const kind = inferVariableKind(parsedValue, expression);
    const preview = serializeVariableForDisplay(parsedValue, kind);

    return {
        name: assignment.name,
        expression,
        kind,
        type: kind === 'number' ? (Number.isInteger(parsedValue) ? 'int' : 'float') : Array.isArray(parsedValue) ? 'list' : parsedValue === null ? 'NoneType' : typeof parsedValue,
        line: assignment.line,
        value: preview,
        raw_value: parsedValue,
        preview,
        metadata: { source: 'editor', line: assignment.line, expression }
    };
}

function parseAndDisplayEditorState() {
    const assignments = parseVariableAssignmentsFromText();
    const metadata = extractMetadataFromEditor();
    const parsedVariables = assignments.reduce((collection, assignment) => {
        collection[assignment.name] = buildVariablePayloadFromAssignment(assignment);
        return collection;
    }, {});

    extractedVariables = parsedVariables;
    window.executedVariables = parsedVariables;
    window.editorMetadata = metadata || {};
    window.editorDocument = buildEditorDocumentPayload(metadata.name);
    displayExtractedVariables(parsedVariables);
    displayMetadataPanel(window.editorMetadata);
    pushMonacoMarkers([]);

    if (assignments.length > 0 || Object.keys(metadata).length > 0) {
        setRuntimeState('Parsed locally');
        showStatus(`Parsed ${assignments.length} assignments and ${Object.keys(metadata).length} metadata fields.`, 'success');
    } else {
        setRuntimeState('Idle');
        showStatus('No top-level variable assignments were found in the editor.', 'info');
    }

    refreshCreateWorkspaceInsights();

    return {
        assignments,
        metadata,
        parsedVariables
    };
}

function displayExtractedVariables(variables) {
    const panel = document.getElementById('variablePanel');
    const list = document.getElementById('variableList');
    if (!list) return;

    extractedVariables = variables || {};

    if (!variables || Object.keys(variables).length === 0) {
        list.innerHTML = '<div class="empty-state">Parse or run the script to collect variables.</div>';
        if (panel) panel.style.display = 'block';
        return;
    }

    list.innerHTML = Object.entries(variables).map(([name, info]) => `
        <article class="variable-card">
            <strong>${escapeHtml(name)}</strong>
            <div class="variable-meta">${escapeHtml(info.kind || info.type || 'value')} - line ${escapeHtml(String(info.metadata?.line || info.line || '-'))}</div>
            <p class="variable-preview">${escapeHtml(stringifyVariableValue(info.preview ?? info.value ?? info.raw_value ?? ''))}</p>
        </article>
    `).join('');

    if (panel) panel.style.display = 'block';
}

function stringifyVariableValue(value) {
    if (typeof value === 'string') return value;

    try {
        return JSON.stringify(value, null, 2);
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
        window.editorExecution = null;
        window.editorMetadata = data.metadata || {};
        window.editorDocument = {
            ...data,
            metadata: data.metadata || {},
            code: typeof data.code === 'object' && data.code !== null ? data.code : { script: codeText || '', language: 'python' }
        };
        displayExtractedVariables(data.variables || {});
        displayMetadataPanel(window.editorMetadata);
        pushMonacoMarkers([]);
        refreshCreateWorkspaceInsights();
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
        'onnx_bytes',
        'assistant_bundle_bytes',
        'bundle_bytes',
        'zip_bytes',
        'assistant_manifest'
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
        'assistant_bundle_bytes',
        'bundle_bytes',
        'zip_bytes',
        'assistant_manifest',
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
    parseVariableAssignmentsFromText().forEach((assignment) => {
        const varName = assignment.name;
        const varValue = String(assignment.expression ?? '').trim();
        const parsedValue = parsePythonLiteral(varValue);

        if (!varName || !varValue || !fieldsToExtract.includes(varName)) return;
        if (isMaterializedEditorField(varName) && parsedValue === varValue && !isQuotedPythonString(varValue)) {
            return;
        }
        metadata[varName] = parsedValue;
    });

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
        content.innerHTML = '<div class="empty-state">Define metadata variables in the script to preview a save-ready document.</div>';
    } else {
        const rows = Object.entries(metadata).map(([key, value]) => `
            <tr>
                <td class="metadata-key">${escapeHtml(key)}</td>
                <td class="metadata-value">${escapeHtml(stringifyVariableValue(value))}</td>
            </tr>
        `).join('');

        content.innerHTML = `
            <table class="metadata-table">
                <thead><tr><th>Field</th><th>Value</th></tr></thead>
                <tbody>${rows}</tbody>
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
    "feature_a": [1.0, 2.0, 3.0, 4.0],
    "feature_b": [10.0, 20.0, 30.0, 40.0],
    "segment": ["north", "south", "north", "west"],
    "target": [0, 1, 0, 1]
})

csv_text = df.to_csv(index=False)


# Name of the dataset.
name = "sample_dataset"

# Description of the dataset.
description = "Dataset created from the editor"

# Object type. To define the payload to be used in the dataset.
object_type = "dataset"

# Path of the dataset. Between, generated, model and dataset.
path = "generated/sample_dataset.csv"

# Version of the dataset. To be used in further updates on the dataset.
version = 1

# Dataset type. Between numerical, categorical, time series, image, videoaudio, text, mixed.
dataset_type = "mixed_dataset"

# Connection string. To be used in datasets that use live updates.
connection_string = ""

`;

    const modelTemplate = `# Model creation template
# Default path: create a scikit-learn artifact with joblib_bytes.
# Alternative PyTorch TorchScript example is included below.

import base64
import io
import joblib
from sklearn.ensemble import RandomForestRegressor


# Write the code for your learning model below:

model = RandomForestRegressor(n_estimators=50, random_state=42)
buffer = io.BytesIO()
joblib.dump(model, buffer)
buffer.seek(0)


# Name of the model.
name = "sample_model"

# Description of the model.
description = "Model created from the editor"

# Object type. To define the payload to be used in the model.
object_type = "learning_model"

# Path of the model. Between, generated, model and dataset.
path = "generated/sample_model.joblib"

# Version of the model. To be used in further updates on the model.
version = 1

# Model type. Between, supervised, un-supervised, generation, classification, regression, deep learning.
model_type = "supervised_model"

# Parameters. The parameters, to be hydrated automatically, on the training and study of the model.
parameters = {"framework": "sklearn", "estimator_class": "RandomForestRegressor"}

# Metrics. The metrics to be used in the training of the model.
metrics = {"task": "regression"}

# Reference data. The standard dataset to look at when training the model.
reference_data = "sample_dataset.csv"

# I/O Features. The standard features/columns to look at when selecting the input and output.
input_features = ["input"]
output_features = ["output"]

# Pre-training model conditions. To be updated whenever there is an operation performed on the dataset.
is_trained = False
is_tested = False
is_deployed = False

# Binary data. The binary used to write in or out the learning model object used in training.
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

    const assistantModelTemplate = `# AssistantModel upload template
# Use the Artifact File selector to choose a .zip bundle.
# The bundle must include assistant_model_manifest.json, tokenizer assets,
# PyTorch/Hugging Face weights, and a chat/prompt template.

name = "sample_assistant_model"
description = "PyTorch/Hugging Face assistant runtime served outside FastAPI"
object_type = "learning_model"
path = "runtime_artifacts/assistant_models/uploads/sample_assistant_model.zip"
version = 1
model_type = "assistant_model"
parameters = {
    "assistant": {
        "provider_type": "openai_compatible",
        "runtime_kind": "pytorch_hf_server",
        "base_url": "http://localhost:8080/v1",
        "model_name": "sample-assistant",
        "model_version": "0.1.0",
        "supported_draft_types": ["dataset_generation", "model_generation", "registry_object"],
        "max_context_tokens": 4096
    }
}
metrics = {"bundle_valid": False, "server_health": "not_checked"}
reference_data = "runtime_artifacts/assistant_datasets/interactions.jsonl"
input_features = ["prompt", "context_pack", "target_type"]
output_features = ["workflow_draft"]
is_trained = False
is_tested = False
is_deployed = False
`;

    if (!window.editor) {
        alert('Editor not initialized. Please refresh the page.');
        return;
    }

    window.editor.setValue(operationId === 'assistant-models' ? assistantModelTemplate : (operationId === 'models' ? modelTemplate : dataTemplate));
    window.executedVariables = {};
    extractedVariables = {};
    window.editorExecution = null;
    window.editorDocument = null;
    pushMonacoMarkers([]);
    clearExecutionOutput();
    displayExtractedVariables({});
    displayMetadataPanel(getUploadMetadata());
    setRuntimeState('Template loaded');
    showStatus('Template loaded. Update the metadata values before creating the object.', 'success');
    refreshCreateWorkspaceInsights();
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
        window.editorExecution = null;
        window.editorDocument = null;
        window.editorMetadata = {};
        extractedVariables = {};

        clearExecutionOutput();
        displayExtractedVariables({});
        displayMetadataPanel({});
        pushMonacoMarkers([]);
        setRuntimeState('Idle');
        showStatus('Editor cleared.', 'success');
        refreshCreateWorkspaceInsights();
    }
}
