

async function executeCode() {
        if (!window.editor) {
            showStatus('⏳ Please wait, Monaco Editor is initializing...', 'error');
            return;
        }

        const code = window.editor.getValue();

        if (!code.trim()) {
            showStatus('⚠️ No code to execute', 'error');
            return;
        }

        try {
            showStatus('▶️ Executing code on backend...', 'info');
            clearExecutionOutput();
            
            const response = await fetch("/execute", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ code: code })
            });
            
            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`HTTP ${response.status}: ${errorText}`);
            }

            const result = await response.json();
           
            if (result.stdout || result.stderr) {
                displayExecutionOutput(result.stdout, result.stderr);
            }

            if (result.variables && Object.keys(result.variables).length > 0) {
                displayExtractedVariables(result.variables);
                showStatus('✅ Code executed successfully. Variables extracted.', 'success');
            } else if (result.error) {
                showStatus('❌ Execution error: ' + result.error, 'error');
            } else {
                showStatus('✅ Code executed (no variables defined)', 'success');
            }

            loadFileList();
        } catch (error) {
            console.error('Execution error:', error);
            showStatus('❌ Execution failed: ' + error.message, 'error');
        }
    }

function displayExecutionOutput(stdout, stderr) {
        const outputContainer = document.getElementById("executionOutput");
        const outputContent = document.getElementById("outputContent");

        outputContent.innerHTML = '';

        if (stdout && stdout.trim()) {
            const stdoutLine = document.createElement("div");
            stdoutLine.className = "output-line success";
            stdoutLine.textContent = "→ Output:\n" + stdout;
            outputContent.appendChild(stdoutLine);
        }

        if (stderr && stderr.trim()) {
            const stderrLine = document.createElement("div");
            stderrLine.className = "output-line error";
            stderrLine.textContent = "⚠ Errors:\n" + stderr;
            outputContent.appendChild(stderrLine);
        }

        outputContainer.classList.add("visible");
    }

function clearExecutionOutput() {
        const outputContainer = document.getElementById("executionOutput");
        const outputContent = document.getElementById("outputContent");
        outputContent.innerHTML = '';
        outputContainer.classList.remove("visible");
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

// TODO - This function needs to be split in two: The display an the formatting
// of the variables in 3 fields: Name, type and value. And have the variables
// placed inside a data structure
function displayExtractedVariables(variables) {
        const panel = document.getElementById("variablePanel");
        const list = document.getElementById("variableList");
        
        if (!list || !panel) return;

        list.innerHTML = '';
        extractedVariables = variables;

        if (!variables || Object.keys(variables).length === 0) {
            list.innerHTML = '<p style="color: #777;">No variables extracted</p>';
            panel.style.display = 'block';
            return;
        }

        Object.entries(variables).forEach(([name, info]) => {
            const item = document.createElement("div");
            item.className = 'variable-item';
            item.innerHTML = `
                <span class="var-name">${escapeHtml(name)}</span>
                <span class="var-type">${escapeHtml(info.type || '')}</span>
                <span class="var-value">${escapeHtml(info.value || '')}</span>
            `;
            list.appendChild(item);
        });

        panel.style.display = 'block';
        showStatus(`✅ Extracted ${Object.keys(variables).length} variables`, 'success');
    }

function escapeHtml(text) {
        const div = document.createElement("div");
        div.textContent = text;
        return div.innerHTML;
    }

async function saveVariablesToFile() {
        const filename = prompt("Enter filename to save variables:", "variables_" + new Date().toISOString().split('T')[0]);

        if (!filename) return;

        if (!extractedVariables || Object.keys(extractedVariables).length === 0) {
            showStatus('⚠️ No variables to save. Execute code first.', 'error');
            return;
        }

        // Ensure editor is ready and get its current content
        if (!window.editor) await window.editorReady;
        const script = (window.editor ? window.editor.getValue() : "");

        try {
            const saveData = {
                name: filename, 
                description: "uploaded code", // placeholder
                object_type: "code",
                size: 0.0, // placeholder
                path: "string", // placeholder
                date: new Date().toISOString(),
                version: 0,
                history: [{"uploaded": "now"}], // placeholder
                variables: extractedVariables,
                code: { script }
            };

            const reserved = new Set(['name', "description", "history"]);
            Object.entries(extractedVariables).forEach(([k, info]) => {
                const val = info && Object.prototype.hasOwnProperty.call(info, 'value') ? info.value : info;

                const targetKey = reserved.has(k) ? `var_${k}` : k;
                try {
                    if (val !== null && typeof val === 'object') {
                        saveData[targetKey] = JSON.parse(JSON.stringify(val));
                    } else {
                        saveData[targetKey] = val;
                    }
                } catch (e) {
                    saveData[targetKey] = String(val);
                }
            });

            console.log("Saving variables payload preview:", { filename, keys: Object.keys(saveData)});
            
            const response = await fetch('/codemodel/create', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(saveData)
            });

            console.log("Response preview:", { filename, keys: Object.keys(response)});
            
            if (!response.ok) {
                throw new Error (`HTTP ${response.status}: ${response.statusText}`);
            }

            const result = await response.json();
            showStatus(`✅ Variables saved as ${filename}`, 'success');
            loadFileList();
        } catch (error) {
            console.error('Save error:', error);
            showStatus('❌ Save error: ' + (error.stack || String(error)), 'error');
            // Some environments may have extra properties
            for (const key of Object.getOwnPropertyNames(error)) {
                console.log(`${key}:`, error[key]);
            }

        }
    }

async function loadFileList() {
        try {
            const response = await fetch('/codemodel/list');

            if(!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const files = await response.json();
            const fileList = document.getElementById("fileList");
            const filePanel = document.getElementById("filePanel");

            if (!fileList || !filePanel) return;

            fileList.innerHTML = '';

            if (!files || files.length === 0) {
                fileList.innerHTML = '<li>No saved files</li>';
                filePanel.style.display = 'block';
                return;
            }

            files.forEach(file => {
                const li = document.createElement("li");
                li.innerHTML = `
                <span>
                    📄 ${escapeHtml(file.name)}
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

            document.querySelectorAll(".load-btn").forEach(btn => {
                btn.addEventListener("click", async (e) => {
                    const fn = e.target.getAttribute('data-filename');
                    await loadVariableFileFromList(fn);
                });
            });

            document.querySelectorAll(".delete-btn").forEach(btn => {
                btn.addEventListener("click", async (e) => {
                    const fn = e.target.getAttribute('data-filename');
                    await deleteVariableFile(fn);
                });
            });

        } catch (error) {
            console.error('File list failed:', error);
        }
    }

async function loadVariableFileFromList(filename) {
        try {
            
            const response = await fetch(`/codemodel/get/${encodeURIComponent(filename)}`);

            console.log("Got response.")

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            console.log("Checked error.")

            const data = await response.json();

            console.log("Got data.")


            await window.editorReady;

            if (!window.editor) {
                throw new Error('Monaco editor failed to initialize. Reload the page.');
            }

            console.log(data.script)

            // Accept either `script` (legacy) or `code` (newer) payload shapes
            const codeCandidate = (data && (data.script ?? data.code)) || null;
            let codeText = null;
            if (codeCandidate) {
                if (typeof codeCandidate === 'string') {
                    codeText = codeCandidate;
                } else if (typeof codeCandidate === 'object' && codeCandidate !== null) {
                    // payload may be { script: '...' } or { code: { script: '...' } }
                    codeText = (codeCandidate.script ?? codeCandidate.text) || JSON.stringify(codeCandidate, null, 2);
                }
            }

            if (codeText) {
                try { window.editor.setValue(codeText); } catch (e) { console.warn('Failed to set editor value', e); }
            } else {
                console.warn('No code field in loaded file');
            }

            // TODO - This is where the error.stack finds the error.
            /*
            if (new_editor) {
                new_editor.setValue(data.code);
            }
            else {
                console.log("Can't retrieve code. Show: ", new_editor)
            }
            */
            
            console.log("Showed data.")

            displayExtractedVariables(data.variables || {});
            showStatus(`✅ Loaded ${filename}`, 'success');
        } catch (error) {
            const errorMsg = error.message || String(error);
            showStatus(`❌ Load failed: ${errorMsg}`, 'error');
            console.error('Load error details:', error);
        }
    }

async function deleteVariableFile(filename) {
        if (!confirm(`Delete ${filename}?`)) return;

        try {
            const response = await fetch(`/codemodel/delete/${encodeURIComponent(filename)}`, {
                method: 'DELETE'
            });

            if (!response.ok) {
                throw new Error('Failed to delete file');
            }

            showStatus(`✅ Deleted ${filename}`, 'success');
            loadFileList();
        } catch (error) {
            showStatus('❌ Delete failed: ' + error.message, 'error');
        }
    }

function exportVariablesAsJSON() {
        if (!extractedVariables || Object.keys(extractedVariables).length === 0) {
            showStatus('⚠️ No variables to export', 'error');
            return;
        }

        const exportData = {
            exportedAt: new Date().toISOString(),
            variables: extractedVariables,
            code: (window.editor ? window.editor.getValue() : "")
        };
        
        const blob = new Blob([JSON.stringify(exportData, null, 2)], {type: 'application/json'});
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `variables_${new Date().getTime()}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        showStatus('✅ Variables exported', 'success');
    }

function showStatus(message, type = 'success') {
        const statusEl = document.getElementById('statusMessage');
        if (!statusEl) return;
        statusEl.textContent = message;
        statusEl.className = `status-message ${type}`;
        statusEl.style.display = 'block';

        //setTimeout(() => {
        //    statusEl.style.display = 'none';
        //}, 5000);
    }

function clearEditor() {
        if (!window.editor) return;
        if (confirm('Clear editor content?')) {
            window.editor.setValue('');
            extractedVariables = {};
            const panel = document.getElementById('variablePanel');
            if (panel) panel.style.display = 'none';
            clearExecutionOutput();
            showStatus('✅ Editor cleared', 'success');
        }
    }

    