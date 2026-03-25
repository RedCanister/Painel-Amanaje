/**
 * main.js - Global utilities and helpers for Painel Amanajé
 * 
 * Provides:
 * - HTML escaping utilities
 * - Status message display
 * - Common helper functions
 * - Global variables initialization
 */

/**
 * Escape HTML entities to prevent XSS attacks
 * @param {string} text - Text to escape
 * @returns {string} - Escaped HTML
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/**
 * Display status messages to the user
 * @param {string} message - Message to display
 * @param {string} type - Message type: 'success', 'error', 'info', 'warning'
 */
function showStatus(message, type = 'info') {
    const statusEl = document.getElementById('statusMessage');
    if (!statusEl) {
        console.warn('Status element #statusMessage not found');
        // Also log to console as fallback
        console.log(`[${type}] ${message}`);
        return;
    }
    
    statusEl.textContent = message;
    statusEl.className = `statusMessage ${type}`;
    
    // Optional: Auto-hide success/info messages after 5 seconds
    if (['success', 'info'].includes(type) && statusEl.classList.contains('statusMessage')) {
        setTimeout(() => {
            statusEl.textContent = '';
            statusEl.className = 'statusMessage';
        }, 5000);
    }
}

/**
 * Clear execution output display
 */
function clearExecutionOutput() {
    const outputContent = document.getElementById('outputContent');
    if (outputContent) {
        outputContent.innerHTML = '';
    }
}

/**
 * Display execution output from code execution
 * @param {string} stdout - Standard output
 * @param {string} stderr - Standard error
 */
function displayExecutionOutput(stdout, stderr) {
    const outputContent = document.getElementById('outputContent');
    if (!outputContent) return;
    
    outputContent.innerHTML = '';
    
    if (stdout) {
        const stdoutDiv = document.createElement('div');
        stdoutDiv.className = 'output-stdout';
        stdoutDiv.innerHTML = `<strong>Output:</strong><pre>${escapeHtml(stdout)}</pre>`;
        outputContent.appendChild(stdoutDiv);
    }
    
    if (stderr) {
        const stderrDiv = document.createElement('div');
        stderrDiv.className = 'output-stderr';
        stderrDiv.innerHTML = `<strong>Errors:</strong><pre>${escapeHtml(stderr)}</pre>`;
        outputContent.appendChild(stderrDiv);
    }
}

/**
 * Display extracted variables from code execution
 * @param {object} variables - Object with extracted variables
 */
function displayExtractedVariables(variables = {}) {
    const variablePanel = document.getElementById('variablePanel');
    const variableList = document.getElementById('variableList');
    
    if (!variablePanel || !variableList) return;
    
    // Store extracted variables globally
    window.extractedVariables = variables;
    
    variableList.innerHTML = '';
    
    if (!variables || Object.keys(variables).length === 0) {
        variableList.innerHTML = '<p style="color: #999;">No variables extracted</p>';
    } else {
        const table = document.createElement('table');
        table.className = 'variables-table';
        table.innerHTML = '<tr><th>Variable</th><th>Value</th></tr>';
        
        Object.entries(variables).forEach(([key, value]) => {
            const row = document.createElement('tr');
            let displayValue = typeof value === 'object' ? JSON.stringify(value) : String(value);
            if (displayValue.length > 100) displayValue = displayValue.substring(0, 100) + '...';
            
            row.innerHTML = `<td>${escapeHtml(key)}</td><td><code>${escapeHtml(displayValue)}</code></td>`;
            table.appendChild(row);
        });
        
        variableList.appendChild(table);
    }
    
    variablePanel.style.display = 'block';
}

/**
 * Parse variable assignments from text
 * Used for extracting Python-like variable assignments
 * @returns {Array} - Array of variable assignments
 */
function parseVariableAssignmentsFromText() {
    if (!window.editor || !window.editor.getValue) return [];
    
    const code = window.editor.getValue();
    const assignments = [];
    
    // Match: variable = value
    const assignmentRegex = /^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.+?)(?=\n|$)/gm;
    let match;
    
    while ((match = assignmentRegex.exec(code)) !== null) {
        assignments.push({
            name: match[1],
            value: match[2].trim()
        });
    }
    
    return assignments;
}

/**
 * Initialize global variables used across the app
 */
const AppState = {
    editorReady: false,
    extractedVariables: {},
    currentFile: null
};

// Log initialization
console.log('✅ main.js loaded - Global utilities initialized');
