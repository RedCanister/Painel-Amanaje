/**
 * Form Utilities - Scalable and modular frontend form handling
 * 
 * This module provides two core functions for dynamic form handling:
 * 1. ToggleFormOptions: Toggle visibility of form sections based on selection
 * 2. PopulateListOptions: Populate select/dropdown elements from backend data
 * 
 * These functions are designed to work with any form, model, and backend structure
 * by using flexible configuration objects.
 * 
 * Architecture:
 * - Data-driven: configuration objects describe form behavior
 * - Scalable: works with any number of fields, models, or routes
 * - Backend-agnostic: compatible with Pydantic models, ORM objects, or plain dicts
 * - Type-safe: includes validation and error handling
 */

/**
 * ToggleFormOptions - Generic form field visibility toggle based on select value
 * 
 * Purpose: Show/hide form sections based on a select element's value
 * This handles the common pattern: "if operation is X, show section A and B; if Y, show C and D"
 * 
 * Configuration object structure:
 * {
 *   triggerElement: HTMLElement or selector (the select that controls visibility),
 *   mapping: {
 *     "optionValue1": ["fieldId1", "fieldId2"],  // show these when option1 selected
 *     "optionValue2": ["fieldId3", "fieldId4"]   // show these when option2 selected
 *   },
 *   allFields: ["fieldId1", "fieldId2", "fieldId3", "fieldId4"]  // all possible fields
 * }
 * 
 * Example:
 * ToggleFormOptions({
 *   triggerElement: '#operationId',
 *   mapping: {
 *     'datasets': ['datasetFields'],
 *     'models': ['modelFields']
 *   },
 *   allFields: ['datasetFields', 'modelFields']
 * });
 */
function ToggleFormOptions(config) {
  // CHANGED: Made function generic to work with any form fields and trigger elements
  // ADDED: Validation of configuration object
  // ADDED: Support for both selector strings and DOM elements
  
  if (!config.triggerElement || !config.mapping || !config.allFields) {
    console.error('ToggleFormOptions: Invalid configuration', config);
    return;
  }

  // Resolve selector to DOM element if needed
  const trigger = typeof config.triggerElement === 'string'
    ? document.querySelector(config.triggerElement)
    : config.triggerElement;

  if (!trigger) {
    console.error('ToggleFormOptions: Trigger element not found', config.triggerElement);
    return;
  }

  // CHANGED: Function to update field visibility
  // Works by: hiding all fields first, then showing only the ones for current selection
  function updateVisibility() {
    const selectedValue = trigger.value;
    const fieldsToShow = config.mapping[selectedValue] || [];

    // Hide all fields first
    config.allFields.forEach(fieldId => {
      const field = document.getElementById(fieldId);
      if (field) {
        field.style.display = 'none';
        // Disable inputs in hidden sections to prevent form submission of hidden data
        const inputs = field.querySelectorAll('input, select, textarea');
        inputs.forEach(input => input.disabled = true);
      }
    });

    // Show and enable only the relevant fields
    fieldsToShow.forEach(fieldId => {
      const field = document.getElementById(fieldId);
      if (field) {
        field.style.display = 'block';
        // Re-enable inputs when section is shown
        const inputs = field.querySelectorAll('input, select, textarea');
        inputs.forEach(input => input.disabled = false);
      }
    });
  }

  // Attach event listener to trigger element
  trigger.addEventListener('change', updateVisibility);

  // Call once on initialization to set initial state
  updateVisibility();
}

/**
 * PopulateListOptions - Generic function to populate select/dropdown elements from backend
 * 
 * Purpose: Fetch data from backend and populate a select element with options
 * Handles both simple lists and complex objects with custom label/value mapping
 * 
 * Configuration object structure:
 * {
 *   selectElement: HTMLElement or selector (the select to populate),
 *   endpoint: '/api/endpoint',  // backend route that returns data
 *   labelField: 'name',  // object property to use as display label
 *   valueField: 'id',  // object property to use as option value
 *   placeholderText: 'Select an option',  // optional placeholder
 *   onSuccess: callback,  // optional callback after population
 *   onError: callback,  // optional error handler
 *   method: 'GET',  // HTTP method (default GET)
 *   headers: {},  // additional headers if needed
 *   transformData: function,  // optional function to transform response data
 * }
 * 
 * Examples:
 * 
 * // Simple: populate from /features endpoint
 * PopulateListOptions({
 *   selectElement: '#datasetId',
 *   endpoint: '/features',
 *   labelField: 'name',
 *   valueField: 'id'
 * });
 * 
 * // With transformation: if backend returns nested data
 * PopulateListOptions({
 *   selectElement: '#modelSelect',
 *   endpoint: '/learning_models',
 *   labelField: 'name',
 *   valueField: 'id',
 *   transformData: (response) => response.data || response  // flatten if needed
 * });
 */
async function PopulateListOptions(config) {
  // CHANGED: Made function generic to work with any endpoint and data structure
  // ADDED: Support for both selector strings and DOM elements
  // ADDED: Error handling and validation
  // ADDED: Optional data transformation for complex responses
  
  if (!config.selectElement || !config.endpoint) {
    console.error('PopulateListOptions: Missing required config', config);
    return;
  }

  // Resolve selector to DOM element if needed
  const selectElement = typeof config.selectElement === 'string'
    ? document.querySelector(config.selectElement)
    : config.selectElement;

  if (!selectElement) {
    console.error('PopulateListOptions: Select element not found', config.selectElement);
    return;
  }

  // Set defaults for optional configuration
  const endpoint = config.endpoint;
  const labelField = config.labelField || 'name';
  const valueField = config.valueField || 'id';
  const placeholder = config.placeholderText || 'Select an option';
  const method = config.method || 'GET';
  const headers = config.headers || {};
  const transformData = config.transformData || (data => data);
  const onSuccess = config.onSuccess || (() => {});
  const onError = config.onError || ((error) => console.error('PopulateListOptions error:', error));

  try {
    // Fetch data from backend
    const response = await fetch(endpoint, { method, headers });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const rawData = await response.json();
    const data = transformData(rawData);

    // Validate data is iterable
    if (!Array.isArray(data)) {
      throw new Error('Response data is not an array');
    }

    // Clear existing options (keeping placeholder if exists)
    selectElement.innerHTML = `<option value="">${placeholder}</option>`;

    // Add new options from data
    data.forEach(item => {
      const label = item[labelField] || 'Unnamed';
      const value = item[valueField] || '';

      const option = document.createElement('option');
      option.value = value;
      option.textContent = label;
      selectElement.appendChild(option);
    });

    // Call success callback if provided
    onSuccess(data);

  } catch (error) {
    console.error('PopulateListOptions: Failed to populate', config.endpoint, error);
    onError(error);
    // Add error message to select
    selectElement.innerHTML = `<option value="">Error loading options</option>`;
  }
}

/**
 * CombinedFormHandler - Advanced function that combines ToggleFormOptions and PopulateListOptions
 * 
 * Purpose: Handle complex forms where toggling sections also requires populating different data
 * 
 * Configuration:
 * {
 *   toggleConfig: { ... },  // same as ToggleFormOptions
 *   populateConfigs: {
 *     "sectionA": { ... },  // different populate configs for each toggle option
 *     "sectionB": { ... }
 *   },
 *   onToggle: (newValue) => {}  // callback when section changes
 * }
 * 
 * Example:
 * CombinedFormHandler({
 *   toggleConfig: {
 *     triggerElement: '#operationId',
 *     mapping: { 'datasets': ['datasetFields'], 'models': ['modelFields'] },
 *     allFields: ['datasetFields', 'modelFields']
 *   },
 *   populateConfigs: {
 *     'datasets': {
 *       selectElement: '#datasetSelect',
 *       endpoint: '/features'
 *     },
 *     'models': {
 *       selectElement: '#modelSelect',
 *       endpoint: '/learning_models'
 *     }
 *   },
 *   onToggle: (value) => console.log('Switched to:', value)
 * });
 */
async function CombinedFormHandler(config) {
  // CHANGED: Advanced handler that coordinates toggle and populate operations
  // ADDED: Automatic population of fields when section visibility changes
  // ADDED: Support for multiple different data sources per section
  
  const trigger = typeof config.toggleConfig.triggerElement === 'string'
    ? document.querySelector(config.toggleConfig.triggerElement)
    : config.toggleConfig.triggerElement;

  if (!trigger) {
    console.error('CombinedFormHandler: Trigger not found');
    return;
  }

  // Store original toggle handler reference
  const originalOnChange = trigger.onchange;

  // Override trigger change behavior to also handle population
  trigger.addEventListener('change', async () => {
    const selectedValue = trigger.value;
    
    // Call toggle handler first
    ToggleFormOptions(config.toggleConfig);

    // Then populate data for the selected section
    if (config.populateConfigs[selectedValue]) {
      const populateConfigs = config.populateConfigs[selectedValue];
      
      // Handle both single populate config or array of configs
      if (Array.isArray(populateConfigs)) {
        for (const cfg of populateConfigs) {
          await PopulateListOptions(cfg);
        }
      } else {
        await PopulateListOptions(populateConfigs);
      }
    }

    // Call custom callback if provided
    if (config.onToggle) {
      config.onToggle(selectedValue);
    }
  });

  // Initialize on page load
  ToggleFormOptions(config.toggleConfig);
  const initialValue = trigger.value;
  if (config.populateConfigs[initialValue]) {
    const populateConfigs = config.populateConfigs[initialValue];
    if (Array.isArray(populateConfigs)) {
      for (const cfg of populateConfigs) {
        await PopulateListOptions(cfg);
      }
    } else {
      await PopulateListOptions(populateConfigs);
    }
  }
}

// Export for use in other scripts (if using modules)
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { ToggleFormOptions, PopulateListOptions, CombinedFormHandler };
}


/**
 * MultiObjectSelector - Advanced multi-object selection with drag-and-drop support
 * 
 * Purpose: Allow users to select multiple objects and organize them for analysis
 * Supports:
 * - Adding/removing objects from selection
 * - Drag-and-drop reordering
 * - Role assignment (input/output for models, features/target for datasets)
 * - Persistence to hidden form fields
 * 
 * Configuration:
 * {
 *   containerId: 'containerDiv',  // Container for the selector
 *   objectType: 'dataset' | 'model' | 'feature',  // Type of objects being selected
 *   endpoint: '/api/endpoint',  // Endpoint to fetch available objects
 *   labelField: 'name',  // Object property for display
 *   valueField: 'id',  // Object property for internal value
 *   
 *   roles: ['input', 'output'],  // Optional: roles for objects (e.g., features vs target)
 *   onSelectionChange: (selectedObjects) => {},  // Callback when selection changes
 *   maxSelection: 10,  // Optional: limit number of selections
 * }
 * 
 * Example:
 * MultiObjectSelector({
 *   containerId: 'featureSelector',
 *   objectType: 'feature',
 *   endpoint: '/features',
 *   roles: ['input_features', 'target_feature'],
 *   onSelectionChange: (selected) => console.log('Selected:', selected)
 * });
 * 
 * The widget is used by the upload analysis page for dataset/model comparisons
 * and can be reused by other pages without inline event handlers.
 */

class MultiObjectSelector {
  constructor(config) {
    // CHANGED: Initialize selector with configuration
    this.config = {
      labelField: 'name',
      valueField: 'id',
      maxSelection: null,
      roles: null,
      onSelectionChange: () => {},
      ...config
    };
    
    this.selectedObjects = [];
    this.availableObjects = [];
    
    // Resolve container
    this.container = typeof config.containerId === 'string'
      ? document.getElementById(config.containerId)
      : config.containerId;
    
    if (!this.container) {
      console.error('MultiObjectSelector: Container not found', config.containerId);
      return;
    }
    
    this.init();
  }
  
  // CHANGED: Initialize the selector UI and fetch available objects
  async init() {
    try {
      const response = await fetch(this.config.endpoint);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      
      this.availableObjects = await response.json();
      this.render();
    } catch (error) {
      console.error('MultiObjectSelector: Failed to fetch objects', error);
      this.container.innerHTML = `<div style="color: red;">Error loading objects: ${error.message}</div>`;
    }
  }
  
  // CHANGED: Render the complete selector UI
  render() {
    const { labelField, valueField, maxSelection, roles } = this.config;
    
    // Available objects list
    let availableHTML = '<div style="margin-bottom: 1.5rem;">';
    availableHTML += '<label style="display: block; margin-bottom: 0.5rem; font-weight: 600;">Available Objects</label>';
    availableHTML += '<div style="border: 1px solid #ddd; border-radius: 4px; padding: 0.5rem; background: #f9f9f9; min-height: 100px;">';
    
    if (this.availableObjects.length === 0) {
      availableHTML += '<p style="color: #999; margin: 0;">No objects available</p>';
    } else {
      this.availableObjects.forEach(obj => {
        const isSelected = this.selectedObjects.some(s => s[valueField] === obj[valueField]);
        const label = obj[labelField] || `Object ${obj[valueField]}`;
        
        availableHTML += `
          <button 
            type="button"
            style="
              margin: 0.25rem;
              padding: 0.5rem 1rem;
              background: ${isSelected ? '#9945ff' : '#e5e7eb'};
              color: ${isSelected ? 'white' : '#374151'};
              border: none;
              border-radius: 4px;
              cursor: pointer;
              transition: all 0.2s;
            "
            data-legacy-mos-option="${JSON.stringify(obj[valueField]).replace(/"/g, '&quot;')}"
            mos-ref
          >
            ${isSelected ? '✓ ' : '+ '}${label}
          </button>
        `;
      });
    }
    
    availableHTML += '</div></div>';
    
    // Selected objects (with drag support)
    let selectedHTML = '<div>';
    selectedHTML += '<label style="display: block; margin-bottom: 0.5rem; font-weight: 600;">Selected Objects';
    if (maxSelection) selectedHTML += ` (${this.selectedObjects.length}/${maxSelection})`;
    selectedHTML += '</label>';
    selectedHTML += '<div style="border: 2px dashed #9945ff; border-radius: 4px; padding: 1rem; min-height: 100px; background: #f3e8ff;">';
    
    if (this.selectedObjects.length === 0) {
      selectedHTML += '<p style="color: #999; margin: 0; text-align: center;">Drag or click objects to add them here</p>';
    } else {
      this.selectedObjects.forEach((obj, idx) => {
        const label = obj[labelField] || `Object ${obj[valueField]}`;
        
        selectedHTML += `
          <div 
            draggable="true"
            style="
              background: white;
              border: 1px solid #9945ff;
              border-radius: 4px;
              padding: 0.75rem;
              margin-bottom: 0.5rem;
              display: flex;
              justify-content: space-between;
              align-items: center;
              cursor: move;
            "
            ondragstart="event.dataTransfer.effectAllowed='move'; event.dataTransfer.setData('index', ${idx})"
            ondragover="event.preventDefault(); event.dataTransfer.dropEffect='move'"
            data-legacy-mos-drop="${idx}"
          >
            <span>⋮⋮ ${label}</span>
            <button 
              type="button"
              data-legacy-mos-remove="${idx}"
              style="
                background: #ef4444;
                color: white;
                border: none;
                border-radius: 3px;
                padding: 0.25rem 0.75rem;
                cursor: pointer;
              "
            >
              ✕
            </button>
          </div>
        `;
      });
    }
    
    selectedHTML += '</div></div>';
    
    // Form fields to store selection
    let hiddenHTML = '<input type="hidden" id="selectedObjectsJson" name="selectedObjects" value="' + 
                     JSON.stringify(this.selectedObjects).replace(/"/g, '&quot;') + '">';
    
    // Combine and set
    this.container.innerHTML = availableHTML + selectedHTML + hiddenHTML;
    
    // Make buttons reference this instance
    this.container.querySelectorAll('[mos-ref]').forEach(btn => btn.mos = this);
    
    // Make selected divs reference this instance
    this.container.querySelectorAll('[ondragstart]').forEach(div => div.mos = this);
  }
  
  // CHANGED: Toggle object in selection
  toggleObject(valueField, obj) {
    const index = this.selectedObjects.findIndex(s => s[valueField] === obj[valueField]);
    
    if (index > -1) {
      // Remove if selected
      this.selectedObjects.splice(index, 1);
    } else {
      // Add if not selected
      if (this.config.maxSelection && this.selectedObjects.length >= this.config.maxSelection) {
        alert(`Maximum ${this.config.maxSelection} objects allowed`);
        return;
      }
      this.selectedObjects.push(obj);
    }
    
    this.config.onSelectionChange(this.selectedObjects);
    this.render();
  }
  
  // CHANGED: Remove object from selection
  removeObject(index) {
    this.selectedObjects.splice(index, 1);
    this.config.onSelectionChange(this.selectedObjects);
    this.render();
  }
  
  // CHANGED: Reorder objects via drag-and-drop
  reorderObjects(event, targetIndex) {
    event.preventDefault();
    const sourceIndex = parseInt(event.dataTransfer.getData('index'));
    
    if (sourceIndex === targetIndex) return;
    
    const [obj] = this.selectedObjects.splice(sourceIndex, 1);
    this.selectedObjects.splice(targetIndex, 0, obj);
    
    this.config.onSelectionChange(this.selectedObjects);
    this.render();
  }
  
  // CHANGED: Get selected objects for submission
  getSelection() {
    return this.selectedObjects;
  }
  
  // CHANGED: Clear all selections
  clear() {
    this.selectedObjects = [];
    this.config.onSelectionChange(this.selectedObjects);
    this.render();
  }

  static escapeHtml(value) {
    return String(value ?? '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  async init() {
    this.config = {
      hiddenFieldName: 'selectedObjects',
      hiddenFieldId: null,
      roleField: 'role',
      defaultRole: null,
      availableTitle: 'Available Objects',
      selectedTitle: 'Selected Objects',
      emptyAvailableText: 'No objects available',
      emptySelectedText: 'Click objects to add them here',
      objects: null,
      initialSelection: [],
      ...this.config
    };
    this.selectedObjects = Array.isArray(this.config.initialSelection)
      ? this.config.initialSelection.map((item) => ({ ...item }))
      : [];

    try {
      if (Array.isArray(this.config.objects)) {
        this.availableObjects = this.config.objects;
      } else {
        const response = await fetch(this.config.endpoint);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        this.availableObjects = Array.isArray(data) ? data : (data.items || data.data || []);
      }
      this.render();
    } catch (error) {
      console.error('MultiObjectSelector: Failed to fetch objects', error);
      this.container.innerHTML = `<div class="alert alert-error">Error loading objects: ${MultiObjectSelector.escapeHtml(error.message)}</div>`;
    }
  }

  getValue(obj) {
    return obj?.[this.config.valueField];
  }

  getLabel(obj) {
    const value = this.getValue(obj);
    return obj?.[this.config.labelField] || `Object ${value ?? '-'}`;
  }

  getRole(obj) {
    const roles = Array.isArray(this.config.roles) ? this.config.roles : [];
    return obj?.[this.config.roleField] || this.config.defaultRole || roles[0] || '';
  }

  isSelected(obj) {
    const value = String(this.getValue(obj));
    return this.selectedObjects.some((item) => String(this.getValue(item)) === value);
  }

  emitSelection() {
    this.config.onSelectionChange(this.getSelection());
  }

  render() {
    const roles = Array.isArray(this.config.roles) ? this.config.roles : [];
    const hiddenId = this.config.hiddenFieldId || `${this.container.id || 'multiObjectSelector'}Selected`;
    const maxSelection = this.config.maxSelection;

    const availableHTML = this.availableObjects.length
      ? this.availableObjects.map((obj, index) => {
          const selected = this.isSelected(obj);
          return `
            <button
              type="button"
              class="mos-option ${selected ? 'selected' : ''}"
              data-mos-available-index="${index}"
              aria-pressed="${selected ? 'true' : 'false'}"
            >
              <span>${selected ? 'Selected' : 'Add'}</span>
              <strong>${MultiObjectSelector.escapeHtml(this.getLabel(obj))}</strong>
            </button>
          `;
        }).join('')
      : `<div class="helper-text">${MultiObjectSelector.escapeHtml(this.config.emptyAvailableText)}</div>`;

    const selectedHTML = this.selectedObjects.length
      ? this.selectedObjects.map((obj, index) => `
          <div class="mos-selected-item" draggable="true" data-mos-selected-index="${index}">
            <span class="mos-drag-handle" aria-hidden="true">::</span>
            <strong>${MultiObjectSelector.escapeHtml(this.getLabel(obj))}</strong>
            ${roles.length ? `
              <select data-mos-role-index="${index}" aria-label="Role for ${MultiObjectSelector.escapeHtml(this.getLabel(obj))}">
                ${roles.map((role) => `
                  <option value="${MultiObjectSelector.escapeHtml(role)}" ${this.getRole(obj) === role ? 'selected' : ''}>
                    ${MultiObjectSelector.escapeHtml(role)}
                  </option>
                `).join('')}
              </select>
            ` : ''}
            <button type="button" class="mos-remove" data-mos-remove-index="${index}" aria-label="Remove ${MultiObjectSelector.escapeHtml(this.getLabel(obj))}">Remove</button>
          </div>
        `).join('')
      : `<div class="helper-text">${MultiObjectSelector.escapeHtml(this.config.emptySelectedText)}</div>`;

    this.container.innerHTML = `
      <div class="mos-shell" data-multi-object-selector="true">
        <div class="mos-column">
          <div class="mos-heading">${MultiObjectSelector.escapeHtml(this.config.availableTitle)}</div>
          <div class="mos-options">${availableHTML}</div>
        </div>
        <div class="mos-column">
          <div class="mos-heading">
            ${MultiObjectSelector.escapeHtml(this.config.selectedTitle)}
            ${maxSelection ? `<span>${this.selectedObjects.length}/${maxSelection}</span>` : ''}
          </div>
          <div class="mos-selected">${selectedHTML}</div>
        </div>
        <input
          type="hidden"
          id="${MultiObjectSelector.escapeHtml(hiddenId)}"
          name="${MultiObjectSelector.escapeHtml(this.config.hiddenFieldName)}"
          value="${MultiObjectSelector.escapeHtml(JSON.stringify(this.getSelection()))}"
        >
      </div>
    `;

    this.bindEvents();
  }

  bindEvents() {
    this.container.querySelectorAll('[data-mos-available-index]').forEach((button) => {
      button.addEventListener('click', () => {
        const index = Number(button.getAttribute('data-mos-available-index'));
        this.toggleObject(this.availableObjects[index]);
      });
    });

    this.container.querySelectorAll('[data-mos-remove-index]').forEach((button) => {
      button.addEventListener('click', () => {
        this.removeObject(Number(button.getAttribute('data-mos-remove-index')));
      });
    });

    this.container.querySelectorAll('[data-mos-role-index]').forEach((field) => {
      field.addEventListener('change', () => {
        const index = Number(field.getAttribute('data-mos-role-index'));
        if (!this.selectedObjects[index]) return;
        this.selectedObjects[index] = {
          ...this.selectedObjects[index],
          [this.config.roleField]: field.value
        };
        this.emitSelection();
        this.render();
      });
    });

    this.container.querySelectorAll('[data-mos-selected-index]').forEach((item) => {
      item.addEventListener('dragstart', (event) => {
        event.dataTransfer.effectAllowed = 'move';
        event.dataTransfer.setData('text/plain', item.getAttribute('data-mos-selected-index'));
      });
      item.addEventListener('dragover', (event) => {
        event.preventDefault();
        event.dataTransfer.dropEffect = 'move';
      });
      item.addEventListener('drop', (event) => {
        this.reorderObjects(event, Number(item.getAttribute('data-mos-selected-index')));
      });
    });
  }

  toggleObject(obj) {
    const value = String(this.getValue(obj));
    const index = this.selectedObjects.findIndex((item) => String(this.getValue(item)) === value);
    if (index > -1) {
      this.selectedObjects.splice(index, 1);
    } else {
      if (this.config.maxSelection && this.selectedObjects.length >= this.config.maxSelection) {
        alert(`Maximum ${this.config.maxSelection} objects allowed`);
        return;
      }
      const roles = Array.isArray(this.config.roles) ? this.config.roles : [];
      this.selectedObjects.push({
        ...obj,
        ...(roles.length ? { [this.config.roleField]: this.config.defaultRole || roles[0] } : {})
      });
    }
    this.emitSelection();
    this.render();
  }

  removeObject(index) {
    if (Number.isNaN(index) || index < 0 || index >= this.selectedObjects.length) return;
    this.selectedObjects.splice(index, 1);
    this.emitSelection();
    this.render();
  }

  reorderObjects(event, targetIndex) {
    event.preventDefault();
    const sourceIndex = Number(event.dataTransfer.getData('text/plain'));
    if (Number.isNaN(sourceIndex) || sourceIndex === targetIndex) return;
    const [obj] = this.selectedObjects.splice(sourceIndex, 1);
    this.selectedObjects.splice(targetIndex, 0, obj);
    this.emitSelection();
    this.render();
  }

  getSelection() {
    return this.selectedObjects.map((item, order) => ({ ...item, order }));
  }

  setAvailableObjects(objects = []) {
    this.availableObjects = Array.isArray(objects) ? objects : [];
    this.render();
  }

  clear() {
    this.selectedObjects = [];
    this.emitSelection();
    this.render();
  }
}

// Export for use with or without modules
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { ToggleFormOptions, PopulateListOptions, CombinedFormHandler, MultiObjectSelector };
}
