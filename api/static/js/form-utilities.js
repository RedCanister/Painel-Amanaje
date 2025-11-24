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


// TODO - I need a mechanism I can reuse on the selection of multiple objects
// In the frontend. I would like a way to select objects of any type on the interface that
// allows me to use them in the same page, without reloading. The select object act
// as values that get sent in to the backend.  If it is a dataset, 
// I need to select the columns to be analyzed. If it is a model, 
// I need to select the dataset's columns that are used as input or output
// If it could be a drag-and-drop selection mechanism I can use to sort around the input or
// output colummns.
