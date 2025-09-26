// Clean UI Integration Script
// This transforms the existing UI into a cleaner, simpler interface

function transformToCleanUI() {
  // Add clean UI styles
  const styleSheet = document.createElement('style');
  styleSheet.textContent = `
    /* Override existing styles with clean UI */
    .page#azl-page .container {
      max-width: 800px !important;
      margin: 0 auto !important;
    }
    
    .page#azl-page .card {
      background: var(--card-bg) !important;
      border-radius: 24px !important;
      padding: 3rem !important;
      border: 1px solid var(--border) !important;
      box-shadow: 0 10px 40px rgba(0,0,0,0.2) !important;
    }
    
    .page#azl-page .card-header {
      border: none !important;
      padding: 0 !important;
      margin-bottom: 2rem !important;
      background: transparent !important;
    }
    
    .page#azl-page .card-header h3 {
      font-size: 3rem !important;
      font-weight: 700 !important;
      text-align: center !important;
      background: linear-gradient(135deg, var(--primary), var(--accent)) !important;
      -webkit-background-clip: text !important;
      -webkit-text-fill-color: transparent !important;
    }
    
    /* Hide all the complex form fields */
    .azl-form-grid {
      display: none !important;
    }
    
    /* Create new clean layout */
    .azl-clean-wrapper {
      display: flex;
      flex-direction: column;
      gap: 2rem;
    }
    
    .azl-clean-input {
      width: 100%;
      padding: 1.5rem 2rem;
      font-size: 1.25rem;
      background: var(--bg);
      border: 2px solid transparent;
      border-radius: 16px;
      color: var(--text);
      transition: all 0.3s;
    }
    
    .azl-clean-input:focus {
      outline: none;
      border-color: var(--primary);
      box-shadow: 0 0 0 4px rgba(139, 92, 246, 0.1);
    }
    
    .azl-clean-actions {
      display: flex;
      align-items: center;
      gap: 2rem;
    }
    
    .azl-clean-btn {
      padding: 1.25rem 3rem;
      background: linear-gradient(135deg, var(--primary), var(--accent));
      border: none;
      border-radius: 16px;
      color: white;
      font-size: 1.25rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.3s;
      display: flex;
      align-items: center;
      gap: 1rem;
      box-shadow: 0 4px 15px rgba(139, 92, 246, 0.3);
    }
    
    .azl-clean-btn:hover {
      transform: translateY(-3px);
      box-shadow: 0 8px 25px rgba(139, 92, 246, 0.4);
    }
    
    .azl-clean-count {
      display: flex;
      align-items: center;
      gap: 0.75rem;
      font-size: 1rem;
      color: var(--text-muted);
    }
    
    .azl-clean-count input {
      width: 70px;
      padding: 0.75rem;
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      text-align: center;
      font-size: 1.125rem;
      color: var(--text);
    }
    
    .azl-clean-advanced {
      margin-top: 2rem;
      padding-top: 2rem;
      border-top: 1px solid var(--border);
    }
    
    .azl-clean-advanced summary {
      cursor: pointer;
      padding: 0.75rem 1rem;
      background: var(--bg);
      border-radius: 12px;
      display: flex;
      align-items: center;
      gap: 0.75rem;
      color: var(--text-muted);
      transition: all 0.2s;
      font-size: 0.9rem;
      font-weight: 600;
      list-style: none;
    }
    
    .azl-clean-advanced summary:hover {
      background: var(--bg-hover);
      color: var(--text);
    }
    
    .azl-clean-options {
      margin-top: 1.5rem;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1rem;
      padding: 1rem;
      background: var(--bg);
      border-radius: 12px;
    }
    
    .azl-clean-option label {
      display: block;
      font-size: 0.875rem;
      color: var(--text-muted);
      margin-bottom: 0.5rem;
      font-weight: 600;
    }
    
    .azl-clean-option input {
      width: 100%;
      padding: 0.625rem 0.875rem;
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      color: var(--text);
      font-size: 0.9rem;
    }
    
    /* Upload page clean styles */
    .page#upload-page .card {
      background: var(--card-bg) !important;
      border-radius: 24px !important;
      padding: 3rem !important;
      border: 1px solid var(--border) !important;
      box-shadow: 0 10px 40px rgba(0,0,0,0.2) !important;
    }
    
    .page#upload-page .card-header h3 {
      font-size: 3rem !important;
      font-weight: 700 !important;
      text-align: center !important;
      background: linear-gradient(135deg, var(--primary), var(--accent)) !important;
      -webkit-background-clip: text !important;
      -webkit-text-fill-color: transparent !important;
    }
    
    .upload-clean-dropzone {
      border: 3px dashed var(--border);
      border-radius: 20px;
      padding: 4rem 2rem;
      text-align: center;
      transition: all 0.3s;
      cursor: pointer;
      background: var(--bg);
      margin-bottom: 2rem;
    }
    
    .upload-clean-dropzone:hover {
      border-color: var(--primary);
      background: linear-gradient(135deg, rgba(139, 92, 246, 0.05), rgba(59, 130, 246, 0.05));
    }
    
    .upload-clean-icon {
      font-size: 4rem;
      margin-bottom: 1rem;
      opacity: 0.5;
    }
    
    .upload-clean-text {
      font-size: 1.25rem;
      margin-bottom: 0.5rem;
      font-weight: 600;
    }
    
    .upload-clean-hint {
      color: var(--text-muted);
      font-size: 0.9rem;
    }
  `;
  document.head.appendChild(styleSheet);
  
  // Transform AutoLearn page
  const azlSection = document.querySelector('#azl-page .azl-section');
  if (azlSection) {
    // Hide existing complex form
    const existingForm = azlSection.querySelector('.azl-form');
    if (existingForm) {
      existingForm.style.display = 'none';
    }
    
    // Create clean UI
    const cleanWrapper = document.createElement('div');
    cleanWrapper.className = 'azl-clean-wrapper';
    cleanWrapper.innerHTML = `
      <div>
        <input 
          type="text" 
          class="azl-clean-input" 
          id="azlTopicClean" 
          placeholder="What topic should the AI learn?"
          onkeyup="document.getElementById('azlTopic').value = this.value"
        />
        <div style="margin-top: 0.75rem; color: var(--text-muted); font-size: 0.9rem;">
          Try "Neural Networks" or "Machine Learning Algorithms"
        </div>
      </div>
      
      <div class="azl-clean-actions">
        <button class="azl-clean-btn" onclick="document.getElementById('autolearnBtn').click()">
          <i class="fas fa-magic"></i>
          <span>Start Learning</span>
        </button>
        
        <div class="azl-clean-count">
          <span>Examples:</span>
          <input 
            type="number" 
            value="5" 
            min="1" 
            max="20"
            onchange="document.getElementById('azlCount').value = this.value"
          />
        </div>
      </div>
      
      <details class="azl-clean-advanced">
        <summary>
          <i class="fas fa-sliders-h"></i>
          Advanced Options
        </summary>
        <div class="azl-clean-options">
          <div class="azl-clean-option">
            <label>Quality Threshold</label>
            <input 
              type="number" 
              placeholder="0.75" 
              step="0.05" 
              min="0" 
              max="1"
              onchange="document.getElementById('azlThreshold').value = this.value"
            />
          </div>
          <div class="azl-clean-option">
            <label>Max Attempts</label>
            <input 
              type="number" 
              placeholder="3" 
              min="1" 
              max="10"
              onchange="document.getElementById('azlMaxAttempts').value = this.value"
            />
          </div>
        </div>
      </details>
    `;
    
    azlSection.insertBefore(cleanWrapper, azlSection.firstChild);
  }
  
  // Transform Upload page
  const uploadSection = document.querySelector('#upload-page .card-body');
  if (uploadSection) {
    const dropzone = uploadSection.querySelector('.dropzone');
    if (dropzone) {
      dropzone.className = 'upload-clean-dropzone';
      dropzone.innerHTML = `
        <div class="upload-clean-icon">📄</div>
        <div class="upload-clean-text">Drop files here or click to browse</div>
        <div class="upload-clean-hint">Supports TXT, PDF, MD files • Max 10MB</div>
      `;
    }
  }
  
  // Simplify the header
  const cardHeaders = document.querySelectorAll('.card-header h3');
  cardHeaders.forEach(header => {
    // Remove extra text
    if (header.textContent.includes('AutoLearn')) {
      header.innerHTML = '<i class="fas fa-robot"></i> AutoLearn';
    }
    if (header.textContent.includes('Upload')) {
      header.innerHTML = '<i class="fas fa-cloud-upload-alt"></i> Upload Knowledge';
    }
  });
  
  // Hide unnecessary elements
  const hideElements = [
    '.azl-head-actions',
    '.azl-sticky',
    '.info-btn',
    '.info-tooltip',
    '.form-hint:not(:first-child)',
    '.upload-hint'
  ];
  
  hideElements.forEach(selector => {
    document.querySelectorAll(selector).forEach(el => {
      el.style.display = 'none';
    });
  });
}

// Apply clean UI on page load
document.addEventListener('DOMContentLoaded', transformToCleanUI);

// Also apply when switching tabs
const originalShowPage = window.showPage;
if (originalShowPage) {
  window.showPage = function(pageName) {
    originalShowPage(pageName);
    setTimeout(transformToCleanUI, 100);
  };
}