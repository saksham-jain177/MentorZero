// Interactive Help Guide for MentorZero

(function() {
  // Create help button
  const helpButton = document.createElement('button');
  helpButton.className = 'help-fab';
  helpButton.innerHTML = '<span class="help-icon">?</span>';
  helpButton.title = 'Help & Guide';
  
  // Create help modal
  const helpModal = document.createElement('div');
  helpModal.className = 'help-modal';
  helpModal.innerHTML = `
    <div class="help-modal-content">
      <div class="help-header">
        <h2>Welcome to MentorZero</h2>
        <button class="help-close">&times;</button>
      </div>
      
      <div class="help-nav">
        <button class="help-nav-btn active" data-section="quick-start">Quick Start</button>
        <button class="help-nav-btn" data-section="chat">Chat Guide</button>
        <button class="help-nav-btn" data-section="upload">Upload Guide</button>
        <button class="help-nav-btn" data-section="autolearn">AutoLearn Guide</button>
        <button class="help-nav-btn" data-section="shortcuts">Shortcuts</button>
        <button class="help-nav-btn" data-section="tips">Pro Tips</button>
      </div>
      
      <div class="help-body">
        <!-- Quick Start Section -->
        <div class="help-section active" id="quick-start">
          <h3>🚀 Quick Start Guide</h3>
          <div class="help-steps">
            <div class="help-step">
              <div class="step-number">1</div>
              <div class="step-content">
                <h4>Start a Chat</h4>
                <p>Go to the <strong>Chat</strong> tab and type any topic or question. The AI will provide structured lessons and explanations.</p>
                <div class="help-example">
                  <code>Example: "Neural Networks" or "Explain machine learning"</code>
                </div>
              </div>
            </div>
            
            <div class="help-step">
              <div class="step-number">2</div>
              <div class="step-content">
                <h4>Upload Knowledge</h4>
                <p>Use the <strong>Upload</strong> tab to add documents. The AI will use these to provide more accurate, context-aware responses.</p>
              </div>
            </div>
            
            <div class="help-step">
              <div class="step-number">3</div>
              <div class="step-content">
                <h4>Train the AI</h4>
                <p>Visit <strong>AutoLearn</strong> to let the AI generate and validate its own training examples. This makes it smarter over time!</p>
              </div>
            </div>
          </div>
        </div>
        
        <!-- Chat Guide Section -->
        <div class="help-section" id="chat">
          <h3>💬 Chat Guide</h3>
          <div class="help-content">
            <h4>How to Use Chat</h4>
            <ol class="help-list">
              <li>Enter a topic or question in the input field</li>
              <li>Press Enter or click Send</li>
              <li>The AI will provide a structured response with:
                <ul>
                  <li>📚 Core concepts</li>
                  <li>🎯 Key points</li>
                  <li>💡 Examples</li>
                  <li>✅ Practice questions</li>
                </ul>
              </li>
              <li>Click "Optimize" to improve any response</li>
              <li>Use "Show Progress" to see learning statistics</li>
            </ol>
            
            <div class="help-tip">
              <strong>💡 Tip:</strong> Be specific with your questions for better responses. Instead of "math", try "linear algebra basics".
            </div>
          </div>
        </div>
        
        <!-- Upload Guide Section -->
        <div class="help-section" id="upload">
          <h3>📄 Upload Guide</h3>
          <div class="help-content">
            <h4>Adding Your Documents</h4>
            <ol class="help-list">
              <li><strong>Drag & Drop:</strong> Simply drag files onto the upload area</li>
              <li><strong>Browse:</strong> Click the upload area to select files</li>
              <li><strong>URL Import:</strong> Paste a URL to fetch online content</li>
            </ol>
            
            <h4>Supported Formats</h4>
            <ul class="help-list">
              <li>📝 Text files (.txt)</li>
              <li>📄 PDF documents (.pdf)</li>
              <li>📋 Markdown files (.md)</li>
              <li>🔗 Web pages (via URL)</li>
            </ul>
            
            <div class="help-process">
              <h4>What Happens Next?</h4>
              <div class="process-flow">
                <span>Upload</span> → <span>Extract Text</span> → <span>Create Embeddings</span> → <span>Index in Vector Store</span>
              </div>
              <p>Your documents are processed locally and used to enhance AI responses with relevant context.</p>
            </div>
          </div>
        </div>
        
        <!-- AutoLearn Guide Section -->
        <div class="help-section" id="autolearn">
          <h3>🤖 AutoLearn Guide</h3>
          <div class="help-content">
            <h4>Self-Improving AI Training</h4>
            <p>AutoLearn uses <strong>Autonomous Zero-label Learning (AZL)</strong> to generate and validate training examples automatically.</p>
            
            <h4>How to Use AutoLearn</h4>
            <ol class="help-list">
              <li>Enter a topic you want the AI to learn about</li>
              <li>Set the number of examples (5-20 recommended)</li>
              <li>Click "Start Learning"</li>
              <li>Watch the real-time progress:
                <ul>
                  <li>✨ Generation of Q&A pairs</li>
                  <li>🔍 Validation checks</li>
                  <li>⚖️ Quality scoring</li>
                  <li>✅ Auto-acceptance of good examples</li>
                </ul>
              </li>
            </ol>
            
            <h4>Advanced Options</h4>
            <div class="help-options">
              <p><strong>Quality Threshold:</strong> Minimum score (0-1) for accepting examples. Default: 0.75</p>
              <p><strong>Max Attempts:</strong> How many times to retry failed examples. Default: 3</p>
            </div>
            
            <div class="help-tip">
              <strong>💡 Tip:</strong> Start with broad topics like "Machine Learning" before diving into specifics like "Gradient Boosting".
            </div>
          </div>
        </div>
        
        <!-- Shortcuts Section -->
        <div class="help-section" id="shortcuts">
          <h3>⌨️ Keyboard Shortcuts</h3>
          <div class="help-shortcuts">
            <div class="shortcut-item">
              <kbd>Alt</kbd> + <kbd>T</kbd>
              <span>Switch between tabs</span>
            </div>
            <div class="shortcut-item">
              <kbd>Alt</kbd> + <kbd>L</kbd>
              <span>Toggle light/dark theme</span>
            </div>
            <div class="shortcut-item">
              <kbd>/</kbd>
              <span>Focus on search/input field</span>
            </div>
            <div class="shortcut-item">
              <kbd>Enter</kbd>
              <span>Send message (in chat)</span>
            </div>
            <div class="shortcut-item">
              <kbd>Esc</kbd>
              <span>Close modals/dialogs</span>
            </div>
          </div>
        </div>
        
        <!-- Pro Tips Section -->
        <div class="help-section" id="tips">
          <h3>🎯 Pro Tips</h3>
          <div class="help-tips-grid">
            <div class="pro-tip-card">
              <h4>🎨 Optimize Responses</h4>
              <p>After receiving a response, click "Optimize" to get alternative versions with different teaching styles.</p>
            </div>
            
            <div class="pro-tip-card">
              <h4>📊 Track Progress</h4>
              <p>Use "Show Progress" to see your learning statistics and interaction history.</p>
            </div>
            
            <div class="pro-tip-card">
              <h4>🔄 Clear Chat</h4>
              <p>Click the clear button to start fresh. Your uploaded documents remain indexed.</p>
            </div>
            
            <div class="pro-tip-card">
              <h4>🌐 Offline First</h4>
              <p>MentorZero works entirely offline once set up. No internet required for core features!</p>
            </div>
            
            <div class="pro-tip-card">
              <h4>📚 Context Matters</h4>
              <p>Upload relevant documents before asking questions for more accurate, contextual responses.</p>
            </div>
            
            <div class="pro-tip-card">
              <h4>🎯 Be Specific</h4>
              <p>Specific questions get better answers. "Explain backpropagation in neural networks" > "AI"</p>
            </div>
          </div>
        </div>
      </div>
      
      <div class="help-footer">
        <p>MentorZero v1.0 • Local AI that learns and improves</p>
      </div>
    </div>
  `;
  
  // Add styles
  const styles = document.createElement('style');
  styles.textContent = `
    /* Floating Action Button */
    .help-fab {
      position: fixed;
      bottom: 24px;
      right: 24px;
      width: 56px;
      height: 56px;
      border-radius: 50%;
      background: radial-gradient(circle at 60% 40%, rgba(99,102,241,0.9), rgba(99,102,241,0.7));
      border: none;
      cursor: pointer;
      box-shadow: 0 8px 24px rgba(99,102,241,0.35), 0 0 0 6px rgba(99,102,241,0.15) inset;
      transition: all 0.3s;
      z-index: 9999;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    
    .help-fab:hover {
      transform: scale(1.1);
      box-shadow: 0 6px 20px rgba(139, 92, 246, 0.5);
    }
    
    .help-fab:active {
      transform: scale(0.95);
    }
    
    .help-icon {
      color: white;
      font-size: 22px;
      font-weight: bold;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Help Modal */
    .help-modal {
      display: none;
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: rgba(0, 0, 0, 0.8);
      backdrop-filter: blur(10px);
      z-index: 10000;
      animation: fadeIn 0.3s;
    }
    
    .help-modal.show {
      display: flex;
      align-items: center;
      justify-content: center;
    }
    
    .help-modal-content {
      background: var(--card-bg);
      border-radius: 20px;
      width: 90%;
      max-width: 800px;
      max-height: 85vh;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
      animation: slideUp 0.3s;
    }
    
    @keyframes fadeIn {
      from { opacity: 0; }
      to { opacity: 1; }
    }
    
    @keyframes slideUp {
      from { transform: translateY(20px); opacity: 0; }
      to { transform: translateY(0); opacity: 1; }
    }
    
    /* Help Header */
    .help-header {
      padding: 1.5rem 2rem;
      background: linear-gradient(135deg, var(--primary), var(--accent));
      color: white;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    
    .help-header h2 {
      margin: 0;
      font-size: 1.5rem;
    }
    
    .help-close {
      background: none;
      border: none;
      color: white;
      font-size: 2rem;
      cursor: pointer;
      opacity: 0.8;
      transition: opacity 0.2s;
      padding: 0;
      width: 32px;
      height: 32px;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    
    .help-close:hover {
      opacity: 1;
    }
    
    /* Help Navigation */
    .help-nav {
      display: flex;
      gap: 0.5rem;
      padding: 1rem;
      background: var(--bg);
      border-bottom: 1px solid var(--border);
      overflow-x: auto;
    }
    
    .help-nav-btn {
      padding: 0.5rem 1rem;
      background: transparent;
      border: 1px solid var(--border);
      border-radius: 8px;
      color: var(--text-muted);
      cursor: pointer;
      transition: all 0.2s;
      white-space: nowrap;
      font-size: 0.9rem;
    }
    
    .help-nav-btn:hover {
      background: var(--bg-hover);
      color: var(--text);
    }
    
    .help-nav-btn.active {
      background: var(--primary);
      color: white;
      border-color: var(--primary);
    }
    
    /* Help Body */
    .help-body {
      flex: 1;
      overflow-y: auto;
      padding: 2rem;
      scrollbar-gutter: stable; /* prevent layout jumps */
    }
    
    .help-section {
      display: none;
    }
    
    .help-section.active {
      display: block;
      animation: fadeIn 0.3s;
    }
    
    .help-section h3 {
      margin-bottom: 1.5rem;
      font-size: 1.5rem;
      color: var(--text);
    }
    
    .help-section h4 {
      margin-top: 1.5rem;
      margin-bottom: 0.75rem;
      color: var(--text);
      font-size: 1.1rem;
    }
    
    /* Help Steps */
    .help-steps {
      display: flex;
      flex-direction: column;
      gap: 1.5rem;
    }
    
    .help-step {
      display: flex;
      gap: 1rem;
      align-items: flex-start;
    }
    
    .step-number {
      flex-shrink: 0;
      width: 32px;
      height: 32px;
      background: var(--primary);
      color: white;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: bold;
    }
    
    .step-content {
      flex: 1;
    }
    
    .step-content h4 {
      margin-top: 0;
      margin-bottom: 0.5rem;
    }
    
    .step-content p {
      color: var(--text-muted);
      line-height: 1.6;
    }
    
    /* Help Example */
    .help-example {
      margin-top: 0.5rem;
      padding: 0.75rem;
      background: var(--bg);
      border-radius: 8px;
      border-left: 3px solid var(--primary);
    }
    
    .help-example code {
      color: var(--primary);
      font-size: 0.9rem;
    }
    
    /* Help List */
    .help-list {
      margin: 1rem 0;
      padding-left: 1.5rem;
      line-height: 1.8;
      color: var(--text-muted);
    }
    
    .help-list li {
      margin-bottom: 0.5rem;
    }
    
    .help-list ul {
      margin-top: 0.5rem;
      margin-left: 1rem;
    }
    
    /* Help Tip */
    .help-tip {
      margin-top: 1.5rem;
      padding: 1rem;
      background: linear-gradient(135deg, rgba(139, 92, 246, 0.1), rgba(59, 130, 246, 0.1));
      border-radius: 8px;
      border-left: 3px solid var(--primary);
    }
    
    /* Process Flow */
    .process-flow {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      margin: 1rem 0;
      flex-wrap: wrap;
    }
    
    .process-flow span {
      padding: 0.5rem 1rem;
      background: var(--bg);
      border-radius: 6px;
      font-size: 0.9rem;
    }
    
    /* Shortcuts */
    .help-shortcuts {
      display: flex;
      flex-direction: column;
      gap: 1rem;
    }
    
    .shortcut-item {
      display: flex;
      align-items: center;
      gap: 1rem;
      padding: 0.75rem;
      background: var(--bg);
      border-radius: 8px;
    }
    
    .shortcut-item kbd {
      padding: 0.25rem 0.5rem;
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 4px;
      font-family: monospace;
      font-size: 0.9rem;
    }
    
    .shortcut-item span {
      color: var(--text-muted);
    }
    
    /* Pro Tips Grid */
    .help-tips-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
      gap: 1rem;
      margin-top: 1rem;
    }
    
    .pro-tip-card {
      padding: 1rem;
      background: var(--bg);
      border-radius: 8px;
      border: 1px solid var(--border);
    }
    
    .pro-tip-card h4 {
      margin-top: 0;
      margin-bottom: 0.5rem;
      font-size: 1rem;
    }
    
    .pro-tip-card p {
      margin: 0;
      color: var(--text-muted);
      font-size: 0.9rem;
      line-height: 1.5;
    }
    
    /* Help Footer */
    .help-footer {
      padding: 1rem 2rem;
      background: var(--bg);
      border-top: 1px solid var(--border);
      text-align: center;
    }
    
    .help-footer p {
      margin: 0;
      color: var(--text-muted);
      font-size: 0.875rem;
    }
    
    /* Responsive */
    @media (max-width: 768px) {
      .help-modal-content {
        width: 95%;
        max-height: 95vh;
        border-radius: 12px;
      }
      
      .help-nav {
        padding: 0.75rem;
      }
      
      .help-body {
        padding: 1.5rem;
      }
      
      .help-tips-grid {
        grid-template-columns: 1fr;
      }
    }
  `;
  
  // Add to page
  document.head.appendChild(styles);
  document.body.appendChild(helpButton);
  document.body.appendChild(helpModal);
  
  // Event handlers
  helpButton.addEventListener('click', () => {
    helpModal.classList.add('show');
    // Lock background scroll when modal open
    document.documentElement.style.overflow = 'hidden';
  });
  
  helpModal.querySelector('.help-close').addEventListener('click', () => {
    helpModal.classList.remove('show');
    document.documentElement.style.overflow = '';
  });
  
  // Close on background click
  helpModal.addEventListener('click', (e) => {
    if (e.target === helpModal) {
      helpModal.classList.remove('show');
      document.documentElement.style.overflow = '';
    }
  });
  
  // Close on Escape key
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && helpModal.classList.contains('show')) {
      helpModal.classList.remove('show');
      document.documentElement.style.overflow = '';
    }
  });
  
  // Navigation tabs
  const navButtons = helpModal.querySelectorAll('.help-nav-btn');
  const sections = helpModal.querySelectorAll('.help-section');
  
  navButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetSection = btn.dataset.section;
      
      // Update active states
      navButtons.forEach(b => b.classList.remove('active'));
      sections.forEach(s => s.classList.remove('active'));
      
      btn.classList.add('active');
      document.getElementById(targetSection).classList.add('active');
    });
  });
  
  // Add pulse animation to help button on first load
  const hasSeenHelp = localStorage.getItem('hasSeenHelp');
  if (!hasSeenHelp) {
    helpButton.style.animation = 'pulse 2s infinite';
    const pulseStyle = document.createElement('style');
    pulseStyle.textContent = `
      @keyframes pulse {
        0% { box-shadow: 0 4px 12px rgba(139, 92, 246, 0.4); }
        50% { box-shadow: 0 4px 20px rgba(139, 92, 246, 0.8); }
        100% { box-shadow: 0 4px 12px rgba(139, 92, 246, 0.4); }
      }
    `;
    document.head.appendChild(pulseStyle);
    
    // Stop pulsing after first click
    helpButton.addEventListener('click', () => {
      helpButton.style.animation = '';
      localStorage.setItem('hasSeenHelp', 'true');
    }, { once: true });
  }
})();