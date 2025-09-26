// Helper to convert our normalized markdown-ish text to HTML with lists and headings
function formatMarkdownHtml(text) {
  let s = (text || '').replace(/\r\n/g, '\n');
  // Headings
  s = s.replace(/\n\s*#{1,6}\s+(.*)/g, '<h4 style="margin-top: 1.5rem; margin-bottom: 0.75rem; color: var(--primary); font-weight: 600;">$1<\/h4>');
  // Bold
  s = s.replace(/\*\*(.*?)\*\*/g, '<strong style="font-weight: 600;">$1<\/strong>');
  // Bullet lists (lines beginning with - or •)
  s = s.replace(/(\n\s*(?:-|•)\s+.*(?:\n\s*(?:-|•)\s+.*)+)/g, (match) => {
    return '<ul style="margin: 1rem 0; padding-left: 1.5rem;">' +
           match.replace(/\n\s*(?:-|•)\s+(.*)/g, '<li style="margin-bottom: 0.5rem;">$1<\/li>') +
           '<\/ul>';
  });
  // Numbered lists
  s = s.replace(/(\n\s*\d+\.\s+.*(?:\n\s*\d+\.\s+.*)+)/g, (match) => {
    return '<ol style="margin: 1rem 0; padding-left: 1.5rem;">' +
           match.replace(/\n\s*\d+\.\s+(.*)/g, '<li style="margin-bottom: 0.5rem;">$1<\/li>') +
           '<\/ol>';
  });
  // Paragraphs
  s = s.replace(/\n\n/g, '<\/p><p style="margin: 1rem 0;">');
  s = s.replace(/\n/g, '<br>');
  if (!s.startsWith('<')) s = '<p style="margin: 1rem 0;">' + s + '<\/p>';
  return s;
}
// Keyboard shortcut: press 'O' over a bot message to Optimize it
document.addEventListener('keydown', (e) => {
  if (e.key.toLowerCase() !== 'o') return;
  const menu = document.querySelector('.message.ai-message:hover .msg-menu-list');
  if (menu && menu.firstChild && typeof menu.firstChild.click === 'function') {
    e.preventDefault();
    menu.firstChild.click();
  }
});

async function reaskVariant(topic, variant, sessionId) {
  if (!topic) {
    showToast('No topic found to re-ask', 'error');
    return;
  }
  try {
    startTopProgress();
    const startAt = Date.now();
    const res = await fetch('/reask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ topic, variant, sessionId: sessionId || null })
    });
    const data = await res.json();
    if (res.ok && data.text) {
      addMessage('bot', data.text, { sessionId, lastUserTopic: topic, createdAt: new Date(), elapsedMs: Date.now() - startAt });
    } else {
      showToast('Failed to re-ask', 'error');
    }
  } catch (err) {
    console.error('Re-ask failed', err);
    showToast('Failed to re-ask', 'error');
  } finally {
    finishTopProgress();
  }
}
import { appState, setSessionId } from './state.js';
import { startTeach } from './teach.js';
import { submitAnswer } from './answer.js';
import { uploadText, scrapeFetch } from './upload.js';
import { setupVoiceControls } from './voice.js';
import { initDashboard, fetchProgressData } from './dashboard.js';

// UI Helper Functions
function log(msg) {
  const out = document.getElementById('output');
  if (out) {
    out.textContent += `\n${msg}`;
  }
  
  // Also add as a message
  addMessage('bot', msg);
}

function readValue(id) {
  const el = document.getElementById(id);
  return el ? el.value : '';
}

function showToast(message, type = '') {
  const toast = document.getElementById('toast');
  if (!toast) return;
  
  // Don't show toast if it's already visible (prevent multiple toasts)
  if (!toast.classList.contains('hidden')) {
    return;
  }
  
  toast.textContent = message;
  toast.className = 'toast ' + type;
  toast.classList.remove('hidden');
  
  // Force reflow to enable animation
  toast.offsetHeight;
  
  setTimeout(() => {
    toast.classList.add('hidden');
  }, 3000);
}

// Add message to conversation
function addMessage(type, content, options = {}) {
  const conversationEl = document.getElementById('conversation-messages');
  if (!conversationEl) return;
  
  // Create message elements
  const messageDiv = document.createElement('div');
  messageDiv.className = type === 'bot' ? 'message ai-message' : 'message user-message';
  // Use provided messageId for restoration, or generate new one
  const messageId = options.messageId || `msg-${Date.now()}-${Math.random().toString(36).slice(2,8)}`;
  messageDiv.dataset.messageId = messageId;
  
  const avatarDiv = document.createElement('div');
  avatarDiv.className = type === 'bot' ? 'avatar ai-avatar' : 'avatar user-avatar';
  avatarDiv.innerHTML = type === 'bot' ? '<i class="fas fa-robot"></i>' : '<i class="fas fa-user"></i>';
  
  const contentDiv = document.createElement('div');
  contentDiv.className = 'message-content';
  
  // Use markdown rendering if it's a bot message
  if (type === 'bot') {
    // Enhanced markdown rendering for headers, bold, lists, with better spacing
    let src = (content || '').replace(/\r\n/g, '\n');
    // Normalize numbered lists to ensure line breaks
    // Join wrapped numbered items properly
    src = src.replace(/\n\s*(\d+)\.\s+/g, '\n$1. ');
    src = src.replace(/(\n\s*\d+\.\s[^\n]+)(?!\n\d+\.\s|\n-\s|\n\*\s|\n\n)/g, '$1\n');
    // Ensure sub-section headers for practice levels render as bold labels
    src = src.replace(/\n\s*\d+\.\s*(Beginner|Intermediate|Advanced) level:/gi, '\n**$1 level:**');
    let formattedContent = formatMarkdownHtml(src);
    
    // Ensure content is wrapped in paragraphs with proper styling
    if (!formattedContent.startsWith('<')) {
      formattedContent = '<p style="margin: 1rem 0;">' + formattedContent + '</p>';
    }
    
    // Wrap rendered segment so we can replace on version navigation
    const renderedWrap = document.createElement('div');
    renderedWrap.className = 'rendered';
    renderedWrap.innerHTML = formattedContent;
    contentDiv.appendChild(renderedWrap);
    contentDiv.classList.add('markdown');

    // Meta timestamps
    const createdAt = options.createdAt || new Date();
    const meta = document.createElement('div');
    meta.className = 'msg-meta';
    const elapsed = (options.elapsedMs !== undefined) ? ` — ${Math.max(1, Math.round(options.elapsedMs/1000))}s` : '';
    meta.textContent = `LLM responded ${formatRelativeTime(createdAt)} (${formatTime(createdAt)})${elapsed}`;
    contentDiv.appendChild(meta);

    // Compact toolbar with actions and version nav
    const toolbar = document.createElement('div');
    toolbar.className = 'message-toolbar';
    let versionIndex = 1; // current version index
    let versions = [{ content }];
    
    // Load existing optimization versions from localStorage
    const savedVersions = loadOptimizationVersions(messageId);
    if (savedVersions && Array.isArray(savedVersions) && savedVersions.length > 0) {
      versions = savedVersions;
      console.log(`Loaded ${versions.length} saved versions for message ${messageId}`);
      
      // If we have saved versions, update the initial content display to show the first version
      // (the version restoration logic below will handle showing the correct selected version)
      if (versions.length > 1) {
        const html = formatMarkdownHtml(versions[0].content);
        contentDiv.querySelector('.rendered')?.remove();
        const wrap = document.createElement('div');
        wrap.className = 'rendered';
        wrap.innerHTML = html;
        contentDiv.insertBefore(wrap, toolbar);
      }
    }

    const onOptimize = async () => {
      try {
        startTopProgress();
        const res = await fetch('/optimize', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ sessionId: options.sessionId || null, text: content })
        });
        const data = await res.json();
        if (res.ok && data.optimized) {
          versions.push({ content: data.optimized });
          versionIndex = versions.length;
          // Replace current content display
          contentDiv.querySelector('.rendered')?.remove();
          const newWrap = document.createElement('div');
          newWrap.className = 'rendered';
          newWrap.innerHTML = formatMarkdownHtml(data.optimized);
          contentDiv.insertBefore(newWrap, toolbar);
          updateToolbarIndicator();
          saveVersionSelection(messageId, versionIndex);
          // Save all versions to localStorage for persistence
          saveOptimizationVersions(messageId, versions);
        } else {
          showToast('Failed to optimize', 'error');
        }
      } catch (e) {
        console.error('Optimize failed', e);
        showToast('Optimize failed', 'error');
      } finally {
        finishTopProgress();
      }
    };

    // Toolbar buttons
    const btnPrev = document.createElement('button');
    btnPrev.className = 'toolbar-btn';
    btnPrev.innerHTML = '<i class="fas fa-angle-left"></i>';
    btnPrev.title = 'Previous version';
    btnPrev.onclick = () => navigateVersion(-1);

    const indicator = document.createElement('span');
    indicator.className = 'toolbar-indicator';
    function updateToolbarIndicator() {
      indicator.textContent = `${versionIndex}/${versions.length}`;
    }
    updateToolbarIndicator();

    const btnNext = document.createElement('button');
    btnNext.className = 'toolbar-btn';
    btnNext.innerHTML = '<i class="fas fa-angle-right"></i>';
    btnNext.title = 'Next version';
    btnNext.onclick = () => navigateVersion(1);

    const btnOptimize = document.createElement('button');
    btnOptimize.className = 'toolbar-btn';
    btnOptimize.title = 'Optimize';
    btnOptimize.innerHTML = '<i class="fas fa-magic"></i>';
    btnOptimize.onclick = onOptimize;

    const btnCopy = document.createElement('button');
    btnCopy.className = 'toolbar-btn';
    btnCopy.title = 'Copy message';
    btnCopy.innerHTML = '<i class="fas fa-copy"></i>';
    btnCopy.onclick = () => navigator.clipboard.writeText((versions[versionIndex - 1].content || '').trim());

    const btnCopyId = document.createElement('button');
    btnCopyId.className = 'toolbar-btn';
    btnCopyId.title = 'Copy Session ID';
    btnCopyId.innerHTML = '<i class="fas fa-id-badge"></i>';
    btnCopyId.onclick = () => options.sessionId && navigator.clipboard.writeText(options.sessionId);

    function navigateVersion(delta) {
      const next = versionIndex + delta;
      if (next < 1 || next > versions.length) return;
      versionIndex = next;
      const html = formatMarkdownHtml(versions[versionIndex - 1].content);
      contentDiv.querySelector('.rendered')?.remove();
      const wrap = document.createElement('div');
      wrap.className = 'rendered';
      wrap.innerHTML = html;
      contentDiv.insertBefore(wrap, toolbar);
      updateToolbarIndicator();
      saveVersionSelection(messageId, versionIndex);
      // Save versions to ensure persistence
      saveOptimizationVersions(messageId, versions);
    }

    toolbar.appendChild(btnCopy);
    toolbar.appendChild(btnOptimize);
    toolbar.appendChild(btnPrev);
    toolbar.appendChild(indicator);
    toolbar.appendChild(btnNext);
    if (options.sessionId) toolbar.appendChild(btnCopyId);
    contentDiv.appendChild(toolbar);

    // Restore saved selection if exists
    const savedIdx = loadVersionSelection(messageId);
    if (savedIdx && savedIdx >= 1 && savedIdx <= versions.length && savedIdx !== versionIndex) {
      versionIndex = savedIdx;
      const html = formatMarkdownHtml(versions[versionIndex - 1].content);
      contentDiv.querySelector('.rendered')?.remove();
      const wrap = document.createElement('div');
      wrap.className = 'rendered';
      wrap.innerHTML = html;
      contentDiv.insertBefore(wrap, toolbar);
      updateToolbarIndicator();
    }
    
    // Add a speak button if TTS is enabled
    if (ttsState.enabled) {
      const speakBtn = document.createElement('button');
      speakBtn.className = 'speak-btn';
      speakBtn.innerHTML = '<i class="fas fa-volume-up"></i>';
      speakBtn.title = 'Read aloud';
      speakBtn.onclick = () => speakText(content);
      contentDiv.appendChild(speakBtn);
    }
    
    // Automatically speak if TTS is enabled
    if (ttsState.enabled) {
      // Use a slight delay to allow the UI to update first
      setTimeout(() => speakText(content), 500);
    }
  } else {
    const textWrap = document.createElement('div');
    textWrap.textContent = content;
    contentDiv.appendChild(textWrap);
    const meta = document.createElement('div');
    meta.className = 'msg-meta';
    const now = new Date();
    meta.textContent = `Sent ${formatRelativeTime(now)} (${formatTime(now)})`;
    contentDiv.appendChild(meta);
  }
  
  // Assemble message
  messageDiv.appendChild(avatarDiv);
  messageDiv.appendChild(contentDiv);
  
  // Add to conversation
  conversationEl.appendChild(messageDiv);
  
  // Scroll to bottom
  conversationEl.scrollTop = conversationEl.scrollHeight;

  // Persist minimal chat history to localStorage
  try {
    const entry = {
      id: messageId,
      type,
      content: String(content || ''),
      ts: Date.now()
    };
    const key = 'mz_chat_history';
    const maxItems = 50;
    const arr = JSON.parse(localStorage.getItem(key) || '[]');
    arr.push(entry);
    while (arr.length > maxItems) arr.shift();
    localStorage.setItem(key, JSON.stringify(arr));
  } catch {}
}

// Page Navigation
function setupNavigation() {
  const navButtons = document.querySelectorAll('.nav-btn');
  navButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetPage = btn.getAttribute('data-page');
      if (!targetPage) return;
      
      // Update active nav button
      navButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      
      // Show target page, hide others
      document.querySelectorAll('.page').forEach(page => {
        page.classList.remove('active');
      });
      const pageEl = document.getElementById(`${targetPage}-page`);
      if (pageEl) {
        pageEl.classList.add('active');
      }

      // Autofocus when Learn tab is activated
      if (targetPage === 'learn') {
        setTimeout(() => {
          const topic = document.querySelector('#learn-page input#topic') || document.querySelector('#learn-page .chat-input');
          if (topic) try { topic.focus(); } catch {}
        }, 40);
      }
    });
  });
}

// Mode Selection
function setupModeSelection() {
  // Using mode switch for selection
  const modeOptions = document.querySelectorAll('.mode-switch-option');
  const modeSelect = document.getElementById('mode');
  
  modeOptions.forEach(option => {
    option.addEventListener('click', () => {
      const mode = option.getAttribute('data-mode');
      if (!mode || !modeSelect) return;
      
      console.log('Mode selected:', mode); // Debug log
      
      // Update hidden select
      modeSelect.value = mode;
      
      // Update current mode in state
      appState.currentMode = mode;
      
      // Update UI for mode options
      modeOptions.forEach(opt => {
        if (opt.getAttribute('data-mode') === mode) {
          opt.classList.add('active');
        } else {
          opt.classList.remove('active');
        }
      });
    });
  });
}

// Mini stats panel toggle and render
function setupStatsPanel() {
  const btn = document.getElementById('statsToggle');
  const panel = document.getElementById('miniStats');
  if (!btn || !panel) return;
  let refreshTimer = null;
  const render = (data) => {
    const mastery = data?.mastery_level ?? 0;
    const accuracy = data?.accuracy ?? 0;
    const diff = data?.current_difficulty || 'beginner';
    const strat = data?.current_strategy || 'neural_compression';
    const attempts = data?.attempts ?? 0;
    const correct = data?.correct_answers ?? 0;
    const streak = data?.streak ?? 0;
    const topicsCompleted = data?.topics_completed ?? 0;
    const pct = Math.max(0, Math.min(100, mastery * 10));
    panel.innerHTML = `
      <div class="mini-stat-header">
        <div class="mini-stat-title">Live Learning Metrics</div>
        <div class="mini-stat-refresh">${new Date().toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'})}</div>
      </div>
      <div class="mini-progress"><div class="mini-progress-bar" style="width:${pct}%"></div></div>
      <div class="mini-stat-row"><div class="mini-stat-title">Mastery Level</div><div class="mini-stat-value">${mastery}/10</div></div>
      <div class="mini-stat-row"><div class="mini-stat-title">Accuracy</div><div class="mini-stat-value">${accuracy}%</div></div>
      <div class="mini-stat-row"><div class="mini-stat-title">Attempts</div><div class="mini-stat-value">${attempts} (${correct} correct) • Streak ${streak}</div></div>
      <div class="mini-stat-row"><div class="mini-stat-title">Topics Mastered</div><div class="mini-stat-value">${topicsCompleted}</div></div>
      <div class="mini-badges">
        <span class="mini-badge">Difficulty: ${diff}</span>
        <span class="mini-badge">Strategy: ${strat.replace(/_/g,' ')}</span>
      </div>
    `;
  };
  const fetchAndRender = async () => {
    try {
      const sid = appState.sessionId;
      if (!sid) { render(null); return; }
      const resp = await fetch(`/progress/${sid}`, { cache: 'no-store' });
      const data = resp.ok ? await resp.json() : null;
      render(data);
    } catch { render(null); }
  };
  btn.addEventListener('click', async () => {
    if (panel.classList.contains('open')) {
      panel.classList.remove('open');
      if (refreshTimer) { clearInterval(refreshTimer); refreshTimer = null; }
      return;
    }
    await fetchAndRender();
    panel.classList.add('open');
    refreshTimer = setInterval(fetchAndRender, 5000);
  });
}

// Topic Tags
function setupTopicTags() {
  const topicTags = document.querySelectorAll('.topic-tag');
  const topicInput = document.getElementById('topic');
  
  topicTags.forEach(tag => {
    tag.addEventListener('click', () => {
      if (topicInput) {
        topicInput.value = tag.textContent;
        topicInput.focus();
      }
    });
  });
}

// Info Button
function setupInfoButton() {
  const infoBtn = document.querySelector('.info-btn');
  const infoTooltip = document.querySelector('.info-tooltip');
  
  if (infoBtn && infoTooltip) {
    infoBtn.addEventListener('click', () => {
      infoTooltip.classList.toggle('hidden');
    });
  }
}

// Voice Controls
function setupVoiceUI() {
  const voiceContainer = document.getElementById('voice-container');
  if (voiceContainer) {
    setupVoiceControls(voiceContainer);
  }
}

// Global variable to store recognition instance
let activeRecognition = null;
let activeAutoLearnStream = null;

// Handle microphone button clicks
async function onMicClick(inputType) {
  const micBtn = document.getElementById(`mic${inputType.charAt(0).toUpperCase() + inputType.slice(1)}Btn`);
  const inputField = document.getElementById(inputType);
  
  if (!micBtn || !inputField) return;
  
  // Check if already recording
  if (activeRecognition) {
    // Stop current recording
    activeRecognition.stop();
    activeRecognition = null;
    
    // Reset all mic buttons
    document.querySelectorAll('.mic-btn').forEach(btn => {
      btn.classList.remove('active');
      btn.innerHTML = '<i class="fas fa-microphone"></i>';
      btn.title = 'Voice input';
    });
    
    return;
  }
  
  // Toggle active state
  const isActive = micBtn.classList.toggle('active');
  
  if (isActive) {
    // Start recording
    micBtn.innerHTML = '<i class="fas fa-microphone-slash"></i>';
    micBtn.title = 'Stop recording';
    
    try {
      // Check if browser supports speech recognition
      if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        const recognition = new SpeechRecognition();
        
        // Store recognition instance globally
        activeRecognition = recognition;
        
        recognition.lang = 'en-US';
        recognition.continuous = true;
        recognition.interimResults = true;
        
        // Store original value
        const originalValue = inputField.value;
        
        recognition.onresult = (event) => {
          // Get the latest result
          const resultIndex = event.results.length - 1;
          const transcript = event.results[resultIndex][0].transcript;
          
          // Update the input field with real-time transcription
          // Preserve any text that was already in the field before recording started
          if (originalValue) {
            inputField.value = originalValue + ' ' + transcript;
          } else {
            inputField.value = transcript;
          }
          
          // If this is a final result, add a space for the next word
          if (event.results[resultIndex].isFinal) {
            inputField.value += ' ';
          }
          
          // Trigger input event to update any dependent UI
          inputField.dispatchEvent(new Event('input'));
        };
        
        recognition.onend = () => {
          // Only reset if this is still the active recognition
          if (activeRecognition === recognition) {
            activeRecognition = null;
            micBtn.classList.remove('active');
            micBtn.innerHTML = '<i class="fas fa-microphone"></i>';
            micBtn.title = 'Voice input';
          }
        };
        
        recognition.onerror = (event) => {
          console.error('Speech recognition error:', event.error);
          showToast(`Voice input error: ${event.error}`, 'error');
          
          activeRecognition = null;
          micBtn.classList.remove('active');
          micBtn.innerHTML = '<i class="fas fa-microphone"></i>';
          micBtn.title = 'Voice input';
        };
        
        recognition.start();
        showToast('Listening... Speak now', '');
      } else {
        // Fallback to server-side voice recognition if available
        showToast('Using server-side voice recognition...', '');
        
        try {
          // Call server-side STT endpoint
          const response = await fetch('/stt', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ language: 'en' })
          });
          
          if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
          }
          
          const data = await response.json();
          if (data.text) {
            inputField.value = data.text;
          } else {
            throw new Error('No transcription returned');
          }
        } catch (error) {
          console.error('Server-side STT error:', error);
          showToast('Server-side voice recognition failed', 'error');
        } finally {
          // End recording
          activeRecognition = null;
          micBtn.classList.remove('active');
          micBtn.innerHTML = '<i class="fas fa-microphone"></i>';
          micBtn.title = 'Voice input';
        }
      }
    } catch (error) {
      console.error('Error with speech recognition:', error);
      showToast('Voice input failed. Please try again.', 'error');
      
      // Reset button
      activeRecognition = null;
      micBtn.classList.remove('active');
      micBtn.innerHTML = '<i class="fas fa-microphone"></i>';
      micBtn.title = 'Voice input';
    }
  } else {
    // Stop recording
    if (activeRecognition) {
      activeRecognition.stop();
      activeRecognition = null;
    }
    
    micBtn.innerHTML = '<i class="fas fa-microphone"></i>';
    micBtn.title = 'Voice input';
  }
}

// Loading state helpers
function setButtonLoading(buttonId, isLoading, loadingText = 'Processing...', originalHtml = null) {
  const button = document.getElementById(buttonId);
  if (!button) return;
  
  if (isLoading) {
    button.disabled = true;
    // Store original HTML if not provided
    if (!originalHtml) {
      button.dataset.originalHtml = button.innerHTML;
    }
    button.innerHTML = `<span class="loading-indicator"></span> ${loadingText}`;
  } else {
    button.disabled = false;
    button.innerHTML = originalHtml || button.dataset.originalHtml || button.innerHTML;
    delete button.dataset.originalHtml;
  }
}

// Global progress bar helpers
function startTopProgress() {
  const bar = document.getElementById('topProgress');
  if (!bar) return;
  bar.style.width = '10%';
  // Animate toward 80% while waiting
  let w = 10;
  bar._timer && clearInterval(bar._timer);
  bar._timer = setInterval(() => {
    w = Math.min(80, w + Math.random() * 10);
    bar.style.width = w + '%';
  }, 200);
}

function finishTopProgress() {
  const bar = document.getElementById('topProgress');
  if (!bar) return;
  if (bar._timer) clearInterval(bar._timer);
  bar.style.width = '100%';
  setTimeout(() => { bar.style.width = '0%'; }, 250);
}

// Expose progress helpers globally so other modules (upload/AZL) can use them
window.startTopProgress = startTopProgress;
window.finishTopProgress = finishTopProgress;

// Update UI elements based on the selected mode
function updateUIForMode(mode) {
  const answerInputContainer = document.querySelector('.answer-input-container');
  const topicInputContainer = document.querySelector('.topic-input-container');
  const voiceContainer = document.getElementById('voice-container');
  const answerInput = document.getElementById('answer');
  const submitBtn = document.getElementById('submitBtn');
  
  // Show answer input container if it's not already visible
  if (answerInputContainer) {
    answerInputContainer.classList.remove('hidden');
  }
  
  // Hide topic input container once conversation starts
  if (topicInputContainer) {
    topicInputContainer.classList.add('hidden');
  }
  
  if (mode === 'explain') {
    // In explain mode, change to generic query entry
    if (answerInput) answerInput.placeholder = 'Enter your query...';
    if (submitBtn) {
      submitBtn.innerHTML = '<i class="fas fa-question-circle"></i> Ask';
    }
    
    // Show copy session ID button
    const copyIdBtn = document.getElementById('copyIdBtn');
    if (copyIdBtn) {
      copyIdBtn.classList.remove('hidden');
    }
    
    // Hide the separate voice container in explain mode
    if (voiceContainer) voiceContainer.style.display = 'none';
  } else if (mode === 'quiz') {
    // In quiz mode, keep as answer submission
    if (answerInput) answerInput.placeholder = 'Your answer...';
    if (submitBtn) {
      submitBtn.innerHTML = '<i class="fas fa-paper-plane"></i> Submit';
    }
    
    // Show the voice container in quiz mode
    if (voiceContainer) voiceContainer.style.display = 'flex';
  }
}

// Main Actions
async function onStart() {
  // Reset session state to ensure we're starting fresh
  import('./state.js').then(({ resetSession }) => {
    resetSession();
  });
  
  // Use a default userId if not found
  const userId = readValue('userId') || 'default_user';
  
  // Get the current value directly from the input field
  const topicInput = document.getElementById('topic');
  
  // Get the topic from the input field
  let topic = '';
  if (topicInput) {
    topic = topicInput.value.trim();
    
    // Force clear any previous state to ensure a fresh query
    appState.lastTopic = '';
    
    // Clear input field immediately after reading its value
    if (topic) {
      console.log('Using topic:', topic);
      // Clear the input field immediately
      topicInput.value = '';
      // Also trigger input event to ensure UI updates
      topicInput.dispatchEvent(new Event('input', { bubbles: true }));
    }
  } else {
    console.warn('Topic input field not found');
  }

  // Safety net: if topic is empty, retry and/or fallback to answer field
  if (!topic) {
    const retryTopicEl = document.getElementById('topic');
    const retryTopic = (retryTopicEl && typeof retryTopicEl.value === 'string') ? retryTopicEl.value.trim() : '';
    if (retryTopic) {
      topic = retryTopic;
      console.log('Recovered topic from DOM on retry:', topic);
    } else {
      const fallbackAnswerEl = document.getElementById('answer');
      const fallbackAnswer = (fallbackAnswerEl && typeof fallbackAnswerEl.value === 'string') ? fallbackAnswerEl.value.trim() : '';
      if (fallbackAnswer) {
        topic = fallbackAnswer;
        console.log('Using fallback from answer field as topic:', topic);
      }
    }
  }
  
  // Get the mode from the hidden select element or use a default
  const modeSelect = document.getElementById('mode');
  const mode = modeSelect && modeSelect.value ? modeSelect.value : 'explain';
  // Removed noisy console log
  
  // Validate topic before proceeding
  if (!topic || topic.length === 0) {
    showToast('Please enter a topic or question to get started!', 'warning');
    // Refocus the topic input
    const topicEl = document.getElementById('topic');
    if (topicEl) topicEl.focus();
    return;
  }
  
  // Check if already processing
  if (appState.isProcessing) {
    showToast('Already processing a request, please wait...', 'warning');
    return;
  }

  // Disable chat area to prevent interactions during request
  document.querySelector('.chat-main')?.classList.add('chat-disabled');
  
  try {
    appState.isProcessing = true;
    setButtonLoading('startBtn', true, 'Connecting to LLM...');
    startTopProgress();
    const startAt = Date.now();
    
    // Clear previous conversation
    const conversationMessages = document.getElementById('conversation-messages');
    if (conversationMessages) {
      // Keep the output pre element but clear other messages
      const outputEl = document.getElementById('output');
      conversationMessages.innerHTML = '';
      if (outputEl) {
        conversationMessages.appendChild(outputEl);
        outputEl.textContent = 'Connecting to local LLM...\nThis may take a moment for the first request.';
      }
    }
    
    // Show answer input container
    const answerInputContainer = document.querySelector('.answer-input-container');
    if (answerInputContainer) {
      answerInputContainer.classList.remove('hidden');
    }
    
    // Hide topic input container
    const topicInputContainer = document.querySelector('.topic-input-container');
    if (topicInputContainer) {
      topicInputContainer.classList.add('hidden');
    }
    
    // Show copy session ID button
    const copyIdBtn = document.getElementById('copyIdBtn');
    if (copyIdBtn) {
      copyIdBtn.classList.remove('hidden');
    }
    
    // Show conversation-level thinking animation while first answer generates
    const startupThinkingId = addThinkingAnimation();
    
    // Send request
    console.log('Sending teach request:', { userId, topic, mode });
    const res = await startTeach({ userId, topic, mode });
    
    // Reset processing state
    appState.isProcessing = false;
    setButtonLoading('startBtn', false);
    finishTopProgress();
    
    if (!res.ok) {
      console.error('Error starting session:', res.data);
      
      // Handle specific error types
      let errorMessage = 'Failed to start session';
      if (res.status === 422) {
        errorMessage = 'Invalid request. Please check your input and try again.';
      } else if (res.data?.error) {
        errorMessage = res.data.error;
      }
      
      showToast(errorMessage, 'error');
      
      // Restore input UI so the user can try again with a fresh query
      const topicInputContainer = document.querySelector('.topic-input-container');
      if (topicInputContainer) topicInputContainer.classList.remove('hidden');
      const answerInputContainer = document.querySelector('.answer-input-container');
      if (answerInputContainer) answerInputContainer.classList.add('hidden');
      // Clear and refocus the topic input for a fresh entry
      const topicEl = document.getElementById('topic');
      if (topicEl) {
        topicEl.value = '';
        topicEl.focus();
      }
      return;
    }
    
    // Store session ID and mode
    if (res.data.session_id) {
      setSessionId(res.data.session_id);
      appState.currentMode = mode;
    }
    
    // Update output
    const output = document.getElementById('output');
    if (output) {
      output.textContent = '';
    }
    
    // Remove startup thinking animation
    removeThinkingAnimation(startupThinkingId);
    
    // Add user message to chat history and clear input for conversational feel
    if (topic) {
      addMessage('user', topic);
      const topicEl2 = document.getElementById('topic');
      if (topicEl2) topicEl2.value = '';
    }
    // Add first content only (hide meta)
    const elapsedMs = Date.now() - startAt;
    addMessage('bot', res.data.first || 'No content available', { sessionId: res.data.session_id, lastUserTopic: topic, createdAt: new Date(), elapsedMs });
    
    // Update UI based on mode
    updateUIForMode(mode);
    
    // Scroll to conversation
    const conversationSection = document.querySelector('.conversation-area');
    if (conversationSection) {
      conversationSection.scrollIntoView({ behavior: 'smooth' });
    }
    
    // Set up voice UI if not already done
    setupVoiceUI();
    
    // Now that we have a topic and session, initialize the dashboard
    initDashboard(topic);

    // Clear the topic field after a successful send to avoid stale reuse
    const topicEl = document.getElementById('topic');
    if (topicEl) topicEl.value = '';
  } catch (error) {
    console.error('Error starting session:', error);
    showToast('Failed to start session. Please try again.', 'error');
    
    // Reset processing state
    appState.isProcessing = false;
    setButtonLoading('startBtn', false);
    finishTopProgress();
  }
  // Re-enable chat area after completion
  document.querySelector('.chat-main')?.classList.remove('chat-disabled');
}

function formatTime(date) {
  try {
    const d = (date instanceof Date) ? date : new Date(date);
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  } catch { return ''; }
}

function formatRelativeTime(fromDate, nowDate = new Date()) {
  try {
    const from = (fromDate instanceof Date) ? fromDate : new Date(fromDate);
    const diffMs = nowDate - from;
    const rtf = new Intl.RelativeTimeFormat(undefined, { numeric: 'auto' });
    const seconds = Math.round(diffMs / 1000);
    if (Math.abs(seconds) < 60) return rtf.format(-seconds, 'second');
    const minutes = Math.round(seconds / 60);
    if (Math.abs(minutes) < 60) return rtf.format(-minutes, 'minute');
    const hours = Math.round(minutes / 60);
    if (Math.abs(hours) < 24) return rtf.format(-hours, 'hour');
    const days = Math.round(hours / 24);
    return rtf.format(-days, 'day');
  } catch { return ''; }
}

function saveVersionSelection(messageId, index) {
  try {
    const key = `msg_version_${messageId}`;
    localStorage.setItem(key, String(index));
  } catch {}
}

function loadVersionSelection(messageId) {
  try {
    const key = `msg_version_${messageId}`;
    const val = localStorage.getItem(key);
    return val ? parseInt(val, 10) : null;
  } catch { return null; }
}

function saveOptimizationVersions(messageId, versions) {
  try {
    const key = `msg_versions_${messageId}`;
    localStorage.setItem(key, JSON.stringify(versions));
  } catch {}
}

function loadOptimizationVersions(messageId) {
  try {
    const key = `msg_versions_${messageId}`;
    const val = localStorage.getItem(key);
    return val ? JSON.parse(val) : null;
  } catch { return null; }
}

function showClearChatModal() {
  const modal = document.getElementById('clearChatModal');
  if (modal) {
    modal.classList.remove('hidden');
  }
}

function hideClearChatModal() {
  const modal = document.getElementById('clearChatModal');
  if (modal) {
    modal.classList.add('hidden');
  }
}

function clearChatHistory() {
  try {
    // Clear conversation area
    const conversationEl = document.getElementById('conversation-messages');
    if (conversationEl) {
      conversationEl.innerHTML = '';
      // Re-add the welcome message
      const welcomeMsg = document.createElement('div');
      welcomeMsg.className = 'system-message';
      welcomeMsg.innerHTML = `
        <div class="message-content">
          <h3>Welcome to MentorZero</h3>
          <p>Your adaptive learning assistant. Ask me about any topic to start learning!</p>
        </div>
      `;
      conversationEl.appendChild(welcomeMsg);
    }
    
    // Clear localStorage data
    const keysToRemove = [];
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i);
      if (key && (key.startsWith('msg_version_') || key.startsWith('msg_versions_') || key === 'mz_chat_history')) {
        keysToRemove.push(key);
      }
    }
    
    keysToRemove.forEach(key => localStorage.removeItem(key));
    
    // Reset session state
    import('./state.js').then(({ resetSession }) => {
      resetSession();
    });
    
    // Show topic input and hide answer input
    const topicInputContainer = document.querySelector('.topic-input-container');
    const answerInputContainer = document.querySelector('.answer-input-container');
    if (topicInputContainer) topicInputContainer.classList.remove('hidden');
    if (answerInputContainer) answerInputContainer.classList.add('hidden');
    
    // Hide copy session ID button
    const copyIdBtn = document.getElementById('copyIdBtn');
    if (copyIdBtn) copyIdBtn.classList.add('hidden');
    
    // Focus on topic input
    const topicEl = document.getElementById('topic');
    if (topicEl) topicEl.focus();
    
    showToast('Chat history cleared successfully', 'success');
    console.log('Chat history cleared');
    
    // Close the modal
    hideClearChatModal();
    
  } catch (error) {
    console.error('Error clearing chat history:', error);
    showToast('Failed to clear chat history', 'error');
  }
}

async function onSubmit() {
  // If no session exists, create one automatically
  if (!appState.sessionId) {
    const userId = readValue('userId') || 'user1';
    const topic = readValue('topic') || readValue('answer');
    const mode = appState.currentMode || 'explain';
    
    if (!topic) {
      showToast('Please enter a topic or question', 'error');
      return;
    }
    
    // Start a session automatically with the current input
    await onStart();
    return;
  }
  
  const ans = readValue('answer');
  if (!ans) {
    showToast(appState.currentMode === 'explain' ? 'Please enter a question' : 'Please enter an answer', 'error');
    return;
  }
  
  // Always use the current answer text, never cached values
  console.log('Submitting current answer:', ans);
  
  // Check if already processing
  if (appState.isProcessing) {
    showToast('Already processing a request, please wait...', 'warning');
    return;
  }
  
  try {
    appState.isProcessing = true;
    setButtonLoading('submitBtn', true, 'Submitting...');
    startTopProgress();
    
    // Add user message
    addMessage('user', ans);
    
    // Add thinking animation
    const thinkingId = addThinkingAnimation();
    
    const res = await submitAnswer({ sessionId: appState.sessionId, userAnswer: ans });
    
    // Remove thinking animation
    removeThinkingAnimation(thinkingId);
    
    // Reset processing state
    appState.isProcessing = false;
    setButtonLoading('submitBtn', false);
    finishTopProgress();
    
    if (!res.ok) {
      showToast(`Error: ${res.data?.error || 'Request failed'}`, 'error');
      return;
    }
    
    // Add bot messages
    if (appState.currentMode === 'explain') {
      // For explain mode, just show the explanation
      const formattedExplanation = formatResponse(res.data.explanation);
      addMessage('bot', formattedExplanation);
    } else {
      // For quiz mode, show feedback and next question
      const formattedFeedback = formatResponse(`Feedback: ${res.data.explanation}`);
      const formattedNext = formatResponse(`Next: ${res.data.next_item}`);
      addMessage('bot', formattedFeedback);
      addMessage('bot', formattedNext);
    }
    
    // Clear answer field
    const answerField = document.getElementById('answer');
    if (answerField) {
      answerField.value = '';
      answerField.focus();
    }
    
    // Update progress data after submission
    const progressData = await fetchProgressData();
    if (progressData) {
      // Update UI with new progress data
      updateDashboardWithProgress(progressData);
    }
  } catch (error) {
    console.error('Error submitting answer:', error);
    showToast('Failed to submit answer. Please try again.', 'error');
    
    // Reset processing state
    appState.isProcessing = false;
    setButtonLoading('submitBtn', false);
    finishTopProgress();
  }
}

// Update dashboard with progress data
function updateDashboardWithProgress(progressData) {
  // Update stats
  const statsCards = document.querySelectorAll('.stat-card .stat-value');
  if (statsCards.length >= 3) {
    statsCards[1].textContent = progressData.topics_completed || '0';
    statsCards[2].textContent = `${progressData.accuracy || '0'}%`;
    
    // Update tooltip
    const accuracyCard = statsCards[2].closest('.stat-card');
    if (accuracyCard) {
      accuracyCard.title = `Mastery Level: ${progressData.mastery_level}/10\nCurrent Difficulty: ${progressData.current_difficulty}\nStrategy: ${progressData.current_strategy}\nStreak: ${progressData.streak}`;
    }
  }
  
  // Update learning path progress
  const progressFill = document.querySelector('.progress-fill');
  if (progressFill) {
    const masteryLevel = progressData.mastery_level || 0;
    const progressPercent = Math.max(10, Math.min(100, masteryLevel * 10));
    progressFill.style.width = `${progressPercent}%`;
  }
  
  // Update difficulty badge
  const difficultyBadge = document.querySelector('.difficulty-badge');
  if (difficultyBadge) {
    const currentDifficulty = progressData.current_difficulty || 'beginner';
    difficultyBadge.textContent = currentDifficulty;
    difficultyBadge.className = `difficulty-badge ${currentDifficulty}`;
  }
  
  // Update strategy badge
  const strategyBadge = document.querySelector('.strategy-badge');
  if (strategyBadge) {
    const currentStrategy = progressData.current_strategy || 'neural_compression';
    const strategyDisplay = currentStrategy
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
    strategyBadge.textContent = strategyDisplay;
  }
}

async function onUpload() {
  // Ensure we have a session ID; if not, create a lightweight one
  if (!appState.sessionId) {
    const sid = (window.crypto && crypto.randomUUID) ? crypto.randomUUID() : `sess-${Date.now()}`;
    try {
      setSessionId(sid);
    } catch {}
  }
  
  const text = readValue('uploadText');
  if (!text) {
    showToast('Please enter text to upload', 'error');
    return;
  }

  // Duplicate detection via content hash (localStorage)
  async function sha256(str) {
    const enc = new TextEncoder();
    const buf = await crypto.subtle.digest('SHA-256', enc.encode(str));
    return Array.from(new Uint8Array(buf)).map(b => b.toString(16).padStart(2, '0')).join('');
  }
  const hash = await sha256(text);
  const lastHash = localStorage.getItem('mz_last_upload_hash');
  if (lastHash === hash) {
    showToast('Same content as last upload – skipping ingest', 'warning');
    return;
  }
  
  // Check if already processing
  if (appState.isProcessing) {
    showToast('Already processing a request, please wait...', 'warning');
    return;
  }
  
  try {
    appState.isProcessing = true;
    setButtonLoading('uploadBtn', true, 'Uploading...');
    
    // Simulate progressive UI while backend processes
    const progress = document.getElementById('uploadProgress');
    const t0 = performance.now();
    const approx = Math.min(100, Math.max(10, Math.round(text.length / 2000)));
    let pct = 10;
    const timer = setInterval(() => {
      pct = Math.min(95, pct + Math.random() * 5);
      if (progress) progress.style.width = pct + '%';
    }, 120);

    const res = await uploadText({ sessionId: appState.sessionId, text });
    
    // Reset processing state
    appState.isProcessing = false;
    setButtonLoading('uploadBtn', false);
    
    if (!res.ok) {
      showToast(`Error: ${res.data?.error || 'Request failed'}`, 'error');
      clearInterval(timer);
      if (progress) progress.style.width = '0%';
      const status = document.getElementById('uploadStatus');
      const kpi = document.getElementById('uploadStatusKpi');
      if (status) status.textContent = 'Error';
      if (kpi) { kpi.classList.remove('status-ok'); kpi.classList.add('status-err'); }
      return;
    }
    
    clearInterval(timer);
    if (progress) progress.style.width = '100%';
    const t1 = performance.now();
    if (!appState.sessionId && res.data?.session_id) {
      try { setSessionId(res.data.session_id); } catch {}
    }
    // Update metrics
    const chunks = res.data.extracted_chunks_count || 0;
    const ms = Math.round(t1 - t0);
    const sizeKb = Math.max(1, Math.round((text.length / 1024)));
    const elC = document.getElementById('uploadChunks'); if (elC) elC.textContent = String(chunks);
    const elT = document.getElementById('uploadTime'); if (elT) elT.textContent = `${ms} ms`;
    const elS = document.getElementById('uploadSize'); if (elS) elS.textContent = `${sizeKb} KB`;
    const status = document.getElementById('uploadStatus');
    const kpi = document.getElementById('uploadStatusKpi');
    if (status) status.textContent = 'OK';
    if (kpi) { kpi.classList.remove('status-err'); kpi.classList.add('status-ok'); }
    showToast(`Indexed ${chunks} chunks in ${ms} ms`, 'success');

    // Persist last upload hash
    localStorage.setItem('mz_last_upload_hash', hash);
    
    const auto = document.getElementById('autoRerunTeach')?.checked === true;
    if (auto) {
      // Switch to learn page
      document.querySelectorAll('.nav-btn').forEach(btn => {
        if (btn.getAttribute('data-page') === 'learn') {
          btn.click();
        }
      });
      
      // Re-run teach
      const userId = readValue('userId');
      const topic = readValue('topic');
      const mode = document.querySelector('input[name="mode-radio"]:checked').value;
      
      // Show loading state
      appState.isProcessing = true;
      setButtonLoading('startBtn', true, 'Regenerating with new context...');
      
      const again = await startTeach({ userId, topic, mode });
      
      // Reset processing state
      appState.isProcessing = false;
      setButtonLoading('startBtn', false);
      
      if (again.ok) {
        // Clear previous conversation
        const conversationMessages = document.getElementById('conversation-messages');
        if (conversationMessages) {
          // Keep the output pre element but clear other messages
          const outputEl = document.getElementById('output');
          conversationMessages.innerHTML = '';
          if (outputEl) {
            conversationMessages.appendChild(outputEl);
            outputEl.textContent = '';
          }
        }
        
        // Add message
        addMessage('bot', again.data.first);
        
        // Update UI based on mode
        updateUIForMode(mode);
        
            // Make sure the answer input container is visible
    const answerInputContainer = document.querySelector('.answer-input-container');
    if (answerInputContainer) {
      answerInputContainer.classList.remove('hidden');
    }
    
    // Scroll to the bottom of the conversation
    const conversationArea = document.querySelector('.conversation-area');
    if (conversationArea) {
      conversationArea.scrollTop = conversationArea.scrollHeight;
    }
      }
    }
  } catch (error) {
    console.error('Error uploading text:', error);
    showToast('Failed to upload text. Please try again.', 'error');
    
    // Reset processing state
    appState.isProcessing = false;
    setButtonLoading('uploadBtn', false);
  }
}

function onCopyId() {
  const id = appState.sessionId;
  if (!id) {
    showToast('No active session', 'error');
    return;
  }
  
  navigator.clipboard?.writeText(id)
    .then(() => {
      showToast('Session ID copied to clipboard', 'success');
    })
    .catch(() => {
      showToast('Failed to copy session ID', 'error');
    });
}

// Reset conversation and UI
function onReset() {
  // Import resetSession from state.js
  import('./state.js').then(({ resetSession }) => {
    resetSession();
    
    // Clear conversation area
    const conversationMessages = document.getElementById('conversation-messages');
    if (conversationMessages) {
      conversationMessages.innerHTML = `
        <div class="system-message">
          <div class="message-content">
            <h3>Welcome to MentorZero</h3>
            <p>Your adaptive learning assistant. Ask me about any topic to start learning!</p>
          </div>
        </div>
      `;
    }
    
    // Show topic input container
    const topicInputContainer = document.querySelector('.topic-input-container');
    if (topicInputContainer) {
      topicInputContainer.classList.remove('hidden');
    }
    
    // Hide answer input container
    const answerInputContainer = document.querySelector('.answer-input-container');
    if (answerInputContainer) {
      answerInputContainer.classList.add('hidden');
    }
    
    // Hide session ID button
    const copyIdBtn = document.getElementById('copyIdBtn');
    if (copyIdBtn) {
      copyIdBtn.classList.add('hidden');
    }
    
    // Clear topic input
    const topicInput = document.getElementById('topic');
    if (topicInput) {
      topicInput.value = '';
      topicInput.focus();
    }
    
    showToast('Conversation reset', 'success');
  });
}

function onToggleTheme() {
  const htmlRoot = document.documentElement;
  const isDark = htmlRoot.classList.toggle('dark');
  
  const themeToggle = document.getElementById('themeToggle');
  if (themeToggle) {
    themeToggle.textContent = isDark ? '☀️' : '🌙';
  }
  
  try {
    localStorage.setItem('theme', isDark ? 'dark' : 'light');
  } catch(e) {
    console.error('Failed to save theme preference:', e);
  }
}

function loadSavedTheme() {
  try {
    const savedTheme = localStorage.getItem('theme');
    const htmlRoot = document.documentElement;
    const themeToggle = document.getElementById('themeToggle');
    
    if (savedTheme === 'dark') {
      htmlRoot.classList.add('dark');
      if (themeToggle) themeToggle.textContent = '☀️';
    } else {
      htmlRoot.classList.remove('dark');
      if (themeToggle) themeToggle.textContent = '🌙';
    }
  } catch(e) {
    console.error('Failed to load theme preference:', e);
  }
}

// Toggle stats panel visibility
function onToggleStats() {
  const statsPanel = document.getElementById('statsPanel');
  const statsToggle = document.getElementById('statsToggle');
  
  if (statsPanel && statsToggle) {
    const isHidden = statsPanel.classList.toggle('hidden');
    statsToggle.innerHTML = isHidden ? 
      '<i class="fas fa-chart-bar"></i> Show Learning Stats' : 
      '<i class="fas fa-chart-bar"></i> Hide Learning Stats';
  }
}

// Initialize AZL UI
function initAzlUI() {
  const proposeBtn = document.getElementById('proposeBtn');
  const acceptBtn = document.getElementById('acceptBtn');
  const rejectBtn = document.getElementById('rejectBtn');
  const autoBtn = document.getElementById('autolearnBtn');
  
  if (proposeBtn) {
    proposeBtn.addEventListener('click', onProposeExamples);
  }
  
  if (acceptBtn) {
    acceptBtn.addEventListener('click', () => onAcceptExample(true));
  }
  
  if (rejectBtn) {
    rejectBtn.addEventListener('click', () => onAcceptExample(false));
  }

  if (autoBtn) {
    autoBtn.addEventListener('click', onAutoLearn);
  }
  const closeBtn = document.getElementById('observerClose');
  if (closeBtn) {
    closeBtn.addEventListener('click', () => {
      document.getElementById('observerModal')?.classList.add('hidden');
      if (activeAutoLearnStream) { activeAutoLearnStream.close(); activeAutoLearnStream = null; }
    });
  }
  // Log copy/download
  const copyBtn = document.getElementById('observerLogCopy');
  const dlBtn = document.getElementById('observerLogDownload');
  const getLogText = () => {
    const t = document.getElementById('observer-timeline-modal');
    if (!t) return '';
    return [...t.querySelectorAll('.observer-event')].map(ev => {
      const badge = ev.querySelector('.ev-badge')?.textContent || '';
      const txt = ev.querySelector('span:nth-child(2)')?.textContent || '';
      return `${badge}: ${txt}`;
    }).join('\n');
  };
  if (copyBtn) {
    copyBtn.addEventListener('click', async () => {
      try { await navigator.clipboard.writeText(getLogText()); showToast('Observer log copied', 'success'); } catch { showToast('Copy failed', 'error'); }
    });
  }
  if (dlBtn) {
    dlBtn.addEventListener('click', () => {
      const blob = new Blob([getLogText()], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `autolearn_log_${Date.now()}.txt`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    });
  }
}

// Propose AZL examples
async function onProposeExamples() {
  const topic = document.getElementById('azlTopic')?.value;
  const count = document.getElementById('azlCount')?.value || 5;
  
  if (!topic) {
    showToast('Please enter a topic', 'error');
    return;
  }
  
  try {
    setButtonLoading('proposeBtn', true);
    startTopProgress();
    
    const response = await fetch('/azl/propose', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ topic, count: parseInt(count) })
    });
    
    if (!response.ok) {
      throw new Error(`Error: ${response.status}`);
    }
    
    const data = await response.json();
    displayExamples(data.examples, data.proposal_id);
    
  } catch (error) {
    console.error('Error proposing examples:', error);
    showToast('Failed to generate examples', 'error');
  } finally {
    setButtonLoading('proposeBtn', false);
    finishTopProgress();
  }
}

// Display AZL examples
function displayExamples(examples, proposalId) {
  const container = document.getElementById('examples-container');
  const examplesSection = document.getElementById('azl-examples');
  
  if (!container || !examplesSection) return;
  
  // Store proposal ID for later use
  container.dataset.proposalId = proposalId;
  
  // Clear previous examples
  container.innerHTML = '';
  
  // Add examples
  examples.forEach((example, index) => {
    const exampleDiv = document.createElement('div');
    exampleDiv.className = 'example-item';
    exampleDiv.dataset.index = index;
    
    const questionDiv = document.createElement('div');
    questionDiv.className = 'example-question';
    questionDiv.textContent = example.question;
    
    const answerDiv = document.createElement('div');
    answerDiv.className = 'example-answer';
    answerDiv.textContent = example.answer;
    
    const actions = document.createElement('div');
    actions.className = 'azl-example-actions';
    const validateBtn = document.createElement('button');
    validateBtn.className = 'btn btn-primary mt-2';
    validateBtn.title = 'Run automated checks';
    validateBtn.innerHTML = '<i class="fas fa-check-circle"></i> Validate';
    validateBtn.onclick = () => validateExample(proposalId, index);

    const regenBtn = document.createElement('button');
    regenBtn.className = 'btn btn-secondary mt-2';
    regenBtn.title = 'Regenerate this example';
    regenBtn.innerHTML = '<i class="fas fa-sync"></i> Regenerate';
    regenBtn.onclick = () => regenerateExample(proposalId, index);

    actions.appendChild(validateBtn);
    actions.appendChild(regenBtn);
    
    exampleDiv.appendChild(questionDiv);
    exampleDiv.appendChild(answerDiv);
    exampleDiv.appendChild(actions);
    container.appendChild(exampleDiv);
  });
  
  // Show examples section
  examplesSection.classList.remove('hidden');
}

// Validate AZL example
async function validateExample(proposalId, exampleIdx) {
  try {
    startTopProgress();
    const response = await fetch('/azl/validate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ proposal_id: proposalId, example_idx: exampleIdx })
    });
    
    if (!response.ok) {
      throw new Error(`Error: ${response.status}`);
    }
    
    const data = await response.json();
    displayValidationResults(data, proposalId, exampleIdx);
    
  } catch (error) {
    console.error('Error validating example:', error);
    showToast('Failed to validate example', 'error');
  } finally {
    finishTopProgress();
  }
}

// Display validation results
function displayValidationResults(results, proposalId, exampleIdx) {
  const container = document.getElementById('validation-container');
  const validationSection = document.getElementById('azl-validation');
  
  if (!container || !validationSection) return;
  
  // Store data for accept/reject
  container.dataset.proposalId = proposalId;
  container.dataset.exampleIdx = exampleIdx;
  
  // Clear previous results
  container.innerHTML = '';
  
  // Summary progress
  const entries = Object.entries(results.validation_results);
  const passedCount = entries.reduce((acc, [,v]) => acc + (v.passed ? 1 : 0), 0);
  const percent = Math.round((passedCount / Math.max(1, entries.length)) * 100);
  const summaryBar = document.createElement('div');
  summaryBar.innerHTML = `<div class="mini-progress"><div class="mini-progress-bar" style="width:${percent}%"></div></div>`;
  container.appendChild(summaryBar);
  
  // Add results
  entries.forEach(([key, value]) => {
    const resultDiv = document.createElement('div');
    resultDiv.className = 'validation-item';
    
    const nameSpan = document.createElement('span');
    nameSpan.textContent = key.charAt(0).toUpperCase() + key.slice(1);
    
    const resultSpan = document.createElement('span');
    resultSpan.className = value.passed ? 'validation-pass' : 'validation-fail';
    const icon = value.passed ? 'check' : 'times';
    resultSpan.innerHTML = `<i class="fas fa-${icon}"></i> ${value.passed ? 'Pass' : 'Fail'}`;
    const reason = value.reason || value.details?.explanation || JSON.stringify(value.details || {});
    if (reason && reason !== '{}' ) {
      resultSpan.title = reason;
    }
    
    resultDiv.appendChild(nameSpan);
    resultDiv.appendChild(resultSpan);
    container.appendChild(resultDiv);
  });
  
  // Add overall result
  const overallDiv = document.createElement('div');
  overallDiv.className = 'validation-item mt-2';
  overallDiv.style.fontWeight = 'bold';
  
  const overallName = document.createElement('span');
  overallName.textContent = 'Overall';
  
  const overallResult = document.createElement('span');
  overallResult.className = results.passed ? 'validation-pass' : 'validation-fail';
  overallResult.innerHTML = results.passed ? 
    '<i class="fas fa-check-circle"></i> Passed All Checks' : 
    '<i class="fas fa-exclamation-circle"></i> Failed Validation';
  
  overallDiv.appendChild(overallName);
  overallDiv.appendChild(overallResult);
  container.appendChild(overallDiv);
  
  // Show validation section
  validationSection.classList.remove('hidden');
}

// Regenerate a single AZL example
async function regenerateExample(proposalId, exampleIdx) {
  try {
    startTopProgress();
    const topic = document.getElementById('azlTopic')?.value || '';
    const response = await fetch('/azl/regenerate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ proposal_id: proposalId, example_idx: exampleIdx, topic })
    });
    if (!response.ok) throw new Error(`Error: ${response.status}`);
    const data = await response.json();
    const container = document.getElementById('examples-container');
    if (!container) return;
    // Update the example card in place
    const card = [...container.querySelectorAll('.example-item')].find(el => parseInt(el.dataset.index) === exampleIdx);
    if (card && data.example) {
      const q = card.querySelector('.example-question');
      const a = card.querySelector('.example-answer');
      if (q) q.textContent = data.example.question || q.textContent;
      if (a) a.textContent = data.example.answer || a.textContent;
      showToast('Example regenerated', 'success');
    }
  } catch (e) {
    console.error('Error regenerating example:', e);
    showToast('Failed to regenerate example', 'error');
  } finally {
    finishTopProgress();
  }
}

// Run automatic self-learning for current topic
async function onAutoLearn() {
  const topic = document.getElementById('azlTopic')?.value;
  const count = parseInt(document.getElementById('azlCount')?.value || '5', 10);
  if (!topic) { showToast('Please enter a topic', 'error'); return; }
  try {
    startTopProgress();
    // Open modal overlay
    const modal = document.getElementById('observerModal');
    const modalTimeline = document.getElementById('observer-timeline-modal');
    const status = document.getElementById('observerStatus');
    const progressBar = document.getElementById('observerProgress');
    const meta = document.getElementById('observerMeta');
    const inlineMeta = document.getElementById('observerInlineMeta');
    const kpiItems = document.getElementById('kpiItems');
    const kpiAccepted = document.getElementById('kpiAccepted');
    const kpiAttempts = document.getElementById('kpiAttempts');
    const kpiJudge = document.getElementById('kpiJudge');
    if (modal && modalTimeline && status && meta) {
      modal.classList.remove('hidden');
      modalTimeline.innerHTML = '';
      meta.textContent = '';
      if (inlineMeta) inlineMeta.textContent = '';
      status.style.visibility = 'visible';
      status.classList.remove('done','error');
      const text = status.querySelector('.run-text'); if (text) text.textContent = 'Running';
    }
    // Stream events
    // Include optional controls if provided
    const th = document.getElementById('azlThreshold')?.value;
    const ma = document.getElementById('azlMaxAttempts')?.value;
    const jm = document.getElementById('azlJudgeModel')?.value;
    const to = document.getElementById('azlTimeout')?.value;
    const db = document.getElementById('azlBudget')?.value;
    const jmarg = document.getElementById('azlJudgeMargin')?.value;
    const qs = new URLSearchParams({ topic, count: String(count) });
    if (th) qs.set('pass_threshold', th);
    if (ma) qs.set('max_attempts', ma);
    if (jm) qs.set('judge_model', jm);
    if (to) qs.set('timeout_seconds', to);
    if (db) qs.set('daily_budget', db);
    if (jmarg) qs.set('judge_margin', jmarg);
    const url = `/azl/autolearn_stream?${qs.toString()}`;
    if (activeAutoLearnStream) { activeAutoLearnStream.close(); }
    // SSE with auto-retry/backoff
    let retry = 0;
    const openStream = () => new EventSource(url);
    let es = openStream();
    activeAutoLearnStream = es;
    const counters = { items: 0, accepted: 0, attempts: 0, lastJudge: null, total: count, completed: 0 };
    const renderMeta = (prefixOnly=false) => {
      if (!meta) return;
      const prefix = meta.dataset.prefix || meta.textContent || '';
      if (prefixOnly) { meta.dataset.prefix = prefix; meta.textContent = prefix; return; }
      meta.textContent = `${prefix} • Items ${counters.items}, Accepted ${counters.accepted}, Attempts ${counters.attempts}`;
      if (inlineMeta) inlineMeta.textContent = meta.textContent;
      if (kpiItems) kpiItems.textContent = String(counters.items);
      if (kpiAccepted) kpiAccepted.textContent = String(counters.accepted);
      if (kpiAttempts) kpiAttempts.textContent = String(counters.attempts);
      if (kpiJudge) kpiJudge.textContent = counters.lastJudge ?? '–';
      if (progressBar && counters.total) {
        const pct = Math.min(100, Math.round((counters.completed / counters.total) * 100));
        progressBar.style.width = pct + '%';
      }
    };
    const add = (cls, text) => {
      if (!modalTimeline) return;
      const row = document.createElement('div');
      row.className = 'observer-event';
      const badge = document.createElement('span');
      badge.className = `ev-badge ${cls}`;
      badge.textContent = cls.replace('ev-','').toUpperCase();
      const content = document.createElement('span');
      content.textContent = text;
      row.appendChild(badge); row.appendChild(content);
      modalTimeline.appendChild(row);
      modalTimeline.scrollTop = modalTimeline.scrollHeight;
    };
    es.addEventListener('start', e => {
      try {
        const d = JSON.parse(e.data);
        if (meta) {
          const prefix = `Threshold ${d.pass_threshold}, weights ${JSON.stringify(d.weights)}${d.judge_model?`, Judge: ${d.judge_model}`:''}${d.max_attempts?`, max_attempts ${d.max_attempts}`:''}`;
          meta.textContent = prefix;
          meta.dataset.prefix = prefix;
        }
        // Initialize totals from backend
        if (typeof d.count === 'number') counters.total = d.count;
        // Reset progress per run
        if (progressBar) progressBar.style.width = '0%';
        counters.completed = 0;
        counters.items = 0; counters.accepted = 0; counters.attempts = 0;
      } catch {}
      add('ev-start', `Start topic '${topic}', count ${count}`);
      renderMeta(true);
    });
    es.addEventListener('proposed', e => { add('ev-start', 'Proposed examples'); renderMeta(); });
    es.addEventListener('item', e => {
      try {
        const d = JSON.parse(e.data);
        counters.items += 1;
        // Reset per-item progress
        if (progressBar) progressBar.style.width = '0%';
        add('ev-item', `Item ${d.index+1}: ${d.question}`);
        renderMeta();
      } catch {}
    });
    es.addEventListener('attempt', e => { try { const d = JSON.parse(e.data); counters.attempts += 1; add('ev-start', `Attempt ${d.attempt} for item ${d.index+1}`); if (progressBar && (d.attempt||0) && (d.max_attempts||0)) { const pct = Math.min(100, Math.round((d.attempt / d.max_attempts) * 100)); progressBar.style.width = pct + '%'; } renderMeta(); } catch {} });
    es.addEventListener('regen_attempt', e => { try { const d = JSON.parse(e.data); add('ev-regen', `Regenerating item ${d.index+1} (attempt ${d.attempt})`); } catch {} });
    es.addEventListener('score', e => {
      try {
        const d = JSON.parse(e.data);
        const s = d.score;
        counters.lastJudge = d.judge && typeof d.judge.correctness === 'number'
          ? `${(d.judge.correctness*100|0)}/${(d.judge.clarity*100|0)}/${(d.judge.usefulness*100|0)}`
          : counters.lastJudge;
        const dur = (typeof d.ms === 'number') ? ` (${d.ms} ms)` : '';
        add('ev-score', `Item ${d.index+1} attempt ${d.attempt}: score ${s.toFixed ? s.toFixed(2) : s}${dur}`);
        if (d.checks) {
          const names = Object.keys(d.checks);
          names.forEach(name => {
            const c = d.checks[name];
            const reason = c?.details?.explanation || c?.details?.error || c?.reason || 'N/A';
            add(c.passed ? 'ev-score' : 'ev-error', `  ${name}: ${c.passed ? 'pass' : 'fail'} (reason: ${reason})`);
          });
        }
        if (d.judge && d.judge.rationale) {
          add('ev-score', `  Judge Rationale: ${d.judge.rationale}`);
        }
        renderMeta();
      } catch {}
    });
    es.addEventListener('regenerated', e => {
      try { const d = JSON.parse(e.data); const dur = (typeof d.ms === 'number') ? ` (${d.ms} ms)` : ''; add('ev-regen', `Item ${d.index+1} regenerated${dur}`); } catch {}
    });
    es.addEventListener('regen_fail', e => { try { const d = JSON.parse(e.data); const bits = []; if (d.reason) bits.push(`reason: ${d.reason}`); if (typeof d.ms === 'number') bits.push(`${d.ms} ms`); if (d.snippet) bits.push(`snippet: ${d.snippet}`); const extra = bits.length ? ` — ${bits.join(' | ')}` : ''; add('ev-error', `Regeneration failed for item ${d.index+1} (attempt ${d.attempt})${extra}`); renderMeta(); } catch { add('ev-error', 'Regeneration failed'); } });
    es.addEventListener('accepted', e => { try { const d = JSON.parse(e.data); counters.accepted += 1; counters.completed += 1; if (progressBar) progressBar.style.width = '100%'; add('ev-accepted', `Accepted item ${d.index+1} in ${d.attempts} attempts (score ${d.score.toFixed ? d.score.toFixed(2) : d.score})`); renderMeta(); } catch {} });
    es.addEventListener('failed', e => { try { const d = JSON.parse(e.data); counters.completed += 1; if (progressBar) progressBar.style.width = '100%'; add('ev-error', `Failed item ${d.index+1} after ${d.attempts} attempts (best ${d.best_score?.toFixed ? d.best_score.toFixed(2) : d.best_score})`); if (d.checks) { const names = Object.keys(d.checks); names.forEach(name => { const c = d.checks[name]; const reason = c?.details?.explanation || c?.details?.error || c?.reason || 'N/A'; add(c.passed ? 'ev-score' : 'ev-error', `  ${name}: ${c.passed ? 'pass' : 'fail'} (reason: ${reason})`); }); } if (d.judge && d.judge.rationale) { add('ev-error', `  Judge Rationale: ${d.judge.rationale}`); } renderMeta(); } catch {} });
    es.addEventListener('done', e => {
      try { const d = JSON.parse(e.data); add('ev-done', `Done: accepted ${d.accepted}, failed ${d.failed}`); showToast(`AutoLearn accepted ${d.accepted}, failed ${d.failed}`, 'success'); renderMeta(); } catch {}
      if (status) { status.style.visibility = 'visible'; status.classList.add('done'); const text = status.querySelector('.run-text'); if (text) text.textContent = 'Done'; }
      es.close();
      activeAutoLearnStream = null;
    });
    const handleError = (label, errorData = null) => {
      // Enhanced error display with recovery suggestions
      if (errorData && errorData.recovery_suggestions) {
        add('ev-error', label);
        add('ev-error', `💡 Suggestions:`);
        errorData.recovery_suggestions.forEach((suggestion, idx) => {
          add('ev-error', `  ${idx + 1}. ${suggestion}`);
        });
      } else {
        add('ev-error', label);
      }
      
      if (status) { 
        status.style.visibility = 'visible'; 
        status.classList.add('error'); 
        const text = status.querySelector('.run-text'); 
        if (text) text.textContent = 'Error'; 
      }
      
      try { es.close(); } catch {}
      activeAutoLearnStream = null;
      
      // Auto-retry with capped exponential backoff up to 3 attempts
      if (retry < 3) {
        retry += 1;
        const delay = Math.min(1000 * Math.pow(2, retry), 8000);
        add('ev-start', `🔄 Retrying stream (attempt ${retry}) in ${delay}ms`);
        setTimeout(() => {
          es = openStream();
          activeAutoLearnStream = es;
          // Re-bind essential handlers for retry
          es.addEventListener('error', () => handleError('Stream connection error'));
          es.addEventListener('backend_error', (e) => {
            try { 
              const d = JSON.parse(e.data); 
              const errorData = d.error_details || null;
              add('ev-error', `❌ Backend Error: ${d.message}`);
              if (errorData && errorData.recovery_suggestions) {
                add('ev-error', `💡 Recovery suggestions:`);
                errorData.recovery_suggestions.forEach((suggestion, idx) => {
                  add('ev-error', `  ${idx + 1}. ${suggestion}`);
                });
              }
            } catch { 
              add('ev-error', '❌ Unknown backend error'); 
            }
            handleError('Backend error');
          });
        }, delay);
      }
    };
    es.addEventListener('backend_error', (e) => {
      try { const d = JSON.parse(e.data); add('ev-error', `Error: ${d.message}`); } catch { add('ev-error', 'Error'); }
      handleError('Backend error');
    });
    es.addEventListener('error', () => handleError('Stream error'));
  } catch (e) {
    console.error('AutoLearn error:', e);
    showToast('AutoLearn failed', 'error');
  } finally {
    finishTopProgress();
  }
}

// Accept or reject example
async function onAcceptExample(accept) {
  const container = document.getElementById('validation-container');
  
  if (!container) return;
  
  const proposalId = container.dataset.proposalId;
  const exampleIdx = container.dataset.exampleIdx;
  const userId = document.getElementById('userId')?.value || 'default';
  
  if (!proposalId || !exampleIdx) {
    showToast('No example selected', 'error');
    return;
  }
  
  try {
    setButtonLoading(accept ? 'acceptBtn' : 'rejectBtn', true);
    startTopProgress();
    
    const response = await fetch('/azl/accept', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        proposal_id: proposalId,
        example_idx: parseInt(exampleIdx),
        accepted: accept,
        message: `Reviewed by ${userId}`
      })
    });
    
    if (!response.ok) {
      throw new Error(`Error: ${response.status}`);
    }
    
    showToast(
      accept ? 'Example accepted and added to knowledge base' : 'Example rejected',
      accept ? 'success' : ''
    );
    
    // Hide validation section
    const validationSection = document.getElementById('azl-validation');
    if (validationSection) {
      validationSection.classList.add('hidden');
    }
    
    // Mark example as processed
    const examples = document.querySelectorAll('.example-item');
    examples.forEach(example => {
      if (example.dataset.index === exampleIdx) {
        example.classList.add(accept ? 'example-accepted' : 'example-rejected');
        const validateBtn = example.querySelector('button');
        if (validateBtn) {
          validateBtn.disabled = true;
          validateBtn.textContent = accept ? 'Accepted' : 'Rejected';
        }
      }
    });
    
  } catch (error) {
    console.error('Error accepting/rejecting example:', error);
    showToast(`Failed to ${accept ? 'accept' : 'reject'} example`, 'error');
  } finally {
    setButtonLoading(accept ? 'acceptBtn' : 'rejectBtn', false);
    finishTopProgress();
  }
}

// Format response to ensure proper line breaks for numbered lists
function formatResponse(text) {
  // Fix numbered lists by ensuring each list item is on a new line
  return text.replace(/(\d+\.)\s+([^\n])/g, '$1 $2\n');
}

// Add thinking animation to conversation
function addThinkingAnimation() {
  const conversationEl = document.getElementById('conversation-messages');
  if (!conversationEl) return null;
  
  // Create thinking animation container
  const thinkingDiv = document.createElement('div');
  thinkingDiv.className = 'thinking-dots';
  thinkingDiv.id = 'thinking-' + Date.now(); // Unique ID
  
  // Add dots
  for (let i = 0; i < 3; i++) {
    const dot = document.createElement('div');
    dot.className = 'dot';
    thinkingDiv.appendChild(dot);
  }
  
  // Add to conversation
  conversationEl.appendChild(thinkingDiv);
  
  // Scroll to bottom
  conversationEl.scrollTop = conversationEl.scrollHeight;
  
  return thinkingDiv.id;
}

// Remove thinking animation
function removeThinkingAnimation(thinkingId) {
  if (!thinkingId) return;
  
  const thinkingDiv = document.getElementById(thinkingId);
  if (thinkingDiv) {
    thinkingDiv.remove();
  }
}

// TTS state
const ttsState = {
  enabled: false,
  speaking: false,
  utterance: null
};

// Toggle TTS functionality
function onToggleTTS() {
  const ttsToggle = document.getElementById('ttsToggle');
  if (!ttsToggle) return;
  
  // Toggle state
  ttsState.enabled = !ttsState.enabled;
  
  // Update button appearance
  if (ttsState.enabled) {
    ttsToggle.classList.add('active');
    showToast('Text-to-speech enabled', 'success');
    
    // Check if speech synthesis is available
    if (!('speechSynthesis' in window)) {
      showToast('Speech synthesis not supported in this browser', 'warning');
    }
  } else {
    ttsToggle.classList.remove('active');
    showToast('Text-to-speech disabled', '');
    
    // Stop any ongoing speech
    if (ttsState.speaking) {
      window.speechSynthesis.cancel();
      ttsState.speaking = false;
    }
  }
  
  // Save preference
  try {
    localStorage.setItem('tts_enabled', ttsState.enabled ? 'true' : 'false');
  } catch (e) {
    console.error('Failed to save TTS preference:', e);
  }
}

// Load saved TTS preference
function loadSavedTTSPreference() {
  try {
    const savedPreference = localStorage.getItem('tts_enabled');
    const ttsToggle = document.getElementById('ttsToggle');
    
    if (savedPreference === 'true' && ttsToggle) {
      ttsState.enabled = true;
      ttsToggle.classList.add('active');
    }
  } catch (e) {
    console.error('Failed to load TTS preference:', e);
  }
}

// Speak text using browser's speech synthesis
async function speakText(text) {
  // Don't speak if TTS is disabled
  if (!ttsState.enabled) return;
  
  // Don't speak if speech synthesis is not available
  if (!('speechSynthesis' in window)) {
    console.warn('Speech synthesis not supported');
    return;
  }
  
  // Cancel any ongoing speech
  if (ttsState.speaking) {
    window.speechSynthesis.cancel();
  }
  
  // Create a new utterance
  const utterance = new SpeechSynthesisUtterance(text);
  ttsState.utterance = utterance;
  
  // Set properties
  utterance.lang = 'en-US';
  utterance.rate = 1.0;
  utterance.pitch = 1.0;
  
  // Try to find a good voice
  const voices = window.speechSynthesis.getVoices();
  if (voices.length > 0) {
    // Prefer a female voice if available
    const femaleVoice = voices.find(voice => 
      voice.name.includes('female') || 
      voice.name.includes('Samantha') || 
      voice.name.includes('Google UK English Female'));
    
    if (femaleVoice) {
      utterance.voice = femaleVoice;
    }
  }
  
  // Set up events
  utterance.onstart = () => {
    ttsState.speaking = true;
  };
  
  utterance.onend = () => {
    ttsState.speaking = false;
  };
  
  utterance.onerror = (event) => {
    console.error('Speech synthesis error:', event);
    ttsState.speaking = false;
  };
  
  // Try browser TTS first
  try {
    window.speechSynthesis.speak(utterance);
  } catch (error) {
    console.error('Browser TTS failed, trying server TTS:', error);
    
    // Fall back to server-side TTS if available
    try {
      const response = await fetch('/tts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, voice: 'default' })
      });
      
      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }
      
      const data = await response.json();
      if (data.audio_path) {
        // Play the audio
        const audio = new Audio(data.audio_path);
        audio.play();
      } else {
        throw new Error('No audio path returned');
      }
    } catch (serverError) {
      console.error('Server-side TTS failed:', serverError);
      showToast('Text-to-speech failed', 'error');
    }
  }
}

// Initialize everything when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
  // Ensure hidden, immutable userId
  try {
    let uid = localStorage.getItem('mz_user_id');
    if (!uid) {
      uid = `u_${Date.now().toString(36)}_${Math.random().toString(36).slice(2,8)}`;
      localStorage.setItem('mz_user_id', uid);
    }
    const userInput = document.getElementById('userId');
    if (userInput) {
      userInput.value = uid;
    }
  } catch {}
  // Set up theme toggle
  const themeToggle = document.getElementById('themeToggle');
  if (themeToggle) {
    themeToggle.addEventListener('click', onToggleTheme);
  }
  
  // Set up TTS toggle
  const ttsToggle = document.getElementById('ttsToggle');
  if (ttsToggle) {
    ttsToggle.addEventListener('click', onToggleTTS);
  }

  // Poll health indicators
  const llmDot = document.getElementById('healthLLM');
  const judgeDot = document.getElementById('healthJudge');
  const pollHealth = async () => {
    try {
      const [llm, judge] = await Promise.all([
        fetch('/llm_health', { cache: 'no-store' }).then(r => r.json()).catch(() => null),
        fetch('/judge_health', { cache: 'no-store' }).then(r => r.json()).catch(() => null)
      ]);
      if (llmDot && llm) {
        llmDot.classList.toggle('ready', !!llm.ready);
        llmDot.classList.toggle('down', !llm.ready);
        llmDot.title = `LLM: ${llm.model || ''} @ ${llm.host || ''} — ${llm.ready ? 'ready' : 'down'}`;
      }
      if (judgeDot && judge) {
        judgeDot.classList.toggle('ready', !!judge.ready);
        judgeDot.classList.toggle('down', !judge.ready);
        judgeDot.title = `Judge: ${judge.model || ''} @ ${judge.host || ''} — ${judge.ready ? 'ready' : 'down'}`;
      }
    } catch {}
  };
  pollHealth();
  setInterval(pollHealth, 30000); // Poll every 30 seconds instead of 8
  
  // Set up stats toggle
  const statsToggle = document.getElementById('statsToggle');
  if (statsToggle) {
    statsToggle.addEventListener('click', onToggleStats);
  }
  
  // Load saved preferences
  loadSavedTheme();
  loadSavedTTSPreference();
  
  // Initialize AZL UI
  initAzlUI();
  
  // Set up event listeners
  document.getElementById('startBtn')?.addEventListener('click', onStart);
  document.getElementById('submitBtn')?.addEventListener('click', onSubmit);
  document.getElementById('uploadBtn')?.addEventListener('click', onUpload);
  document.getElementById('scrapeBtn')?.addEventListener('click', onScrape);
  document.getElementById('clearChatBtn')?.addEventListener('click', showClearChatModal);

  // Upload panel coexistence: when Upload Knowledge (details) opens, hide chat area; when closed, show chat area
  const uploadDetails = document.querySelector('#learn-page details.clean-details');
  if (uploadDetails) {
    const chatArea = document.querySelector('#learn-page .chat-layout');
    // Our upload card is inside the details, so we only hide the conversation + input area, not the side bar
    const convo = document.querySelector('#learn-page .conversation-area');
    const inputArea = document.querySelector('#learn-page .chat-input-area');
    uploadDetails.addEventListener('toggle', () => {
      const open = uploadDetails.open;
      if (convo) convo.style.display = open ? 'none' : '';
      if (inputArea) inputArea.style.display = open ? 'none' : '';
    });
  }
  
  // Modal event listeners
  document.getElementById('closeModalBtn')?.addEventListener('click', hideClearChatModal);
  document.getElementById('cancelClearBtn')?.addEventListener('click', hideClearChatModal);
  document.getElementById('confirmClearBtn')?.addEventListener('click', clearChatHistory);
  
  // Close modal when clicking outside
  document.getElementById('clearChatModal')?.addEventListener('click', (e) => {
    if (e.target.id === 'clearChatModal') {
      hideClearChatModal();
    }
  });
  
  // Close modal with Escape key
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      const modal = document.getElementById('clearChatModal');
      if (modal && !modal.classList.contains('hidden')) {
        hideClearChatModal();
      }
    }
  });
  
  // Drag-and-drop upload to textarea
  const dz = document.getElementById('uploadDrop');
  const ta = document.getElementById('uploadText');
  if (dz && ta) {
    const prevent = (e) => { e.preventDefault(); e.stopPropagation(); };
    ['dragenter','dragover','dragleave','drop'].forEach(ev => dz.addEventListener(ev, prevent));
    dz.addEventListener('drop', async (e) => {
      const files = e.dataTransfer?.files;
      if (!files || files.length === 0) return;
      const file = files[0];
      const text = await file.text();
      ta.value = text;
      ta.dispatchEvent(new Event('input'));
    });
  }
  // Clicking dropzone opens native file picker
  const fileInput = document.getElementById('uploadFile');
  if (dz && fileInput && ta) {
    dz.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', async () => {
      const f = fileInput.files && fileInput.files[0];
      if (!f) return;
      const text = await f.text();
      ta.value = text;
      ta.dispatchEvent(new Event('input'));
      showToast(`Loaded ${f.name}`, 'info');
    });
  }
  document.getElementById('copyIdBtn')?.addEventListener('click', onCopyId);
  // Reset button removed as per user request
  
  // Set up microphone buttons
  document.getElementById('micTopicBtn')?.addEventListener('click', () => onMicClick('topic'));
  document.getElementById('micAnswerBtn')?.addEventListener('click', () => onMicClick('answer'));
  
  // Enter key in answer field
  document.getElementById('answer')?.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
      onSubmit();
    }
  });
  
  // Enter key in topic field
  document.getElementById('topic')?.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
      onStart();
    }
  });
  
  // Set up UI components
  setupNavigation();
  setupModeSelection();
  setupTopicTags();
  setupInfoButton();
  
  // Advanced RAG options (MMR/top-k + rewrite cache TTL)
  window.MZ_RAG = window.MZ_RAG || { topK: 5, mmr: 0.65, rewriteTTL: 300 };
  
  // Initialize UI based on default mode
  updateUIForMode(appState.currentMode);
  // Setup mini stats panel
  setupStatsPanel();

  // Restore chat history from localStorage
  try {
    const key = 'mz_chat_history';
    const arr = JSON.parse(localStorage.getItem(key) || '[]');
    if (Array.isArray(arr) && arr.length) {
      const conversationEl = document.getElementById('conversation-messages');
      if (conversationEl) {
        conversationEl.innerHTML = '';
      }
      for (const m of arr) {
        addMessage(m.type === 'user' ? 'user' : 'bot', m.content || '', { messageId: m.id });
      }
    }
  } catch {}

  // Keyboard shortcuts
  document.addEventListener('keydown', (e) => {
    // Alt+L -> theme toggle
    if (e.altKey && (e.key === 'l' || e.key === 'L')) {
      e.preventDefault();
      onToggleTheme();
      return;
    }
    // Alt+T -> cycle tabs
    if (e.altKey && (e.key === 't' || e.key === 'T')) {
      e.preventDefault();
      const order = ['home','learn','about','azl'];
      const activeIdx = order.findIndex(p => document.getElementById(`${p}-page`)?.classList.contains('active'));
      const next = order[(activeIdx + 1) % order.length];
      document.querySelectorAll('.nav-btn').forEach(btn => {
        if (btn.getAttribute('data-page') === next) btn.click();
      });
      return;
    }
    // '/' focus input on Learn (robust + logs)
    if (!e.altKey && !e.ctrlKey && !e.metaKey && e.key === '/') {
      const tag = (e.target && e.target.tagName) ? e.target.tagName.toLowerCase() : '';
      const editing = (e.target && (e.target.isContentEditable || tag === 'input' || tag === 'textarea'));
      if (editing) return;
      e.preventDefault();
      const ensureFocus = () => {
        const topic = document.querySelector('#learn-page input#topic') || document.querySelector('#learn-page .chat-input');
        if (topic && typeof topic.focus === 'function') {
          topic.focus();
          try { if (typeof topic.select === 'function') topic.select(); } catch {}
          console.debug("Focused Learn input via '/'");
        } else {
          console.debug("Learn input not found for '/'");
        }
      };
      const learnPage = document.getElementById('learn-page');
      if (!learnPage?.classList.contains('active')) {
        const btn = Array.from(document.querySelectorAll('.nav-btn')).find(b => b.getAttribute('data-page') === 'learn');
        if (btn) btn.click();
        setTimeout(ensureFocus, 60);
      } else {
        ensureFocus();
      }
    }
  });

  // Capture-phase listener to guarantee '/' focus works even if other handlers stop propagation
  const focusSlashHandler = (e) => {
    if (e.key !== '/' || e.altKey || e.ctrlKey || e.metaKey) return;
    const tag = (e.target && e.target.tagName) ? e.target.tagName.toLowerCase() : '';
    const editing = (e.target && (e.target.isContentEditable || tag === 'input' || tag === 'textarea'));
    if (editing) return;
    e.preventDefault();
    const ensureFocus = () => {
      const topic = document.querySelector('#learn-page input#topic') || document.querySelector('#learn-page .chat-input');
      if (topic && typeof topic.focus === 'function') {
        topic.focus();
        try { if (typeof topic.select === 'function') topic.select(); } catch {}
        console.debug("Focused Learn input via capture '/'");
      }
    };
    const learnPage = document.getElementById('learn-page');
    if (!learnPage?.classList.contains('active')) {
      const btn = Array.from(document.querySelectorAll('.nav-btn')).find(b => b.getAttribute('data-page') === 'learn');
      if (btn) btn.click();
      setTimeout(ensureFocus, 60);
    } else {
      ensureFocus();
    }
  };
  window.addEventListener('keydown', focusSlashHandler, true);
  window.focusLearnInput = () => focusSlashHandler(new KeyboardEvent('keydown', { key: '/' }));

  // Autofocus Learn on page load if Learn is active
  if (document.getElementById('learn-page')?.classList.contains('active')) {
    setTimeout(() => {
      const topic = document.querySelector('#learn-page input#topic') || document.querySelector('#learn-page .chat-input');
      if (topic) try { topic.focus(); } catch {}
    }, 50);
  }
});

async function onScrape() {
  const url = readValue('uploadUrl');
  if (!url) { showToast('Enter a URL to fetch', 'error'); return; }
  if (!appState.sessionId) {
    const sid = (window.crypto && crypto.randomUUID) ? crypto.randomUUID() : `sess-${Date.now()}`;
    try { setSessionId(sid); } catch {}
  }
  try {
    setButtonLoading('scrapeBtn', true, 'Fetching...');
    const res = await scrapeFetch({ sessionId: appState.sessionId, url });
    setButtonLoading('scrapeBtn', false);
    if (!res.ok) { showToast(res.data?.error || 'Fetch failed', 'error'); return; }
    if (!appState.sessionId && res.data?.session_id) {
      try { setSessionId(res.data.session_id); } catch {}
    }
    const chunks = res.data.extracted_chunks_count || 0;
    showToast(`Fetched and indexed ${chunks} chunks`, 'success');
  } catch (e) {
    setButtonLoading('scrapeBtn', false);
    showToast('Fetch failed', 'error');
  }
}
