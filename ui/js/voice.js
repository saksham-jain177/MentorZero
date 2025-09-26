// Voice control functionality
import { appState } from './state.js';

export function setupVoiceControls(container) {
  // Create voice input UI
  const voiceUI = document.createElement('div');
  voiceUI.className = 'voice-input-section';
  
  const title = document.createElement('div');
  title.className = 'voice-input-title';
  title.textContent = 'Voice Input';
  
  const buttonsContainer = document.createElement('div');
  buttonsContainer.className = 'voice-input-buttons';
  
  // Create speak button
  const speakBtn = document.createElement('button');
  speakBtn.className = 'btn btn-primary';
  speakBtn.innerHTML = '<i class="fas fa-microphone"></i> Speak Your Answer';
  speakBtn.onclick = startRecording;
  
  // Add elements to container
  buttonsContainer.appendChild(speakBtn);
  voiceUI.appendChild(title);
  voiceUI.appendChild(buttonsContainer);
  
  // Clear container and add our UI
  container.innerHTML = '';
  container.appendChild(voiceUI);
  
  // Recording state
  let mediaRecorder = null;
  let audioChunks = [];
  let isRecording = false;
  
  // Start recording function
  function startRecording() {
    if (isRecording) return;
    
    if (!navigator.mediaDevices) {
      alert('Your browser does not support audio recording');
      return;
    }
    
    navigator.mediaDevices.getUserMedia({ audio: true })
      .then(stream => {
        isRecording = true;
        speakBtn.innerHTML = '<i class="fas fa-stop-circle"></i> Stop Recording';
        speakBtn.classList.add('recording');
        
        // Create media recorder
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];
        
        // Collect audio chunks
        mediaRecorder.addEventListener('dataavailable', event => {
          audioChunks.push(event.data);
        });
        
        // When recording stops
        mediaRecorder.addEventListener('stop', () => {
          // Convert to blob
          const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
          
          // Reset UI
          speakBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
          
          // Send to server for transcription
          sendAudioForTranscription(audioBlob);
          
          // Stop all tracks
          stream.getTracks().forEach(track => track.stop());
        });
        
        // Start recording
        mediaRecorder.start();
        
        // Update button to stop recording
        speakBtn.onclick = stopRecording;
      })
      .catch(error => {
        console.error('Error accessing microphone:', error);
        alert('Could not access microphone. Please check permissions.');
      });
  }
  
  // Stop recording function
  function stopRecording() {
    if (!isRecording || !mediaRecorder) return;
    
    mediaRecorder.stop();
    isRecording = false;
  }
  
  // Send audio to server
  async function sendAudioForTranscription(audioBlob) {
    if (!appState.sessionId) {
      alert('Please start a session first');
      resetUI();
      return;
    }
    
    try {
      // Create form data
      const formData = new FormData();
      formData.append('audio_file', audioBlob, 'recording.wav');
      
      // Send to server
      window.startTopProgress && window.startTopProgress();
      const response = await fetch('/stt?language=en', {
        method: 'POST',
        body: formData
      });
      
      if (!response.ok) {
        throw new Error(`Server returned ${response.status}`);
      }
      
      const data = await response.json();
      
      // Fill answer input with transcription
      const answerInput = document.getElementById('answer');
      if (answerInput && data.text) {
        answerInput.value = data.text;
        answerInput.focus();
      }
      
      resetUI();
    } catch (error) {
      console.error('Error transcribing audio:', error);
      alert('Failed to transcribe audio. Please try again or type your answer.');
      resetUI();
    }
    finally {
      window.finishTopProgress && window.finishTopProgress();
    }
  }
  
  // Reset UI
  function resetUI() {
    speakBtn.innerHTML = '<i class="fas fa-microphone"></i> Speak Your Answer';
    speakBtn.classList.remove('recording');
    speakBtn.onclick = startRecording;
    isRecording = false;
  }
}