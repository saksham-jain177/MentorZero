// Global application state
export const appState = {
	sessionId: null,
	isProcessing: false,  // Flag to prevent multiple concurrent requests
	currentMode: 'explain', // Current mode (explain or quiz)
	lastTopic: null,       // Store the last topic used
};

export function setSessionId(id) {
	appState.sessionId = id;
}

export function resetSession() {
	appState.sessionId = null;
	appState.isProcessing = false;
	appState.lastTopic = null;
	console.log('Session state reset');
	
	// Force clear any cached data in localStorage
	try {
		localStorage.removeItem('lastQuery');
		localStorage.removeItem('lastSessionId');
	} catch(e) {
		console.error('Failed to clear localStorage:', e);
	}
}