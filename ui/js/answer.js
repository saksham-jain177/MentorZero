import { request } from './api.js';

export async function submitAnswer({ sessionId, userAnswer }) {
	const interactionId = `i_${Date.now()}`;
	return await request('/submit_answer', {
		method: 'POST',
		json: { sessionId: sessionId, userAnswer: userAnswer }
	});
}
