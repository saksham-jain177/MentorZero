import { request } from './api.js';
import { setSessionId } from './state.js';

export async function startTeach({ userId, topic, mode }) {
	// Ensure we only send the current input and never a stale value
	const cleanTopic = typeof topic === 'string' ? topic.trim() : '';
	const res = await request('/teach', { 
		method: 'POST', 
		json: { 
			userId: userId, 
			topic: cleanTopic, 
			mode: mode 
		} 
	});
	
	if (!res.ok) return res;
	
	const { session_id, first } = res.data;
	setSessionId(session_id);
	return { ok: true, data: { session_id: session_id, first: first } };
}