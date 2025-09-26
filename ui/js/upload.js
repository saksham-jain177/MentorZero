import { request } from './api.js';
// Progress bar helpers are exposed on window from main.js

export async function uploadText({ sessionId, text }) {
	try {
		if (window.startTopProgress) window.startTopProgress();
		return await request('/upload_url_or_text', {
			method: 'POST',
			json: { sessionId: sessionId, text: text }
		});
	} finally {
		if (window.finishTopProgress) window.finishTopProgress();
	}
}

export async function scrapeFetch({ sessionId, url }) {
	try {
		if (window.startTopProgress) window.startTopProgress();
		return await request('/scrape_fetch', {
			method: 'POST',
			json: { sessionId, url }
		});
	} finally {
		if (window.finishTopProgress) window.finishTopProgress();
	}
}
