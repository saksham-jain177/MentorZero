export async function request(path, { method = 'GET', json } = {}) {
	const headers = { 'Accept': 'application/json', 'Cache-Control': 'no-store' };
	let body;
	if (json !== undefined) {
		headers['Content-Type'] = 'application/json';
		body = JSON.stringify(json);
	}
	let resp;
	try {
		resp = await fetch(path, { method, headers, body, cache: 'no-store' });
		const text = await resp.text();
		let data;
		try { data = text ? JSON.parse(text) : {}; } catch { data = { error: text || `HTTP ${resp.status}` }; }
		return { ok: resp.ok, status: resp.status, data };
	} catch (err) {
		return { ok: false, status: 0, data: { error: String(err) } };
	}
}
