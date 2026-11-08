/* ==========================================================================
   Activity / jobs helpers: pure functions, no DOM access.
   Loaded before app.js; exposed as window.CodeAssistActivity (and via
   module.exports so tests/js/test_activity.mjs can import it under node).
   ========================================================================== */
(function (root, factory) {
    const api = factory();
    if (typeof module !== 'undefined' && module.exports) module.exports = api;
    root.CodeAssistActivity = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
    'use strict';

    /* Phases in pipeline order; hue drives the chip colour (CSS turns it into light/dark friendly tones). */
    const PHASES = ['INTAKE', 'DECOMPOSE', 'BASELINE', 'CONTRACT', 'IMPLEMENT', 'HARDEN', 'TEST', 'SANDBOX', 'FIX', 'CROSSCHECK', 'DELIVER', 'RETRIEVE', 'INGEST', 'VECTORIZE', 'LLM'];
    const PHASE_HUE = {
        INTAKE: 200, DECOMPOSE: 262, BASELINE: 215, CONTRACT: 285, IMPLEMENT: 150, HARDEN: 35, TEST: 170,
        SANDBOX: 320, FIX: 12, CROSSCHECK: 55, DELIVER: 130, RETRIEVE: 190, INGEST: 240, VECTORIZE: 300, LLM: 225
    };
    const STATUSES = ['start', 'ok', 'fail', 'skip', 'info'];
    const JOB_STATES = ['QUEUED', 'RUNNING', 'COMPLETED', 'FAILED', 'CANCELLED'];

    function phaseHue(phase) {
        const p = String(phase || '').toUpperCase();
        if (p in PHASE_HUE) return PHASE_HUE[p];
        let h = 0;                                    // unknown phase: stable hash so it still gets its own colour
        for (let i = 0; i < p.length; i++) h = (h * 31 + p.charCodeAt(i)) % 360;
        return h;
    }
    function phaseColor(phase) { return `hsl(${phaseHue(phase)} 70% 50%)`; }

    /* "log" is the default for events from before kind existed. */
    function eventKind(ev) {
        const k = ev && typeof ev.kind === 'string' ? ev.kind.toLowerCase() : '';
        return ['log', 'action', 'telemetry', 'job'].includes(k) ? k : 'log';
    }

    function normStatus(s) {
        const v = String(s == null ? '' : s).toLowerCase();
        return STATUSES.includes(v) ? v : 'info';
    }

    function toInt(v) { const n = Number(v); return Number.isFinite(n) ? Math.max(0, Math.round(n)) : 0; }

    function normalizeJob(j) {
        if (!j || typeof j !== 'object') return null;
        const raw = String(j.state == null ? '' : j.state).toUpperCase();
        const state = JOB_STATES.includes(raw) ? raw : 'QUEUED';
        const progress = toInt(j.progress), total = toInt(j.total);
        return Object.assign({}, j, {
            state, progress, total,
            kind: j.kind || 'vectorize',
            error: j.error || null,
            pct: total > 0 ? Math.min(100, Math.round(progress / total * 100)) : null
        });
    }
    const isActiveJob = j => !!j && (j.state === 'QUEUED' || j.state === 'RUNNING');
    const isTerminalJob = j => !!j && (j.state === 'COMPLETED' || j.state === 'FAILED' || j.state === 'CANCELLED');

    /* Header badge model from GET /repos/{id}/search-mode (job may be overridden by a fresher SSE job). */
    function searchModeInfo(mode, jobOverride) {
        if (!mode) return null;
        const job = normalizeJob(jobOverride || mode.job);
        let vs = String(mode.vector_status || 'none').toLowerCase();
        if (isActiveJob(job)) vs = job.state === 'RUNNING' ? 'running' : 'queued';
        else if (job && job.state === 'FAILED') vs = 'failed';
        else if (job && job.state === 'COMPLETED' && (vs === 'none' || vs === 'queued' || vs === 'running')) vs = 'ready';
        const retrieval = String(mode.retrieval_mode || '').toLowerCase();
        const pct = isActiveJob(job) ? job.pct : null;
        if (vs === 'running') return { kind: 'vectorizing', label: pct == null ? 'Vectorizing…' : `Vectorizing ${pct}%`, pct, canBuild: false, job };
        if (vs === 'queued') return { kind: 'vectorizing', label: 'Vectorize queued', pct: null, canBuild: false, job };
        if (vs === 'ready') return { kind: 'hybrid', label: retrieval === 'vector' ? 'Vector search' : 'Hybrid (vectors ready)', pct: null, canBuild: false, job };
        if (vs === 'failed') return { kind: 'failed', label: 'Vectorize failed', pct: null, canBuild: true, job };
        return { kind: 'agentic', label: 'Agentic search', pct: null, canBuild: true, job };
    }

    function tsMs(ts) { const t = Date.parse(ts); return Number.isFinite(t) ? t : 0; }

    /* Merge start + closing events (closing.parent_uid === start.uid) into one row each. Duplicate uids collapse. */
    function mergeActionEvents(events) {
        const byUid = new Map(), order = [];
        (events || []).forEach(ev => {
            if (!ev || typeof ev !== 'object') return;
            const uid = ev.uid != null ? String(ev.uid) : null;
            if (uid && byUid.has(uid)) { byUid.set(uid, ev); return; }          // later copy wins, position stays
            const key = uid || `anon:${order.length}`;
            byUid.set(key, ev); order.push(key);
        });
        const starts = new Map();                                              // start uid -> row
        const rows = [];
        const mk = (key, s, e) => ({
            uid: key, start: s, end: e || null,
            ts: s.ts, startMs: tsMs(s.ts), endMs: e ? tsMs(e.ts) : null,
            request_id: s.request_id || (e && e.request_id) || null,
            thread_id: s.thread_id || (e && e.thread_id) || null,
            repo_id: s.repo_id || (e && e.repo_id) || null,
            phase: String(s.phase || '').toUpperCase(), agent: s.agent || '', action: s.action || '',
            status: e ? normStatus(e.status) : normStatus(s.status),
            open: !e && normStatus(s.status) === 'start',
            detail: (e && e.detail) || s.detail || '',
            data: mergeData(s.data, e && e.data),
            duration_ms: e && e.duration_ms != null ? e.duration_ms : (s.duration_ms != null ? s.duration_ms : null),
            model: (e && e.model) || s.model || null
        });
        order.forEach(k => {
            const ev = byUid.get(k);
            if (normStatus(ev.status) === 'start') { const r = mk(k, ev, null); starts.set(k, r); rows.push(r); }
        });
        order.forEach(k => {
            const ev = byUid.get(k);
            if (normStatus(ev.status) === 'start') return;
            const parent = ev.parent_uid != null ? starts.get(String(ev.parent_uid)) : null;
            if (parent && !parent.end) {
                Object.assign(parent, mk(parent.uid, parent.start, ev));
                starts.set(parent.uid, parent);
            } else rows.push(mk(k, ev, null));
        });
        // Object.assign above rebuilt fields on the existing row objects, so `rows` already holds the merged rows.
        return rows.map((r, i) => [r, i]).sort((a, b) => (a[0].startMs - b[0].startMs) || (a[1] - b[1])).map(x => x[0]);
    }
    function mergeData(a, b) {
        const isObj = x => x && typeof x === 'object' && !Array.isArray(x);
        if (isObj(a) && isObj(b)) return Object.assign({}, a, b);
        if (isObj(b) ? Object.keys(b).length : b != null) return b;
        return a == null ? {} : a;
    }

    function rowEndMs(r, now) {
        if (r.endMs != null) return r.endMs;
        if (r.duration_ms != null) return r.startMs + r.duration_ms;
        return r.open ? (now || r.startMs) : r.startMs;
    }

    const snippet = (t, n) => { const s = String(t == null ? '' : t).replace(/\s+/g, ' ').trim(); return s.length > n ? s.slice(0, Math.max(0, n - 1)) + '…' : s; };

    function groupKey(r) { return r.request_id ? String(r.request_id) : (r.thread_id ? 'thread:' + r.thread_id : 'system'); }

    /* Rows -> groups per request, newest group first, rows inside a group in time order. */
    function groupByRequest(rows, opts) {
        opts = opts || {};
        const queries = opts.queries || {}, now = opts.now || Date.now();
        const map = new Map();
        (rows || []).forEach(r => {
            const k = groupKey(r);
            if (!map.has(k)) map.set(k, { key: k, request_id: r.request_id || null, thread_id: r.thread_id || null, rows: [] });
            map.get(k).rows.push(r);
        });
        const groups = [...map.values()].map(g => {
            let first = Infinity, last = 0, ok = 0, fail = 0, skip = 0, open = 0;
            g.rows.forEach(r => {
                first = Math.min(first, r.startMs); last = Math.max(last, rowEndMs(r, now));
                if (r.open) open++; else if (r.status === 'fail') fail++; else if (r.status === 'skip') skip++; else ok++;
            });
            const intake = g.rows.find(r => r.phase === 'INTAKE' && r.data && (r.data.query || r.data.user_query));
            const q = (g.request_id && queries[g.request_id]) || (g.thread_id && queries['thread:' + g.thread_id]) ||
                (intake && (intake.data.query || intake.data.user_query)) || '';
            g.query = snippet(q, 90);
            g.startMs = first === Infinity ? 0 : first; g.endMs = last;
            g.durationMs = Math.max(0, last - g.startMs);
            g.counts = { total: g.rows.length, ok, fail, skip, open };
            g.status = open ? 'running' : fail ? 'fail' : 'ok';
            g.label = g.request_id ? (g.query || 'Request ' + String(g.request_id).slice(0, 8)) : (g.thread_id ? (g.query || 'Thread activity') : 'System activity');
            return g;
        });
        return groups.sort((a, b) => (b.endMs - a.endMs) || (b.startMs - a.startMs));
    }

    function filterRows(rows, f) {
        f = f || {};
        const phases = f.phases instanceof Set ? f.phases : new Set(f.phases || []);
        const text = String(f.text || '').trim().toLowerCase();
        return (rows || []).filter(r => {
            if (phases.size && !phases.has(r.phase)) return false;
            if (f.failuresOnly && r.status !== 'fail') return false;
            if (f.threadId && String(r.thread_id) !== String(f.threadId)) return false;
            if (text && !`${r.phase} ${r.agent} ${r.action} ${r.detail} ${r.model || ''} ${r.status}`.toLowerCase().includes(text)) return false;
            return true;
        });
    }

    /* Apply a DOM-row budget: walk newest groups first, only expanded groups spend budget. */
    function selectVisible(groups, isOpen, cap) {
        let used = 0, hidden = 0;
        const out = groups.map(g => {
            if (!isOpen(g)) return { group: g, rows: [], open: false, truncated: 0 };
            const room = Math.max(0, cap - used);
            const rows = g.rows.length > room ? g.rows.slice(g.rows.length - room) : g.rows;   // keep the newest rows
            used += rows.length; hidden += g.rows.length - rows.length;
            return { group: g, rows, open: true, truncated: g.rows.length - rows.length };
        });
        return { groups: out, shown: used, hidden };
    }

    function fmtDuration(ms) {
        if (ms == null || !Number.isFinite(Number(ms))) return '';
        ms = Number(ms);
        if (ms < 1000) return Math.round(ms) + ' ms';
        if (ms < 60000) return (ms / 1000).toFixed(ms < 10000 ? 1 : 0) + ' s';
        const m = Math.floor(ms / 60000), s = Math.round((ms % 60000) / 1000);
        return s === 60 ? `${m + 1}m 0s` : `${m}m ${s}s`;
    }

    return { PHASES, STATUSES, JOB_STATES, phaseHue, phaseColor, eventKind, normStatus, normalizeJob, isActiveJob, isTerminalJob, searchModeInfo, mergeActionEvents, groupByRequest, filterRows, selectVisible, fmtDuration, snippet, rowEndMs };
});
