const Render = {
    escapeHtml: function (s) {
        return String(s == null ? "" : s)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#39;");
    },
    formatInline: function (s) {
        let cleaned = s
            .replace(/\$?\\rightarrow\$?/g, "→")
            .replace(/\$?\\leftarrow\$?/g, "←")
            .replace(/\$?\\Rightarrow\$?/g, "⇒");
        return cleaned
            // 1. Markdown Links [text](url)
            .replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, '<a href="$2" class="text-blue-400 hover:underline inline-flex items-center gap-1" target="_blank" rel="noopener noreferrer">$1</a>')
            .replace(/`([^`\n]+)`/g, "<code class='bg-slate-100 text-emerald-700 dark:bg-slate-950 dark:text-emerald-300 px-1.5 py-0.5 rounded font-mono text-xs border border-slate-300 dark:border-slate-800'>$1</code>")
            .replace(/\*\*\*([^*\n]+)\*\*\*/g, "<strong class='text-slate-900 dark:text-white font-semibold'><em>$1</em></strong>")
            .replace(/\*\*([^*\n]+)\*\*/g, "<strong class='text-slate-900 dark:text-white font-semibold'>$1</strong>")
            .replace(/__([^_\n]+)__/g, "<strong class='text-slate-900 dark:text-white font-semibold'>$1</strong>")
            .replace(/(^|[\s(])\*([^*\n]+)\*(?=[\s.,!?:;)]|$)/g, "$1<em>$2</em>");
    },
    renderBlock: function (raw) {
        let normalized = String(raw == null ? "" : raw)
            .replace(/\r\n/g, "\n")
            .replace(/([^\n])\s+(#{1,6}\s+)/g, "$1\n$2")
            .replace(/([.:;!?])\s+(\*\s+\*\*)/g, "$1\n$2")
            .replace(/^\n+|\n+$/g, "");      // no spacer at the edges of a block next to a code fence

        const lines = this.escapeHtml(normalized).split("\n");
        const out = [];
        const SPACER = "<div class='h-2'></div>";
        const BLOCK_START = /^\s*(\||(&gt;\s*)?#{1,6}\s|[-*]\s|\d+\.\s|([-*_]){3,}\s*$)/;
        let inTable = false;

        for (let i = 0; i < lines.length; i++) {
            let line = lines[i];

            if (/^\s*\|.*\|\s*$/.test(line)) {
                if (/^\s*\|[\s\-:|]+\|\s*$/.test(line)) continue;
                const cells = line.trim().slice(1, -1).split("|").map(c => this.formatInline(c.trim()));
                if (!inTable) {
                    out.push("<div class='overflow-x-auto'><table class='w-full text-left border-collapse my-3 text-xs border border-slate-300 dark:border-slate-700'><thead><tr class='bg-slate-100 dark:bg-slate-900 border-b border-slate-300 dark:border-slate-700'>" + cells.map(c => "<th class='p-2 font-semibold text-slate-700 dark:text-slate-200'>" + c + "</th>").join("") + "</tr></thead><tbody>");
                    inTable = true;
                } else {
                    out.push("<tr class='border-b border-slate-200 dark:border-slate-800'>" + cells.map(c => "<td class='p-2 text-slate-600 dark:text-slate-300'>" + c + "</td>").join("") + "</tr>");
                }
                continue;
            } else if (inTable) {
                out.push("</tbody></table></div>");
                inTable = false;
            }

            if (!line.trim()) {                       // blank line: one small spacer, never stacked
                if (out.length && out[out.length - 1] !== SPACER) out.push(SPACER);
                continue;
            }

            const h = line.match(/^(&gt;\s*)?(#{1,6})\s+(.*)$/);
            if (h) {
                const lvl = Math.min(h[2].length + 1, 6);
                out.push(`<h${lvl} class="font-bold text-slate-900 dark:text-white mt-4 mb-1.5 text-base">${this.formatInline(h[3])}</h${lvl}>`);
                continue;
            }

            if (/^\s*([-*_]){3,}\s*$/.test(line)) {
                out.push("<hr class='border-slate-300 dark:border-slate-700 my-3'>");
                continue;
            }

            const ul = line.match(/^(\s*)[-*]\s+(.*)$/);
            if (ul) {
                out.push(`<div class="flex items-start space-x-2 my-1.5 pl-2"><span class="text-blue-400 font-bold">•</span><div class="flex-1 min-w-0">${this.formatInline(ul[2])}</div></div>`);
                continue;
            }

            const ol = line.match(/^(\s*)(\d+\.)\s+(.*)$/);
            if (ol) {
                out.push(`<div class="flex items-start space-x-2 my-1.5 pl-2"><span class="text-blue-400 font-mono text-xs font-bold">${ol[2]}</span><div class="flex-1 min-w-0">${this.formatInline(ol[3])}</div></div>`);
                continue;
            }

            const next = lines[i + 1];                // soft line break only when another plain text line follows
            const softBreak = next !== undefined && next.trim() !== "" && !BLOCK_START.test(next);
            out.push(this.formatInline(line) + (softBreak ? "<br>" : ""));
        }
        if (inTable) out.push("</tbody></table></div>");
        return out.join("");
    },
    // Answers end with "### Delivery notes" (assumptions / limitations / verified / unverified / requirements):
    // rendered as a collapsible callout card. A marker inside a code fence is not a heading.
    findDeliveryNotes: function (src) {
        const re = /^#{2,4}[ \t]*Delivery notes[ \t]*:?[ \t\r]*$/gim;
        let m;
        while ((m = re.exec(src))) {
            if ((src.slice(0, m.index).split("```").length - 1) % 2 === 0) return { index: m.index, length: m[0].length };
        }
        return null;
    },
    renderMarkdown: function (text) {
        const src = String(text == null ? "" : text);
        const dn = this.findDeliveryNotes(src);
        if (!dn) return this.renderFenced(src);
        const body = src.slice(dn.index + dn.length);
        return this.renderFenced(src.slice(0, dn.index)) +
            "<details class='delivery-notes' open><summary>Delivery notes</summary><div class='delivery-notes-body'>" +
            this.renderFenced(body) + "</div></details>";
    },
    renderFenced: function (text) {
        return String(text == null ? "" : text).split("```").map((part, i) => {
            if (i % 2 === 0) return this.renderBlock(part);
            const lang = (part.match(/^([^\n]*)\n/) || [, ""])[1].trim();
            const cleaned = part.replace(/^[^\n]*\n/, "").replace(/\n$/, "");
            return "<div class='codeblock my-3 rounded-xl border border-slate-300 dark:border-slate-700 bg-slate-100 dark:bg-slate-950 overflow-hidden'>" +
                "<div class='flex items-center justify-between px-3 py-1.5 bg-slate-200/80 dark:bg-slate-900/80 border-b border-slate-300 dark:border-slate-800 text-[11px] text-slate-600 dark:text-slate-400 font-mono'><span>" + this.escapeHtml(lang || "code") + "</span>" +
                "<button type='button' data-copy='code' class='hover:text-slate-900 dark:hover:text-white'><span>Copy</span></button></div>" +
                "<pre class='p-3.5 font-mono text-xs overflow-x-auto text-slate-800 dark:text-slate-200'><code>" + this.escapeHtml(cleaned) + "</code></pre></div>";
        }).join("");
    }
};

/* ================= State & helpers ================= */
const $ = id => document.getElementById(id);
const icons = () => { if (window.lucide) lucide.createIcons(); };
const esc = s => Render.escapeHtml(s);
const LS = {
    get(k, d) { try { const v = localStorage.getItem('crag.' + k); return v === null ? d : JSON.parse(v); } catch { return d; } },
    set(k, v) { try { localStorage.setItem('crag.' + k, JSON.stringify(v)); } catch { /* storage unavailable */ } }
};
const GENERAL = '__general__';
const S = { web: LS.get('web', 'auto'), repoId: null, threadId: null, repos: [], tab: null, sending: false, files: [], unread: 0, collapsed: LS.get('collapsed', null), poll: null, stream: null, streamKey: '', tlimit: LS.get('tlimit', null), abortController: null };
S.model = LS.get('model', '');          // '' = server default
S.runner = LS.get('runner', '');        // '' = the server's LLM_RUNNER; else a runner name from /api/v1/runners (chat only: embeddings never switch)
S.tokenMode = LS.get('tokenMode', '');  // '' = the server's TOKEN_MODE until the user picks one; auto | lean | normal
const AX = window.CodeAssistActivity;   // pure helpers from static/js/activity.js (loaded first)
// Activity timeline state. events: uid -> ActionEvent (capped); rowIndex: merged-row lookup for lazy detail rendering.
S.act = { events: new Map(), queries: {}, done: new Set(), groupOpen: new Map(), rowOpen: new Set(), phases: new Set(), failuresOnly: false, text: '', thisThread: false, dirty: false, liveDirty: false, raf: 0, hasRunning: false, repo: null };
S.rowIndex = new Map();
S.live = null;            // pending chat request: { id, threadId, requestId, uids:Set, query }
S.jobs = {};              // job_id -> normalized Job (last seen)
S.searchMode = null;      // GET /repos/{id}/search-mode for the selected repo
S.searchJob = null;       // fresher job from SSE for the selected repo
const TABS = ['chat', 'repos', 'ingest', 'telemetry', 'activity'];
const STATUS_HISTORY_LIMIT = 5;
const TITLES = { chat: 'Chat', repos: 'Repositories', ingest: 'Ingest repository', telemetry: 'Telemetry & logs', activity: 'Agent activity' };
const isMobile = () => window.innerWidth < 1024;
const GENERAL_SUGGESTIONS = ['What is new in the latest FastAPI release?', 'Compare Chroma and FAISS for a small RAG POC', 'Summarize this page: https://'];
const SUGGESTIONS = ['Give me an architecture overview', 'Where is the main entry point?', 'Which modules have the most dependencies?', 'Spot risky or untested code paths'];

document.addEventListener('DOMContentLoaded', async () => {
    const theme = LS.get('theme', 'dark');
    document.documentElement.classList.toggle('dark', theme === 'dark');
    icons();
    const panel = $('thread-panel');
    panel.dataset.open = String(isMobile() ? false : LS.get('threads', true));
    setupComposer();
    setupAttachments();
    wireChatSettings();
    loadRunners();      // fills the runner switch, then loadModels()
    initTokenMode();
    syncLimitSelect(S.tlimit);
    buildActivityControls();
    setupNotificationStream();
    setInterval(() => { if (S.tab === 'activity' && S.act.hasRunning) { S.act.dirty = true; scheduleRender(); } }, 1000);   // tick running durations
    $('chat-messages').addEventListener('scroll', updateJumpBtn);
    window.addEventListener('hashchange', () => switchTab(location.hash.slice(1)));
    await loadRepositories();
    const hinted = location.hash.slice(1);
    switchTab(TABS.includes(hinted) ? hinted : (S.repos.length ? 'chat' : 'ingest'));
});

/* ================= Navigation & layout ================= */
function applyLayout() {
    $('sidebar').dataset.collapsed = String(S.collapsed ?? S.tab === 'chat'); // auto-compact in chat unless the user chose
}

function switchTab(tabId) {
    if (!TABS.includes(tabId)) tabId = 'chat';
    S.tab = tabId;
    document.querySelectorAll('.tab-content').forEach(el => {
        const on = el.id === 'tab-' + tabId;
        el.classList.toggle('hidden', !on);
        if (el.id === 'tab-telemetry' || el.id === 'tab-activity') el.classList.toggle('flex', on);
    });
    document.querySelectorAll('#sidebar nav button').forEach(b => {
        const on = b.id === 'nav-' + tabId;
        b.classList.toggle('active', on);
        on ? b.setAttribute('aria-current', 'page') : b.removeAttribute('aria-current');
    });
    document.querySelectorAll('.mob-nav-btn').forEach(b => b.classList.toggle('active', b.dataset.mobTab === tabId));
    $('current-tab-title').textContent = TITLES[tabId];
    $('current-tab-title').classList.toggle('hidden', tabId === 'chat');
    $('chat-ctx').classList.toggle('hidden', tabId !== 'chat');
    $('chat-ctx').classList.toggle('flex', tabId === 'chat');
    $('focus-btn').classList.toggle('lg:block', tabId === 'chat');
    $('focus-btn').classList.toggle('lg:hidden', tabId !== 'chat');
    if (tabId !== 'chat') document.body.classList.remove('focus');
    if (location.hash !== '#' + tabId) history.replaceState(null, '', '#' + tabId);
    if (tabId === 'activity') renderActivity();
    if (tabId === 'telemetry') { S.unread = 0; renderBadge(); $('telemetry-console').scrollTop = 0; }   // newest line is on top
    applyLayout();
    closeMobileNav();
    if (tabId === 'repos' || tabId === 'chat') loadRepositories();
    if (tabId === 'chat') updateComposer();
}

function toggleSidebar() {
    if (isMobile()) return openMobileNav();
    S.collapsed = $('sidebar').dataset.collapsed !== 'true';
    LS.set('collapsed', S.collapsed);
    applyLayout();
}
function openMobileNav() { $('sidebar').dataset.open = 'true'; $('scrim').classList.remove('hidden'); }
function closeMobileNav() { $('sidebar').dataset.open = 'false'; $('scrim').classList.add('hidden'); }

function toggleThreads() {
    const p = $('thread-panel');
    const open = p.dataset.open !== 'true';
    p.dataset.open = String(open);
    if (!isMobile()) LS.set('threads', open);
}
function toggleFocus() { document.body.classList.toggle('focus'); }

function toggleTheme() {
    const isDark = document.documentElement.classList.toggle('dark');
    LS.set('theme', isDark ? 'dark' : 'light');
}

document.addEventListener('keydown', e => {
    const typing = /INPUT|TEXTAREA|SELECT/.test(document.activeElement?.tagName || '');
    if (e.altKey && !e.ctrlKey && !e.metaKey) {
        const n = { Digit1: 0, Digit2: 1, Digit3: 2, Digit4: 3, Digit5: 4 }[e.code];
        if (n !== undefined) { e.preventDefault(); return switchTab(TABS[n]); }
        if (e.code === 'KeyB') { e.preventDefault(); return toggleSidebar(); }
        if (e.code === 'KeyT') { e.preventDefault(); switchTab('chat'); return toggleThreads(); }
        if (e.code === 'KeyF') { e.preventDefault(); switchTab('chat'); return toggleFocus(); }
        if (e.code === 'KeyN') { e.preventDefault(); switchTab('chat'); return toggleNewThreadForm(); }
    } else if (e.key === 'Escape') {
        document.body.classList.remove('focus'); closeMobileNav();
    } else if (e.key === '/' && !typing && S.tab === 'chat') {
        e.preventDefault(); $('chat-input-query').focus();
    }
});

/* ================= Toasts & live notification stream ================= */
function showToast(title, message, type = 'info', ms) {
    const container = $('toast-container');
    if (!container) return;
    const bar = { success: 'bg-emerald-500', error: 'bg-red-500', warning: 'bg-amber-500' }[type] || 'bg-blue-500';
    const toast = document.createElement('div');
    toast.className = 'pointer-events-auto flex rounded-xl border border-slate-300 dark:border-slate-700 shadow-2xl bg-white dark:bg-slate-800 text-slate-900 dark:text-white overflow-hidden cursor-pointer';
    toast.innerHTML = `<span class="w-1 ${bar} shrink-0"></span><div class="flex-1 min-w-0 px-3.5 py-3"><h5 class="text-xs font-semibold">${esc(title)}</h5><p class="text-[11px] text-slate-600 dark:text-slate-300 mt-0.5 break-words">${esc(message)}</p></div>`;
    toast.dataset.msg = String(message);
    toast.onclick = () => toast.remove();
    container.appendChild(toast);
    setTimeout(() => toast.remove(), ms || 4500);
}

/* ---- Failures: a failed request must never be silent. The server answers JSON {detail, kind?, ...} (402 quota, 401 credentials, 504 timeout,
   503 unreachable, 500 internal); anything else (plain text, HTML, a dead connection) is described here as well. ---- */
async function describeFailure(res) {
    let detail = '', kind = '', body = null;
    try { body = await res.clone().json(); } catch { /* not JSON */ }
    if (body && typeof body === 'object') {
        const d = body.detail;
        if (typeof d === 'string') detail = d;
        else if (Array.isArray(d)) detail = d.map(x => (x && (x.msg || x.message)) || (typeof x === 'string' ? x : JSON.stringify(x))).join('; ');   // 422 validation errors
        else if (d && typeof d === 'object') detail = d.message || JSON.stringify(d);
        kind = String(body.kind || body.error || '');
    }
    if (!detail) { try { detail = (await res.text()).replace(/\s+/g, ' ').trim().slice(0, 300); } catch { /* body unreadable */ } }
    if (!detail) detail = `Server returned ${res.status}${res.statusText ? ' ' + res.statusText : ''}`;
    const titles = { timeout: 'Model timeout', unavailable: 'Model unreachable', internal: 'Server error', model_quota: 'Model quota used up', model_auth: 'Model credentials rejected' };
    const byStatus = { 401: 'Model credentials rejected', 402: 'Model quota used up', 413: 'Attachment too large', 415: 'Unsupported attachment', 503: 'Model unreachable', 504: 'Model timeout' };
    const title = titles[kind] || byStatus[res.status] || (res.status >= 500 ? 'Server error' : `Request failed (HTTP ${res.status})`);
    return { title, message: detail, kind, status: res.status };
}

function failureOf(err) {       // an Error thrown by the chat flow -> {title, message}
    if (err && err.failure) return err.failure;
    const m = (err && err.message) || String(err);
    return { title: 'Message failed', message: /failed to fetch|networkerror|load failed|network request failed/i.test(m)
        ? `Could not reach the CodeAssist server (${m}). Is it still running? Your message is back in the box.` : m };
}

function errorBubbleHtml(title, message) {
    return `<div class="flex gap-3 group scroll-mt-4" role="alert">
        <div class="msg-avatar w-7 h-7 mt-0.5 rounded-lg bg-red-500/15 border border-red-500/30 flex items-center justify-center text-red-400 shrink-0"><i data-lucide="alert-triangle" class="w-4 h-4"></i></div>
        <div class="min-w-0 flex-1"><div class="text-sm rounded-xl border border-red-500/40 bg-red-500/10 text-red-700 dark:text-red-300 px-3.5 py-2.5 break-words">
            <div class="font-semibold text-xs mb-0.5">${esc(title)}</div><div class="whitespace-pre-wrap">${esc(message)}</div></div></div></div>`;
}

function showFailure(f) {       // chat error bubble + error toast (longer than the default, and it replaces the same text from the event stream)
    const col = $('chat-col');
    if (col) { col.insertAdjacentHTML('beforeend', errorBubbleHtml(f.title, f.message)); icons(); col.lastElementChild.scrollIntoView({ block: 'nearest' }); }
    document.querySelectorAll('#toast-container [data-msg]').forEach(t => { if (t.dataset.msg === String(f.message)) t.remove(); });
    showToast(f.title, f.message, 'error', 12000);
}

function setConn(ok) {
    $('conn-status').dataset.ok = String(ok);
    $('logo-dot').dataset.ok = String(ok);
    $('conn-label').textContent = ok ? 'Live' : 'Reconnecting…';
}

function renderBadge() {
    const b = $('telemetry-badge');
    b.textContent = S.unread > 99 ? '99+' : S.unread;
    b.classList.toggle('hidden', !S.unread);
}

/* ---- Telemetry & logs: one SSE stream per (selected repo, last-N). Server replays stored history, then streams live. ---- */
const TELEMETRY_CAP = 500;       // lines kept in the console (never below the chosen N)

function streamRepo() { return S.repoId && S.repoId !== GENERAL ? S.repoId : ''; }

function renderScopeHint() {
    const el = $('telemetry-scope');
    if (!el) return;
    const r = S.repos.find(x => String(x.repo_id) === String(streamRepo()));
    el.textContent = streamRepo()
        ? `General events + events of ${r ? r.title : streamRepo()}`
        : 'General events only. Select a repository to include its events.';
}

function syncLimitSelect(n) {
    const sel = $('telemetry-limit');
    if (!sel || n == null) return;
    if (![...sel.options].some(o => o.value === String(n))) {
        const opt = new Option(String(n), String(n));
        const after = [...sel.options].find(o => Number(o.value) > Number(n));
        sel.add(opt, after || null);
    }
    sel.value = String(n);
}

function onTelemetryLimitChange(v) {
    S.tlimit = Math.max(0, parseInt(v, 10) || 0);
    LS.set('tlimit', S.tlimit);
    setupNotificationStream();
}

function fmtTs(ts) {
    const d = ts ? new Date(ts) : new Date();
    if (isNaN(d)) return String(ts || '');
    const time = d.toLocaleTimeString([], { hour12: false });
    return d.toDateString() === new Date().toDateString() ? time : `${d.toLocaleDateString([], { month: 'short', day: 'numeric' })} ${time}`;
}

function prettyMessage(ev) {
    if (ev.title === 'Telemetry Trace') {      // LLM trace: JSON payload -> one readable line
        try {
            const t = JSON.parse(ev.message);
            const extra = (t.cached_input_tokens ? ` · cached ${t.cached_input_tokens}` : '') + (t.est_tokens_saved ? ` · saved ~${t.est_tokens_saved}` : '') +
                (t.runner ? ` · ${t.runner}` : '') + (t.token_mode ? ` · mode ${t.token_mode}` : '');
            const asked = t.requested_model && t.requested_model !== t.model ? ` (asked ${t.requested_model})` : '';      // the id really called comes first
            return `${t.phase} · ${t.model}${asked} · in ${t.input_tokens} / out ${t.output_tokens} / total ${t.total_tokens} tok · ${t.duration_sec}s${extra}`;
        } catch { /* not JSON: show as is */ }
    }
    if (ev.title === 'Token usage') {          // one summary line per question: the sum of its LLM calls
        try {
            const t = JSON.parse(ev.message);
            return `${t.calls} call${t.calls === 1 ? '' : 's'} · in ${t.input_tokens}${t.cached_input_tokens ? ` (${t.cached_input_tokens} cached)` : ''} / out ${t.output_tokens} tok` +
                `${t.est_tokens_saved ? ` · saved ~${t.est_tokens_saved}` : ''} · ${t.runner} · mode ${t.token_mode}`;
        } catch { /* not JSON: show as is */ }
    }
    return ev.message;
}

function telemetryLine(ev) {
    const level = String(ev.level || 'info').toLowerCase();
    const colour = { error: 'text-red-400', warning: 'text-amber-400', success: 'text-emerald-400' }[level] || 'text-blue-300';
    const line = document.createElement('div');
    line.className = `${colour} border-b border-slate-200 dark:border-slate-900 pb-1`;
    line.textContent = `${fmtTs(ev.ts)}  [${ev.level}] ${ev.scope === 'repo' ? '(repo) ' : ''}${ev.title} — ${prettyMessage(ev)}`;
    return line;
}

function updateLiveStatus(msg) {
    const st = S.liveStatus && $(S.liveStatus);
    if (!st) return;
    if (!S.statusHistory) S.statusHistory = [];
    if (S.statusHistory[0] !== msg) {
        S.statusHistory.unshift(msg);
        if (S.statusHistory.length > STATUS_HISTORY_LIMIT) S.statusHistory.pop();
    }
    st.textContent = msg.length > 160 ? msg.slice(0, 157) + '…' : msg;
    st.title = msg;
    const histEl = $(S.liveStatus.replace('_s', '_h'));
    if (histEl) {
        histEl.innerHTML = S.statusHistory.slice(1).map((m, i) => `<div class="text-[10px] opacity-60 py-0.5">${i + 1}. ${esc(m)}</div>`).join('');
        histEl.classList.toggle('hidden', S.statusHistory.length <= 1);
    }
}

function setupNotificationStream() {
    const repo = streamRepo();
    const key = `${repo}|${S.tlimit}`;
    if (S.stream && S.streamKey === key) return;           // same scope: keep the open stream
    if (S.stream) S.stream.close();
    S.streamKey = key;
    renderScopeHint();
    if (S.act.repo !== repo) { S.act.repo = repo; S.act.events.clear(); S.act.rowOpen.clear(); scheduleRender(); }   // other repo: other timeline
    loadActionHistory();

    const qs = new URLSearchParams();
    if (repo) qs.set('repo_id', repo);
    if (S.tlimit != null) qs.set('limit', String(S.tlimit)); // none chosen yet: the server default applies
    const evtSource = S.stream = new EventSource(`${window.location.origin}/api/v1/notifications/stream${qs.toString() ? '?' + qs : ''}`);
    evtSource.onopen = () => setConn(true);
    evtSource.onerror = () => setConn(false);

    // Stored events: replaces the console (also after an automatic reconnect, so nothing is duplicated). No toasts.
    evtSource.addEventListener("history", (event) => {
        setConn(true);
        try {
            const data = JSON.parse(event.data);
            if (S.tlimit == null) syncLimitSelect(data.limit);
            const consoleDiv = $('telemetry-console');
            const frag = document.createDocumentFragment();
            const logs = [], actions = [];
            (data.events || []).forEach(ev => {
                const k = AX.eventKind(ev);
                if (k === 'action') { if (ev.data) actions.push(ev.data); }          // Activity only: never floods the console
                else if (k === 'job') applyJob(ev.data, true);
                else logs.push(ev);
            });
            if (actions.length) addActionEvents(actions);
            logs.forEach(ev => frag.prepend(telemetryLine(ev)));                   // oldest first in -> newest on top
            if (!frag.childNodes.length) {
                const none = document.createElement('div');
                none.className = 'text-slate-500';
                none.textContent = '[System] No stored events yet for this scope.';
                frag.append(none);
            }
            consoleDiv.replaceChildren(frag);
        } catch (err) {
            console.error("Failed to parse telemetry history:", err, event.data);
        }
    });

    evtSource.addEventListener("notification", (event) => {
        setConn(true);
        try {
            const data = typeof event.data === 'string' ? JSON.parse(event.data) : event.data;
            const kind = AX.eventKind(data);
            if (kind === 'action') return onActionEvent(data.data);              // Activity + live status only: no console, toast or unread badge
            if (kind === 'job') return applyJob(data.data, false);               // job UI + toasts
            const level = String(data.level || 'info').toLowerCase();
            const st = S.liveStatus && $(S.liveStatus);
            if (st) updateLiveStatus(`${data.title} — ${prettyMessage(data)}`);
            if (!(st && level === 'info')) showToast(data.title, prettyMessage(data), level);   // info is already in the loader
            const consoleDiv = $('telemetry-console');
            consoleDiv.querySelector(':scope > .text-slate-500')?.remove();          // placeholder line
            consoleDiv.prepend(telemetryLine(data));
            const cap = Math.max(TELEMETRY_CAP, S.tlimit || 0);
            while (consoleDiv.children.length > cap) consoleDiv.lastChild.remove();  // newest is first: drop the oldest
            if (S.tab !== 'telemetry') { S.unread++; renderBadge(); }
        } catch (err) {
            console.error("Failed to parse SSE notification event:", err, event.data);
        }
    });
}
function clearConsole() { $('telemetry-console').innerHTML = ''; }
function copyConsole() { copyText($('telemetry-console').innerText).then(() => showToast('Copied', 'Telemetry log copied to clipboard.', 'success')); }

function openTelemetryLegend() {
    $('telemetry-legend-modal').classList.remove('hidden');
    icons();
}
function closeTelemetryLegend() {
    $('telemetry-legend-modal').classList.add('hidden');
}

function copyText(text) {
    if (navigator.clipboard && window.isSecureContext) return navigator.clipboard.writeText(text);
    const ta = document.createElement('textarea');
    ta.value = text; ta.style.position = 'fixed'; ta.style.opacity = '0';
    document.body.appendChild(ta); ta.select(); document.execCommand('copy'); ta.remove();
    return Promise.resolve();
}

document.addEventListener('click', e => {
    const btn = e.target.closest('[data-copy]');
    if (!btn) return;
    const text = btn.dataset.copy === 'code'
        ? btn.closest('.codeblock').querySelector('code').innerText
        : btn.closest('[data-raw]').dataset.raw;
    copyText(text).then(() => {
        const label = btn.querySelector('span') || btn;
        const old = label.textContent;
        label.textContent = 'Copied';
        setTimeout(() => (label.textContent = old), 1400);
    });
});

/* ================= Ingestion ================= */
let ingestTimer = null;
function renderRecentPaths() {
    const paths = LS.get('recent', []);
    $('recent-paths').innerHTML = paths.map(p => `<button type="button" data-path="${esc(p)}" class="text-[11px] font-mono text-slate-500 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white bg-slate-100 dark:bg-slate-900 border border-slate-300 dark:border-slate-800 hover:border-slate-400 dark:hover:border-slate-600 rounded-lg px-2.5 py-1 max-w-full truncate" title="${esc(p)}">${esc(p)}</button>`).join('');
}
document.addEventListener('click', e => {
    const b = e.target.closest('[data-path]');
    if (b) { $('ingest-repo-path').value = b.dataset.path; $('ingest-repo-path').focus(); }
});

async function triggerIngestion() {
    const pathInput = $('ingest-repo-path');
    const repoPath = pathInput.value.trim();
    if (!repoPath) {
        showToast("Path required", "Enter the absolute path to a repository.", "error");
        pathInput.focus();
        return;
    }
    const card = $('ingest-progress-card'), btn = $('ingest-btn');
    const t0 = Date.now();
    card.classList.remove('hidden'); btn.disabled = true;
    ingestTimer = setInterval(() => ($('ingest-timer').textContent = ((Date.now() - t0) / 1000).toFixed(1) + 's'), 100);
    try {
        const response = await fetch(`/api/v1/ingest?repo_root=${encodeURIComponent(repoPath)}`, { method: 'POST' });
        const result = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(result.detail || result.message || `Server returned ${response.status}`);
        LS.set('recent', [repoPath, ...LS.get('recent', []).filter(p => p !== repoPath)].slice(0, 5));
        pathInput.value = '';
        if (result.repo_id && !S.repoId) LS.set('repo', result.repo_id);        // preselect the new repo once it is listed
        showToast("Ingestion queued", result.message || repoPath, "success");
        await loadRepositories();
        switchTab('repos');
    } catch (err) {
        showToast("Ingestion failed", err.message, "error");
    } finally {
        clearInterval(ingestTimer); card.classList.add('hidden'); btn.disabled = false; renderRecentPaths();
    }
}

/* ================= Repositories ================= */
const DONE = /complete|ready|done|ingested|success/i, FAILED = /fail|error/i;
function statusChip(status) {
    const s = String(status || 'unknown');
    const cls = DONE.test(s) ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
        : FAILED.test(s) ? 'bg-red-500/10 text-red-400 border-red-500/20'
            : 'bg-amber-500/10 text-amber-400 border-amber-500/20 animate-pulse';
    return `<span class="px-2 py-1 rounded-full text-[10px] border ${cls}">${esc(s)}</span>`;
}

function renderRepos() {
    const q = ($('repo-filter')?.value || '').toLowerCase();
    const rows = S.repos.filter(r => `${r.repo_id} ${r.title} ${r.root_path}`.toLowerCase().includes(q));
    $('repo-table-body').innerHTML = rows.length ? rows.map(r => {
        const pct = r.total_files ? Math.round((r.processed_files / r.total_files) * 100) : 0;
        return `<tr class="hover:bg-slate-100 dark:hover:bg-slate-900/40">
            <td data-cell="name" class="py-3.5 px-5 font-mono text-xs"><span class="text-blue-600 dark:text-blue-400 font-semibold">${esc(r.repo_id)}</span><br><span class="text-slate-500 dark:text-slate-400">${esc(r.title)}</span></td>
            <td data-cell="status" class="py-3.5 px-5">${statusChip(r.status)}</td>
            <td data-cell="files" class="py-3.5 px-5 font-mono text-xs"><div class="whitespace-nowrap">${esc(r.processed_files)}/${esc(r.total_files)}</div><div class="repo-bar w-full min-w-[3rem] h-1 bg-slate-200 dark:bg-slate-800 rounded-full mt-1.5"><div class="h-1 bg-blue-500 rounded-full" style="width:${pct}%"></div></div></td>
            <td data-cell="chunks" class="py-3.5 px-5 font-mono text-xs text-blue-600 dark:text-blue-400 font-semibold">${esc(r.total_chunks)}</td>
            <td data-cell="path" class="py-3.5 px-5 font-mono text-xs text-slate-500 dark:text-slate-400 truncate max-w-xs" title="${esc(r.root_path)}">${esc(r.root_path)}</td>
            <td data-cell="actions" class="py-3.5 px-5">
                <div class="repo-actions flex justify-end gap-2">
                    <button data-chat-repo="${esc(r.repo_id)}" class="bg-blue-600/20 hover:bg-blue-600/30 text-blue-600 dark:text-blue-400 border border-blue-500/30 px-3 py-1.5 rounded-lg text-xs whitespace-nowrap">Open chat</button>
                    <button data-reset-repo="${esc(r.repo_id)}" class="bg-red-500/10 hover:bg-red-500/20 text-red-600 dark:text-red-400 border border-red-500/30 px-3 py-1.5 rounded-lg text-xs whitespace-nowrap transition-colors">Reset index</button>
                    <button data-remove-repo="${esc(r.repo_id)}" class="bg-red-500/10 hover:bg-red-500/20 text-red-600 dark:text-red-400 border border-red-500/30 px-3 py-1.5 rounded-lg text-xs whitespace-nowrap transition-colors">Remove</button>
                </div>
            </td>
        </tr>`;
    }).join('') : `<tr><td colspan="6" class="py-14 text-center text-sm text-slate-500">${S.repos.length ? 'No repositories match your filter.' : 'No repositories yet. <button class="text-blue-400 hover:underline" onclick="switchTab(\'ingest\')">Ingest your first one</button>.'}</td></tr>`;
}

async function resetRepoIndex(repoId) {
    if (!confirm(`Are you sure you want to reset the index for ${repoId}? This will permanently delete all search indexes and chunks for this repository.`)) return;

    try {
        const res = await fetch(`/api/v1/repos/${encodeURIComponent(repoId)}/reset-index`, { method: 'POST' });
        if (!res.ok) throw new Error(`Server returned ${res.status}`);

        // Assuming showToast is a global function in your app.js
        showToast("Index reset", `Successfully reset index for ${repoId}`, "success");

        // Refresh the list to show updated status
        await loadRepositories();
    } catch (err) {
        showToast("Reset failed", err.message, "error");
    }
}

document.addEventListener('click', async e => {
    const b = e.target.closest('[data-chat-repo]');
    if (b) quickChat(b.dataset.chatRepo);

    const rb = e.target.closest('[data-reset-repo]');
    if (rb) {
        await resetRepoIndex(rb.dataset.resetRepo);
    }

    const rm = e.target.closest('[data-remove-repo]');
    if (rm) {
        await removeRepo(rm.dataset.removeRepo);
    }

});

async function loadRepositories() {
    try {
        const res = await fetch('/api/v1/repos');
        if (!res.ok) throw new Error(`Server returned ${res.status}`);
        const data = await res.json();
        S.repos = data; //await res.json();

        if (data.used_docker !== undefined) {
            if (data.used_docker) {
                showToast("Sandbox", "Verification performed in Docker Sandbox ✅", "success");
            } else {
                showToast("Sandbox", "Docker unavailable: Verification skipped (Static Analysis only) ⚠️", "warning");
            }
        }

        $('nav-repo-count').textContent = S.repos.length;
        $('nav-repo-count').title = `${S.repos.length} repositories ingested`;
        renderRepos();

        const select = $('chat-repo-select');
        const keep = S.repoId || select.value || LS.get('repo', '');
        select.innerHTML = `<option value="">Select repository</option><option value="${GENERAL}">🌐 General (web)</option>` +
            S.repos.map(r => `<option value="${esc(r.repo_id)}">${esc(r.title)} (${esc(r.repo_id)})</option>`).join('');
        const pick = (keep === GENERAL || S.repos.some(r => String(r.repo_id) === String(keep))) ? keep : (S.repos.length === 1 ? S.repos[0].repo_id : '');
        select.value = pick;
        if (String(pick) !== String(S.repoId || '')) await onRepoChanged();
        else if (!S.repoId && !$('chat-col').children.length) showEmpty(S.repos.length ? 'pick' : 'none');

        clearTimeout(S.poll);
        if (S.tab === 'repos' && S.repos.some(r => !DONE.test(r.status) && !FAILED.test(r.status))) S.poll = setTimeout(loadRepositories, 4000);
    } catch (err) {
        console.error("Failed to load repositories:", err);
        showToast("Couldn't load repositories", err.message, "error");
    }
}

async function removeRepo(repoId) {
    if (!confirm(`Are you sure you want to permanently remove ${repoId}? This will delete all associated data and stop watching the repository.`)) return;

    try {
        const res = await fetch(`/api/v1/repos/${encodeURIComponent(repoId)}`, { method: 'DELETE' });
        if (!res.ok) throw new Error(`Server returned ${res.status}`);
        showToast("Repo removed", `Repository ${repoId} has been removed successfully.`, "success");
        await loadRepositories();
    } catch (e) {
        showToast("Error", e.message, "error");
    }
}

function quickChat(repoId) {
    switchTab('chat');
    $('chat-repo-select').value = repoId;
    onRepoChanged();
}

/* ================= Threads ================= */
function renderThreads(threads) {
    const list = $('thread-list');
    list.innerHTML = threads.length ? threads.map(t =>
        `<div id="thread_btn_${esc(t.thread_id)}" data-thread="${esc(t.thread_id)}" data-title="${esc(t.title)}" data-tip="${esc(t.title)}"  role="button" tabindex="0" class="thread-item px-2.5 py-2 rounded-lg cursor-pointer hover:bg-slate-200 dark:hover:bg-slate-800 text-xs text-slate-600 dark:text-slate-300 flex items-center gap-2 transition-colors">
            <i data-lucide="message-square" class="w-3.5 h-3.5 text-blue-600 dark:text-blue-400 shrink-0"></i><span class="truncate">${esc(t.title)}</span><button type="button" data-del="${esc(t.thread_id)}" title="Delete chat" aria-label="Delete chat" class="ml-auto p-1 shrink-0 text-slate-400 dark:text-slate-500 hover:text-red-500 dark:hover:text-red-400"><i data-lucide="trash-2" class="w-3.5 h-3.5"></i></button></div>`).join('')
        : '<div class="text-xs text-slate-500 text-center py-6">Select a repository to see its threads</div>';
    icons();
}

/* Thread title tooltip */
const threadTip = document.body.appendChild(Object.assign(document.createElement('div'), { id: 'thread-tip' }));
const hideThreadTip = () => { threadTip.style.display = 'none'; };

document.addEventListener('mouseover', e => {
    const item = e.target.closest('.thread-item');
    if (!item) return hideThreadTip();
    const label = item.querySelector('span.truncate');
    if (!label || label.scrollWidth <= label.clientWidth) return hideThreadTip(); // show only if truncated
    const r = item.getBoundingClientRect();
    threadTip.textContent = item.dataset.tip;
    threadTip.style.left = `${r.right + 8}px`;
    threadTip.style.top = `${r.top + r.height / 2}px`;
    threadTip.style.display = 'block';
});
document.addEventListener('scroll', hideThreadTip, true); // capture: catches #thread-list scroll

document.addEventListener('click', e => {
    const d = e.target.closest('[data-del]');
    if (d) return deleteThread(d.dataset.del);
    const t = e.target.closest('[data-thread]');
    if (t) selectThread(t.dataset.thread, t.dataset.title);
});

async function deleteThread(id) {
    if (!confirm('Delete this chat and all its messages? This cannot be undone.')) return;
    try {
        const res = await fetch(`/api/v1/threads/${encodeURIComponent(id)}`, { method: 'DELETE' });
        if (!res.ok) throw new Error(`Server returned ${res.status}`);
        await onRepoChanged(String(S.threadId) === String(id) ? null : S.threadId);
    } catch (err) {
        showToast("Couldn't delete chat", err.message, "error");
    }
}

document.addEventListener('keydown', e => {
    if ((e.key === 'Enter' || e.key === ' ') && e.target.matches?.('[data-thread]')) { e.preventDefault(); e.target.click(); }
});

async function onRepoChanged(preferThread) {
    S.repoId = $('chat-repo-select').value || null;
    LS.set('repo', S.repoId || '');
    setupNotificationStream();      // telemetry follows the selection: general events + this repo's events
    fetchSearchMode(streamRepo());
    if (!S.repoId) {
        S.threadId = null;
        renderThreads([]);
        $('active-thread-badge').textContent = 'No thread selected';
        showEmpty(S.repos.length ? 'pick' : 'none');
        return updateComposer();
    }
    try {
        const res = await fetch(`/api/v1/repos/${S.repoId}/threads`);
        if (!res.ok) throw new Error(`Server returned ${res.status}`);
        const threads = await res.json();
        if (!threads.length) return createThreadRequest('Session 1');
        renderThreads(threads);
        const want = threads.find(t => String(t.thread_id) === String(preferThread || LS.get('thread:' + S.repoId, ''))) || threads[0];
        await selectThread(want.thread_id, want.title);
    } catch (err) {
        showToast("Couldn't load threads", err.message, "error");
    }
}

function toggleNewThreadForm() {
    if (!S.repoId) {
        showToast("Repository required", "Select a repository first.", "warning");
        return;
    }
    if ($('thread-panel').dataset.open !== 'true') toggleThreads();
    const box = $('new-thread-box'), input = $('new-thread-title-input');
    box.classList.toggle('hidden');
    if (!box.classList.contains('hidden')) {
        input.value = `Thread ${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;
        input.focus(); input.select();
    }
}

async function submitNewThread() {
    const title = $('new-thread-title-input').value.trim() || "New conversation";
    $('new-thread-box').classList.add('hidden');
    await createThreadRequest(title);
}

async function createThreadRequest(title) {
    try {
        const res = await fetch('/api/v1/repos/threads', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ repo_id: S.repoId, title })
        });
        if (!res.ok) throw new Error(`Server returned ${res.status}`);
        const result = await res.json();
        await onRepoChanged(result.thread_id);
    } catch (err) {
        showToast("Couldn't create thread", err.message, "error");
    }
}

/* ================= Messages ================= */
function msgHtml(role, content, meta = '', atts = [], requestId = '') {
    if (role === 'user') {
        const chips = atts.length ? `<div class="flex flex-wrap gap-1.5 ${content ? 'mb-2' : ''}">${atts.map(a => `<span class="inline-flex items-center gap-1 bg-blue-800/60 rounded-md px-2 py-0.5 text-xs"><i data-lucide="${a.kind === 'image' ? 'image' : 'file-text'}" class="w-3 h-3"></i>${esc(a.filename)}</span>`).join('')}</div>` : '';
        return `<div class="flex justify-end"><div class="bg-blue-600 text-white rounded-2xl rounded-br-md px-4 py-2.5 text-sm max-w-[92%] sm:max-w-[85%] whitespace-pre-wrap break-words">${chips}${esc(content)}</div></div>`;
    }
    return `<div class="flex gap-3 group scroll-mt-4">
        <div class="msg-avatar w-7 h-7 mt-0.5 rounded-lg bg-blue-600/15 border border-blue-500/25 flex items-center justify-center text-blue-400 shrink-0"><i data-lucide="bot" class="w-4 h-4"></i></div>
        <div class="min-w-0 flex-1" data-raw="${esc(content)}">
            <div class="text-sm text-slate-700 dark:text-slate-200 leading-relaxed break-words">${Render.renderMarkdown(content)}</div>
            <div class="msg-actions mt-1 flex items-center gap-3 text-[11px] text-slate-500 opacity-0 group-hover:opacity-100 focus-within:opacity-100 transition-opacity">
                <button data-copy="msg" class="hover:text-slate-900 dark:hover:text-slate-200 flex items-center gap-1"><i data-lucide="copy" class="w-3 h-3"></i><span>Copy</span></button>${meta}
            </div>${requestId ? stepsPillHtml(requestId) : ''}
        </div></div>`;
}

function showEmpty(kind, title = '') {
    const col = $('chat-col');
    if (kind === 'thread') {
        col.innerHTML = `<div data-welcome class="pt-[5vh] sm:pt-[12vh] text-center space-y-5">
            <div class="mx-auto w-11 h-11 rounded-xl bg-blue-600/15 border border-blue-500/25 flex items-center justify-center text-blue-400"><i data-lucide="sparkles" class="w-5 h-5"></i></div>
            <div><h3 class="text-lg font-semibold text-slate-900 dark:text-white">Ask about ${esc(title)}</h3><p class="text-sm text-slate-500 dark:text-slate-400 mt-1">${S.repoId === GENERAL ? 'General questions. The web is searched when needed (see the Web chip below).' : 'Answers are grounded in the ingested chunks of this repository.'}</p></div>
            <div class="flex flex-wrap justify-center gap-2">${(S.repoId === GENERAL ? GENERAL_SUGGESTIONS : SUGGESTIONS).map(s => `<button type="button" data-suggest class="text-xs text-slate-600 dark:text-slate-300 bg-slate-100 dark:bg-slate-900 hover:bg-slate-200 dark:hover:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-full px-3.5 py-2 transition-colors">${esc(s)}</button>`).join('')}</div></div>`;
    } else {
        const none = kind === 'none';
        col.innerHTML = `<div data-welcome class="pt-[6vh] sm:pt-[14vh] text-center space-y-4">
            <div class="mx-auto w-11 h-11 rounded-xl bg-slate-200 dark:bg-slate-800 border border-slate-300 dark:border-slate-700 flex items-center justify-center text-slate-500 dark:text-slate-400"><i data-lucide="${none ? 'folder-git-2' : 'database'}" class="w-5 h-5"></i></div>
            <div><h3 class="text-lg font-semibold text-slate-900 dark:text-white">${none ? 'No repositories yet' : 'Choose a repository'}</h3>
            <p class="text-sm text-slate-500 dark:text-slate-400 mt-1">${none ? 'Ingest a codebase to start asking questions about it.' : 'Pick one from the selector above to load its threads.'}</p></div>
            ${none ? '<button onclick="switchTab(\'ingest\')" class="bg-blue-600 hover:bg-blue-500 text-white text-sm font-medium px-5 py-2.5 rounded-xl">Ingest a repository</button>' : ''}</div>`;
    }
    icons();
}
document.addEventListener('click', e => {
    const b = e.target.closest('[data-suggest]');
    if (b) { $('chat-input-query').value = b.textContent; sendChatMessage(); }
});

async function selectThread(threadId, threadTitle) {
    S.threadId = threadId;
    LS.set('thread:' + S.repoId, threadId);
    $('active-thread-badge').textContent = threadTitle;
    document.querySelectorAll('.thread-item').forEach(el => el.classList.toggle('active', el.dataset.thread === String(threadId)));
    if (isMobile()) $('thread-panel').dataset.open = 'false';
    updateComposer();

    const col = $('chat-col');
    col.innerHTML = '<div class="space-y-4 animate-pulse"><div class="h-4 w-2/3 bg-slate-800 rounded"></div><div class="h-4 w-1/2 bg-slate-800 rounded"></div><div class="h-4 w-3/5 bg-slate-800 rounded"></div></div>';
    try {
        const res = await fetch(`/api/v1/threads/${threadId}/messages`);
        if (!res.ok) throw new Error(`Server returned ${res.status}`);
        const messages = await res.json();
        if (S.threadId !== threadId) return; // user switched while loading
        if (!messages.length) return showEmpty('thread', threadTitle);
        col.innerHTML = messages.map(m => msgHtml(m.role === 'user' ? 'user' : 'assistant', m.content, '', m.attachments || [], m.request_id || '')).join('');
        icons();
        scrollDown(true);
        $('chat-input-query').focus();
    } catch (err) {
        col.innerHTML = '';
        showToast("Couldn't load messages", err.message, "error");
    }
}

/* ================= Composer & sending ================= */
function autosize() {
    const t = $('chat-input-query');
    t.style.height = 'auto';
    t.style.height = Math.min(t.scrollHeight, 208) + 'px';
}

function updateComposer() {
    const ready = !!(S.repoId && S.threadId);
    const t = $('chat-input-query');
    t.disabled = !ready;
    t.placeholder = ready ? (S.repoId === GENERAL ? 'Ask anything (uses the web when needed)' : 'Ask anything about the codebase') : (S.repoId ? 'Loading thread…' : 'Select a repository to start chatting');
    $('send-btn').disabled = !ready || S.sending;
    $('attach-btn').disabled = !ready || S.sending;
    const stopBtn = $('stop-chat-btn');
    const sendBtn = $('send-btn');
    if (stopBtn) {
        stopBtn.classList.toggle('hidden', !S.sending);
        stopBtn.classList.toggle('flex', S.sending);
        stopBtn.disabled = !S.sending;
    }
    if (sendBtn) {
        sendBtn.classList.toggle('hidden', S.sending);
        sendBtn.classList.toggle('flex', !S.sending);
    }
}

function setupComposer() {
    const t = $('chat-input-query');
    t.addEventListener('input', autosize);
    t.addEventListener('keydown', e => {
        if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); sendChatMessage(); }
    });
    updateComposer();
}

function distFromBottom() { const m = $('chat-messages'); return m.scrollHeight - m.scrollTop - m.clientHeight; }
function updateJumpBtn() { const b = $('jump-btn'); b.classList.toggle('hidden', distFromBottom() < 300); b.classList.toggle('flex', distFromBottom() >= 300); }
function scrollDown(force) {
    const m = $('chat-messages');
    if (force || distFromBottom() < 160) m.scrollTop = m.scrollHeight;
}

/* ================= Attachments (paste / pick / drop) ================= */
const MAX_FILES = 5, MAX_MB = 15, PASTE_AS_FILE_CHARS = 4000;
const isImg = f => (f.type || '').startsWith('image/');

function setupAttachments() {
    document.addEventListener('paste', e => {
        if (S.tab !== 'chat' || !S.repoId || !S.threadId || S.sending) return;
        const el = e.target;
        if (el.matches?.('input, textarea, [contenteditable]') && el.id !== 'chat-input-query') return;
        const cd = e.clipboardData; if (!cd) return;
        if (cd.files.length) { e.preventDefault(); return addFiles([...cd.files]); }
        const text = cd.getData('text/plain');
        if (text.length > PASTE_AS_FILE_CHARS) { // heavy text becomes an attachment instead of flooding the box
            e.preventDefault();
            addFiles([new File([text], `pasted-text-${S.files.length + 1}.txt`, { type: 'text/plain' })]);
        }
    });
    $('attach-input').addEventListener('change', e => { addFiles([...e.target.files]); e.target.value = ''; });
    const form = $('chat-form');
    form.addEventListener('dragover', e => e.preventDefault());
    form.addEventListener('drop', e => { e.preventDefault(); if (S.repoId && S.threadId && !S.sending) addFiles([...e.dataTransfer.files]); });
    $('attach-strip').addEventListener('click', e => {
        const b = e.target.closest('[data-rm-file]'); if (!b) return;
        const [f] = S.files.splice(+b.dataset.rmFile, 1);
        if (f?._url) URL.revokeObjectURL(f._url);
        renderAttachStrip();
    });
}

function addFiles(list) {
    for (const f of list) {
        if (S.files.length >= MAX_FILES) { showToast('Limit reached', `Max ${MAX_FILES} attachments per message.`, 'warning'); break; }
        if (f.size > MAX_MB * 1048576) { showToast('File too large', `${f.name} exceeds ${MAX_MB} MB.`, 'warning'); continue; }
        // clipboard screenshots arrive as "image.png"; give them a unique readable name
        S.files.push(!f.name || f.name === 'image.png'
            ? new File([f], `pasted-image-${S.files.length + 1}-${Date.now() % 100000}.${(f.type.split('/')[1] || 'png')}`, { type: f.type })
            : f);
    }
    renderAttachStrip();
}

function renderAttachStrip() {
    const strip = $('attach-strip');
    strip.classList.toggle('hidden', !S.files.length);
    strip.classList.toggle('flex', !!S.files.length);
    strip.innerHTML = S.files.map((f, i) => {
        if (isImg(f)) f._url = f._url || URL.createObjectURL(f);
        const thumb = isImg(f) ? `<img src="${f._url}" alt="" class="w-8 h-8 rounded object-cover">` : `<i data-lucide="file-text" class="w-4 h-4 text-slate-400 dark:text-slate-500 mx-1"></i>`;
        return `<div class="flex items-center gap-2 bg-slate-100 dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-lg pl-1 pr-1.5 py-1 text-xs text-slate-600 dark:text-slate-300 max-w-[220px]">${thumb}<span class="truncate">${esc(f.name)}</span><button type="button" data-rm-file="${i}" class="text-slate-400 dark:text-slate-500 hover:text-slate-900 dark:hover:text-white px-1" aria-label="Remove ${esc(f.name)}">✕</button></div>`;
    }).join('');
    icons();
}

/* Runner switch: the list comes from the server's registry, so a newly registered runner shows up by itself. */
async function loadRunners() {
    const sel = $('runner-select');
    if (!sel) { loadModels(); return; }
    try {
        const r = await (await fetch('/api/v1/runners')).json();
        const list = r.runners || [];
        S.runnerLabels = {}; S.runnerTimeouts = {}; list.forEach(x => { S.runnerLabels[x.name] = x.label || x.name; S.runnerTimeouts[x.name] = x.timeout_s; });
        sel.innerHTML = `<option value="">Default (${esc(runnerLabel(r.default))})</option>` + list.map(x =>
            `<option value="${esc(x.name)}"${x.configured ? '' : ' disabled'} title="${esc(x.configured ? `${x.label || x.name} (${x.name}): ${x.remote ? 'remote / paid runner' : 'local runner'}` : (x.error || 'not configured'))}">${esc(x.label || x.name)}${x.remote ? ' ☁' : ''}${x.configured ? '' : ' (not set up)'}</option>`).join('');
        S.defaultRunner = r.default; S.serverTokenMode = r.token_mode || 'auto';
        const ok = list.find(x => x.name === S.runner && x.configured);
        if (S.runner && !ok) { S.runner = ''; LS.set('runner', ''); }      // saved runner is gone or no longer configured
        sel.value = S.runner;
        sel.title = sel.ariaLabel = `Runner (chat model backend; default ${runnerLabel(r.default)})`;
        initTokenMode();
    } catch { /* stays on Default */ }
    sel.onchange = () => { S.runner = sel.value; LS.set('runner', S.runner); loadModels(); };         // a model not offered by the new runner falls back to Default inside loadModels
    loadModels();
}

function runnerLabel(name) { return (S.runnerLabels && S.runnerLabels[name]) || name || 'default'; }       // human label from /api/v1/runners; the id (name) is what requests and telemetry carry

function initTokenMode() {
    const sel = $('token-mode-select'); if (!sel) return;
    sel.value = S.tokenMode || S.serverTokenMode || 'auto';
    sel.onchange = () => { S.tokenMode = sel.value; LS.set('tokenMode', S.tokenMode); };
    updateSettingsChip();
}

/* Chat settings chip: runner / model / token mode stay collapsed behind one chip so the message box keeps its width. */
function updateSettingsChip() {
    const chip = $('settings-chip'), lab = $('settings-chip-label'), badge = $('settings-chip-mode');
    if (!chip || !lab) return;
    const picked = id => { const s = $(id); const o = s && s.selectedOptions && s.selectedOptions[0]; return o ? o.textContent.trim() : ''; };
    const model = picked('model-select').replace(/^Default \((.*)\)$/, '$1') || 'Default model';
    const runner = S.runner || S.defaultRunner || 'default';
    const mode = ($('token-mode-select') || {}).value || 'auto';
    lab.textContent = model;
    if (badge) { badge.textContent = mode; badge.classList.toggle('hidden', mode === 'auto'); }
    chip.title = `Runner: ${runnerLabel(runner)} (${runner}) · Model: ${model} · Tokens: ${mode}. Click to change.`;
}

function toggleChatSettings(open) {
    const panel = $('chat-settings-panel'), chip = $('settings-chip'); if (!panel || !chip) return;
    const show = open === undefined ? panel.classList.contains('hidden') : !!open;
    panel.classList.toggle('hidden', !show);
    chip.setAttribute('aria-expanded', String(show));
    if (show) { const first = $('runner-select'); if (first) first.focus(); }
}

function wireChatSettings() {
    ['runner-select', 'model-select', 'token-mode-select'].forEach(id => { const el = $(id); if (el) el.addEventListener('change', updateSettingsChip); });
    const tt = $('task-timer-toggle');       // "Show task timer": the live elapsed-time toast
    if (tt) { tt.checked = taskTimerOn(); tt.addEventListener('change', () => { LS.set('taskTimer', tt.checked); if (!tt.checked) taskEnd('hide'); }); }
    document.addEventListener('click', e => {
        const box = $('chat-settings');
        if (box && !box.contains(e.target)) toggleChatSettings(false);
    });
    document.addEventListener('keydown', e => {
        const panel = $('chat-settings-panel');
        if (e.key === 'Escape' && panel && !panel.classList.contains('hidden')) { toggleChatSettings(false); const c = $('settings-chip'); if (c) c.focus(); }
    });
}

function shortModel(m) { return String(m).replace(/^@cf\/[^/]+\//, ''); }       // @cf/google/gemma-4-26b-a4b-it -> gemma-4-26b-a4b-it

async function loadModels() {
    const sel = $('model-select'); if (!sel) return;
    try {
        let res = await fetch('/api/v1/models' + (S.runner ? `?runner=${encodeURIComponent(S.runner)}` : ''));
        if (res.status === 400 && S.runner) {      // the picked runner is not usable (any more): back to the server default
            S.runner = ''; LS.set('runner', ''); const rs = $('runner-select'); if (rs) rs.value = '';
            res = await fetch('/api/v1/models');
        }
        const r = await res.json();
        // The list comes from the chosen runner (the switch, else LLM_RUNNER): Ollama tags or the Cloudflare CF_MODELS list.
        const runner = r.runner || 'ollama';
        const short = shortModel;
        S.defaultModel = r.default;
        const models = (r.models || []).filter(m => m !== r.default);
        sel.innerHTML = `<option value="" title="${esc(r.default)} (${esc(runnerLabel(runner))})">Default (${esc(short(r.default))})</option>` +
            `<optgroup label="${esc(runnerLabel(runner))} models">` +
            models.map(m => `<option value="${esc(m)}" title="${esc(m)}">${esc(short(m))}</option>`).join('') +
            `</optgroup>`;
        sel.title = sel.ariaLabel = `Reasoning model (runner: ${runnerLabel(runner)})`;
        sel.dataset.runner = runner;
        // A pick saved under another runner (e.g. an Ollama tag after switching to Cloudflare) falls back to Default.
        if (S.model && S.model !== r.default && !models.includes(S.model)) { S.model = ''; LS.set('model', ''); }
        sel.value = S.model === r.default ? '' : S.model;
    } catch { /* stays on Default */ }
    sel.onchange = () => { S.model = sel.value; LS.set('model', S.model); };
    updateSettingsChip();
}

/* Stop: shared by the composer's stop button and the task toast. */
async function stopChat() {
    const ctl = S.abortController;
    if (!ctl) return;
    try {
        const params = new URLSearchParams({ thread_id: S.threadId });
        if (S.repoId) params.set('repo_id', S.repoId);
        const res = await fetch(`/api/v1/chat/stop?${params.toString()}`, { method: 'POST' });
        if (res.ok) {
            ctl.abort();
            showToast("Cancelled", "Chat request terminated.", "info");
        } else if (res.status === 404) {
            ctl.abort();
        }
    } catch (err) {
        console.error("Stop request failed:", err);
        ctl.abort();
    }
}

/* ---- Task toast: a live elapsed-time counter while a chat / change request runs, against the runner's per-call timeout.
   Colour: < 50% of the limit blue, 50-90% amber, > 90% red. A multi-step task legitimately outlives one call's limit: when the
   activity stream reports a finished LLM call (action "llm_call") for THIS request, the per-call part restarts ("step N · this call
   m:ss / T"), the total keeps running. Toggle: "Show task timer" in the chat settings (localStorage crag.taskTimer, default on). ---- */
const TASK_LIMIT_FALLBACK_S = 120;
const taskTimerOn = () => LS.get('taskTimer', true) !== false;
const fmtClock = s => { s = Math.max(0, Math.floor(s)); return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`; };

function taskLimit() {
    const t = S.runnerTimeouts && S.runnerTimeouts[S.runner || S.defaultRunner];
    return { s: t > 0 ? t : TASK_LIMIT_FALLBACK_S, known: t > 0 };
}

function taskTone(ratio) { return ratio > 0.9 ? 'red' : ratio >= 0.5 ? 'amber' : 'blue'; }
const TASK_TONES = { blue: ['bg-blue-500', 'text-blue-500'], amber: ['bg-amber-500', 'text-amber-500'], red: ['bg-red-500', 'text-red-500'], green: ['bg-emerald-500', 'text-emerald-500'] };

function taskRender() {
    const T = S.task; if (!T || !T.el) return;
    const now = performance.now(), total = (now - T.t0) / 1000, call = (now - T.callT0) / 1000;
    const ratio = call / T.limit;
    const tone = TASK_TONES[taskTone(ratio)];
    const q = sel => T.el.querySelector(sel);
    const stepped = T.step > 0;
    q('[data-t=time]').textContent = stepped ? `step ${T.step + 1} · this call ${fmtClock(call)} / ${fmtClock(T.limit)} · total ${fmtClock(total)}` : fmtClock(total);
    q('[data-t=time]').className = `text-[11px] font-mono mt-0.5 ${tone[1]}`;
    const bar = q('[data-t=bar]'); bar.style.width = Math.min(100, ratio * 100).toFixed(1) + '%';
    bar.className = `h-full ${tone[0]} transition-[width] duration-700`;
    q('[data-t=accent]').className = `w-1 ${tone[0]} shrink-0`;
    q('[data-t=note]').textContent = ratio > 1
        ? (stepped ? `This call is past the ${T.limit}s limit — it may be retrying. ` : '') + "Longer than one call's limit — the task may be on a later step or retrying."
        : (T.known ? `limit ${T.limit}s per model call` : `about ${T.limit}s per model call (no hard limit on this runner)`) + (T.lastMs ? ` · last call ${fmtClock(T.lastMs / 1000)}` : '');
}

function taskStart() {
    taskEnd('hide');
    if (!taskTimerOn()) return;
    const container = $('toast-container'); if (!container) return;
    const lim = taskLimit();
    const modelName = S.model ? shortModel(S.model) : (S.defaultModel ? shortModel(S.defaultModel) : 'default model');
    const el = document.createElement('div');
    el.className = 'pointer-events-auto flex rounded-xl border border-slate-300 dark:border-slate-700 shadow-2xl bg-white dark:bg-slate-800 text-slate-900 dark:text-white overflow-hidden';
    el.setAttribute('role', 'status');
    el.innerHTML = `<span data-t="accent" class="w-1 bg-blue-500 shrink-0"></span><div class="flex-1 min-w-0 px-3.5 py-3">
        <div class="flex items-start justify-between gap-2"><h5 data-t="title" class="text-xs font-semibold min-w-0 break-words">${esc(`Working · ${runnerLabel(S.runner || S.defaultRunner)} · ${modelName}`)}</h5>
            <button type="button" data-t="stop" class="shrink-0 text-[10px] font-medium px-2 py-0.5 rounded-md bg-red-600 hover:bg-red-500 text-white" title="Stop this request">Stop</button></div>
        <p data-t="time" class="text-[11px] font-mono mt-0.5">0:00</p>
        <div class="h-1 rounded bg-slate-200 dark:bg-slate-700 mt-1.5 overflow-hidden"><div data-t="bar" class="h-full w-0 bg-blue-500"></div></div>
        <p data-t="note" class="text-[10px] text-slate-600 dark:text-slate-300 mt-1 break-words"></p></div>`;
    el.querySelector('[data-t=stop]').onclick = () => stopChat();
    container.appendChild(el);
    const now = performance.now();
    S.task = { el, t0: now, callT0: now, step: 0, lastMs: 0, limit: lim.s, known: lim.known, uids: new Set(), timer: setInterval(taskRender, 1000) };
    taskRender();
}

/* An LLM call of this request finished (Activity event, kind action "llm_call"): the next call starts now. */
function taskStep(ev) {
    const T = S.task; if (!T) return;
    if (ev.uid != null) { if (T.uids.has(String(ev.uid))) return; T.uids.add(String(ev.uid)); }
    T.step += 1; T.callT0 = performance.now(); T.lastMs = Number(ev.duration_ms) || 0;
    taskRender();
}

/* outcome: 'done' -> "Done in m:ss" (gone after 3 s); 'failed' -> "Failed after m:ss" (stays until clicked); 'hide'/'stopped' -> removed. */
function taskEnd(outcome) {
    const T = S.task; if (!T) return;
    clearInterval(T.timer); S.task = null;
    const el = T.el; if (!el) return;
    const total = fmtClock((performance.now() - T.t0) / 1000);
    if (outcome !== 'done' && outcome !== 'failed') { el.remove(); return; }
    const failed = outcome === 'failed', tone = TASK_TONES[failed ? 'red' : 'green'];
    el.querySelector('[data-t=accent]').className = `w-1 ${tone[0]} shrink-0`;
    el.querySelector('[data-t=title]').textContent = failed ? `Failed after ${total}` : `Done in ${total}`;
    el.querySelector('[data-t=title]').className = `text-xs font-semibold min-w-0 break-words ${tone[1]}`;
    ['[data-t=stop]', '[data-t=time]', '[data-t=note]'].forEach(s => el.querySelector(s).remove());
    el.querySelector('[data-t=bar]').parentElement.remove();
    if (failed) { el.classList.add('cursor-pointer'); el.title = 'Click to dismiss'; el.onclick = () => el.remove(); }
    else setTimeout(() => el.remove(), 3000);
}

async function sendChatMessage(e) {
    if (e) e.preventDefault();
    if (S.sending) return;

    S.abortController = new AbortController();
    const stopBtn = $('stop-chat-btn');
    if (stopBtn) stopBtn.onclick = stopChat;
    if (!S.repoId || !S.threadId) {
        showToast("Selection required", "Select a repository and thread first.", "warning");
        return;
    }
    const input = $('chat-input-query');
    const query = input.value.trim();
    const files = S.files.slice();
    if (!query && !files.length) return;

    const col = $('chat-col');
    col.querySelector('[data-welcome]')?.remove();
    col.insertAdjacentHTML('beforeend', msgHtml('user', query, '', files.map(f => ({ filename: f.name, kind: isImg(f) ? 'image' : 'file' }))));
    input.value = ''; autosize();
    S.files = []; renderAttachStrip();
    S.sending = true; updateComposer();

    const id = 'loader_' + Date.now();
    // col.insertAdjacentHTML('beforeend', `<div id="${id}" class="flex gap-3"><div class="msg-avatar w-7 h-7 rounded-lg bg-blue-600/15 border border-blue-500/25 flex items-center justify-center text-blue-400 shrink-0"><i data-lucide="bot" class="w-4 h-4"></i></div>
    //     <div class="flex items-center gap-2.5 text-sm text-slate-400 pt-0.5"><span class="think flex gap-1"><span></span><span></span><span></span></span>${files.length ? 'Reading attachments' : (S.repoId === GENERAL ? 'Thinking' : 'Retrieving relevant chunks')}${S.web !== 'off' ? ' · may search the web' : ''} <span id="${id}_t" class="font-mono text-xs text-slate-600">0s</span></div></div>`);
    col.insertAdjacentHTML('beforeend', `<div id="${id}" class="flex gap-3"><div class="msg-avatar w-7 h-7 rounded-lg bg-blue-600/15 border border-blue-500/25 flex items-center justify-center text-blue-400 shrink-0"><i data-lucide="bot" class="w-4 h-4"></i></div>
        <div class="flex flex-col gap-1 text-sm text-slate-400 pt-0.5">
            <div class="flex items-center gap-2.5"><span class="think flex gap-1"><span></span><span></span><span></span></span><span id="${id}_s" class="min-w-0 break-words">${files.length ? 'Reading attachments' : (S.repoId === GENERAL ? 'Thinking' : 'Retrieving relevant chunks')}${S.web !== 'off' ? ' · may search the web' : ''}</span> <span id="${id}_t" class="font-mono text-xs text-slate-600 shrink-0">0s</span></div>
            <div id="${id}_h" class="hidden pl-5 border-l border-slate-300 dark:border-slate-700"></div>
            <div id="${id}_a" class="act-live hidden" aria-label="Live steps"></div>
        </div></div></div>`);
    icons(); scrollDown(true);
    S.liveStatus = id + '_s';
    S.live = { id, threadId: S.threadId, requestId: null, uids: new Set(), query };
    const t0 = performance.now();
    const tick = setInterval(() => { const el = $(id + '_t'); if (el) el.textContent = Math.floor((performance.now() - t0) / 1000) + 's'; }, 1000);
    let outcome = 'done';
    taskStart();

    try {
        let res;
        const signal = S.abortController.signal;
        if (files.length) {
            const fd = new FormData();
            fd.append('repo_id', S.repoId); fd.append('thread_id', S.threadId); fd.append('query', query); fd.append('web', S.web); fd.append('model', S.model); fd.append('runner', S.runner); fd.append('token_mode', S.tokenMode);
            files.forEach(f => fd.append('files', f, f.name));
            res = await fetch('/api/v1/chat/attach', { method: 'POST', body: fd, signal });
        } else {
            res = await fetch('/api/v1/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ repo_id: S.repoId, thread_id: S.threadId, query, web: S.web, model: S.model, runner: S.runner, token_mode: S.tokenMode }),
                signal
            });
        }
        if (res.status === 499) {
            outcome = 'stopped';
            $(id)?.remove();
            showToast("Stopped", "Request cancelled by user.", "info");
            return;
        }
        if (!res.ok) {          // 402 / 401 / 504 / 503 / 500 ...: show the server's reason (JSON detail, else the text), never nothing
            const failure = await describeFailure(res);
            const er = new Error(failure.message); er.failure = failure; throw er;
        }
        const result = await res.json();
        files.forEach(f => f._url && URL.revokeObjectURL(f._url));
        $(id)?.remove();
        const rid = result.request_id || S.live?.requestId || '';
        if (rid) { S.act.done.add(rid); S.act.queries[rid] = query; scheduleRender(); }
        col.insertAdjacentHTML('beforeend', msgHtml('assistant', result.answer, `<span>${((performance.now() - t0) / 1000).toFixed(1)}s</span>`, [], rid));
        icons();
        col.lastElementChild.scrollIntoView({ block: 'start' }); // start reading at the top of the answer
        if (result && result.base_system) {
            const connLabel = document.getElementById('conn-label');
            if (connLabel) {
                connLabel.innerText = `Live (${result.base_system})`;
            }
        }

    } catch (err) {
        if (err.name === 'AbortError') {
            outcome = 'stopped';
            $(id)?.remove();
            showToast("Stopped", "Request cancelled by user.", "info");
        } else {            // HTTP error (err.failure) or the fetch itself failed (server down, connection reset)
            outcome = 'failed';
            $(id)?.remove();
            input.value = query; autosize();
            S.files = files; renderAttachStrip();
            showFailure(failureOf(err));
        }
    } finally {
        taskEnd(outcome);
        if (S.live?.requestId) S.act.done.add(S.live.requestId);   // stopped/failed: nothing more will close its open steps
        clearInterval(tick); S.liveStatus = null; S.live = null;
        S.sending = false; S.abortController = null; updateComposer(); input.focus();
    }
}

// Web chip: Auto -> On -> Off. Auto = only when needed; a leading /web forces it for one message.
function renderWebBtn() {
    const m = S.web, b = $('web-btn');
    if (!b) return;
    b.dataset.mode = m;
    $('web-btn-label').textContent = '🌐 ' + { auto: 'Auto', on: 'On', off: 'Off' }[m];
}
function cycleWeb() {
    S.web = { auto: 'on', on: 'off', off: 'auto' }[S.web] || 'auto';
    LS.set('web', S.web); renderWebBtn();
}
window.cycleWeb = cycleWeb;
renderWebBtn();

renderRecentPaths();

// "Apply to repo" links in change proposals POST to the API instead of navigating.
document.addEventListener('click', async e => {
    const a = e.target.closest('a[href$="/apply"]');
    if (!a) return;
    e.preventDefault();
    if (a.dataset.busy) return;
    a.dataset.busy = '1';
    const label = a.textContent;
    try {
        const post = async (force) => {
            const headers = phrase ? { 'X-Confirm-Phrase': phrase } : {};
            const r = await fetch(a.getAttribute('href') + (force ? '?force=true' : ''), { method: 'POST', headers });
            // const body = await r.json();
            const body = await r.json().catch(() => ({})); 
            return { ok: r.ok, status: r.status, body: body };
        };
        // try {
        a.textContent = 'Applying...';
        let phrase = '';
        let res = await post(false);
        console.log('Apply response:', res);
        if (res.status === 428) {                       // server wants the phrase: ask once, reuse it
            phrase = prompt('Enter the confirmation phrase to apply to the repo:');
            if (phrase === null) { a.textContent = label; return; }
            res = await post(false, phrase);
        }
        if (res.status === 409 && (res.body.conflicts || []).length) {
            const names = res.body.conflicts.map(c => c.path).join('\n');
            if (confirm('These files changed on disk since ingest:\n' + names + '\n\nOverwrite them anyway?')) res = await post(true);
            else { a.textContent = label; return; }
        }
        // a.textContent = res.ok ? `Applied (${res.body.written.length + res.body.deleted.length} file(s))`
        //                        : 'Apply failed: ' + (res.body.detail || res.body.note || res.status);
        if (res.ok) {
            const nApplied = res.body.count ?? ((res.body.written || []).length + (res.body.deleted || []).length);
            a.textContent = `Applied ${nApplied} file(s)`;
            showToast('Success', `Successfully applied changes to the repository.`, 'success');

            if (res.body && res.body.used_docker !== undefined) {
                if (res.body.used_docker) {
                    showToast("Sandbox", "Verification performed in Docker Sandbox ✅", "success");
                } else {
                    showToast("Sandbox", "Docker unavailable: Verification skipped (Static Analysis only) ⚠️", "warning");
                }
            }
            
        } else {
            const errorMsg = res.body.detail || res.body.note || `Error code: ${res.status}`;
            a.textContent = 'Apply failed';
            showToast('Apply Failed', errorMsg, 'error');
        }
    } catch (err) {
        a.textContent = 'Apply failed';
        showToast('Apply Failed', err.message, 'error');
    } finally {
        delete a.dataset.busy;
    }
});

/* ================= Agent activity (timeline, live steps, per-answer steps) ================= */
/* Pure merge/group/filter logic lives in static/js/activity.js (AX). This section is DOM + state only. */
const ACT_EVENT_CAP = 8000;     // events kept in memory
const ACT_DOM_CAP = 1500;       // rows rendered at once
const LIVE_ROWS = 12;           // rows shown in the pending-answer panel

function parseMaybeJson(v) { if (typeof v !== 'string') return v; try { return JSON.parse(v); } catch { return null; } }

function addActionEvents(list) {
    const A = S.act;
    for (const ev of list) {
        if (!ev || typeof ev !== 'object' || ev.uid == null) continue;
        A.events.set(String(ev.uid), ev);
    }
    if (A.events.size > ACT_EVENT_CAP) {                        // Map iterates in insertion order: drop the oldest 10 %
        let drop = Math.ceil(ACT_EVENT_CAP * 0.1);
        for (const k of A.events.keys()) { if (drop-- <= 0) break; A.events.delete(k); }
    }
    A.dirty = true;
    scheduleRender();
}

function liveMatches(ev) {
    const L = S.live;
    if (!L) return false;
    if (L.requestId && ev.request_id && String(ev.request_id) === String(L.requestId)) return true;
    return !!ev.thread_id && String(ev.thread_id) === String(L.threadId);   // request_id is only known once the response lands
}

function onActionEvent(raw) {
    const ev = parseMaybeJson(raw);
    if (!ev || typeof ev !== 'object') return;
    addActionEvents([ev]);
    const L = S.live;
    if (!S.sending || !L || !liveMatches(ev)) return;
    if (ev.action === 'llm_call') taskStep(ev);         // a model call of this request finished: the per-call timer restarts
    if (ev.uid != null) L.uids.add(String(ev.uid));
    if (!L.requestId && ev.request_id) { L.requestId = ev.request_id; S.act.queries[ev.request_id] = L.query; }
    S.act.liveDirty = true; scheduleRender();
    const status = AX.normStatus(ev.status);
    if (status !== 'ok' && status !== 'skip') updateLiveStatus([ev.phase, ev.agent, ev.action].filter(Boolean).join(' · '));
}

async function loadActionHistory() {
    try {
        const qs = new URLSearchParams({ limit: '500' });
        if (streamRepo()) qs.set('repo_id', streamRepo());
        const res = await fetch('/api/v1/actions?' + qs);
        if (!res.ok) return;
        const body = await res.json();
        addActionEvents(body.events || []);
    } catch { /* older backend without /actions: the timeline fills from SSE only */ }
}

/* ---- rendering helpers ---- */
function actStatusIcon(r) {
    if (r.open) return '<span class="act-spin" role="img" aria-label="running"></span>';
    const m = { ok: ['✓', 'ok'], fail: ['✗', 'failed'], skip: ['–', 'skipped'], info: ['•', 'info'] }[r.status] || ['•', 'info'];
    return `<span class="act-ic act-${esc(r.status)}" role="img" aria-label="${m[1]}">${m[0]}</span>`;
}
function actPhaseChip(phase) { return `<span class="act-phase" style="--ph:${AX.phaseHue(phase)}">${esc(phase || '—')}</span>`; }
function actTime(r) { return r.startMs ? fmtTs(new Date(r.startMs).toISOString()) : ''; }

function actJson(data) {
    if (data == null || (typeof data === 'object' && !Object.keys(data).length)) return '';
    let txt;
    try { txt = JSON.stringify(data, null, 2); } catch { txt = String(data); }
    return txt.length > 20000 ? txt.slice(0, 20000) + '\n… (truncated)' : txt;
}
function actDetailHtml(r) {
    const json = actJson(r.data);
    const times = `${esc(actTime(r))}${r.endMs ? ' → ' + esc(fmtTs(new Date(r.endMs).toISOString())) : ''}`;
    return `<div class="act-row-d">
        ${r.detail ? `<p class="act-detail-full">${esc(r.detail)}</p>` : ''}
        <p class="act-meta">${times}${r.request_id ? ' · request ' + esc(r.request_id) : ''}${r.thread_id ? ' · thread ' + esc(r.thread_id) : ''}${r.model ? ' · ' + esc(r.model) : ''}</p>
        ${json ? `<pre class="act-json">${esc(json)}</pre>` : '<p class="act-meta">No data payload.</p>'}
    </div>`;
}

/* One merged row. `inline` rows (mini timelines) render their detail up-front (hidden); the big list renders it only when expanded. */
function actRowHtml(r, inline) {
    S.rowIndex.set(r.uid, r);
    const open = S.act.rowOpen.has(r.uid);
    const label = [r.phase, r.agent, r.action].filter(Boolean).join(' ');
    return `<div class="act-row" data-status="${r.open ? 'start' : esc(r.status)}">
        <button type="button" class="act-row-h" data-act-row="${esc(r.uid)}" aria-expanded="${open}" aria-label="${esc(label)}: toggle details">
            ${actStatusIcon(r)}${actPhaseChip(r.phase)}
            <span class="act-agent">${esc(r.agent)}</span><span class="act-action">${esc(r.action)}</span>
            ${r.model ? `<span class="act-model">${esc(r.model)}</span>` : ''}
            ${r.detail ? `<span class="act-detail">${esc(r.detail)}</span>` : ''}
            <span class="act-dur">${esc(AX.fmtDuration(r.duration_ms))}</span>
        </button>${open || inline ? actDetailHtml(r).replace('class="act-row-d"', open ? 'class="act-row-d"' : 'class="act-row-d" hidden') : ''}
    </div>`;
}

/* Rows whose request already answered but never got a closing event would spin forever: show them as skipped. */
function settleRows(rows) {
    rows.forEach(r => { if (r.open && r.request_id && S.act.done.has(r.request_id)) { r.open = false; r.status = 'skip'; } });
    return rows;
}

function scheduleRender() {
    if (S.act.raf) return;
    const run = () => {
        S.act.raf = 0;
        if (S.act.dirty && S.tab === 'activity') renderActivity();
        if (S.act.liveDirty) renderLive();
    };
    S.act.raf = (window.requestAnimationFrame || (f => setTimeout(f, 16)))(run);
}

function actGroupOpen(g, firstKey, filtering) {
    if (S.act.groupOpen.has(g.key)) return S.act.groupOpen.get(g.key);
    return g.status === 'running' || g.key === firstKey || filtering;
}

function renderActivity() {
    const A = S.act, list = $('activity-list');
    A.dirty = false;
    if (!list) return;
    const focusKey = document.activeElement?.closest?.('#activity-list') ? (document.activeElement.dataset.actRow || document.activeElement.dataset.actGroup) : null;
    const rows = settleRows(AX.mergeActionEvents([...A.events.values()]));
    const filtering = !!(A.phases.size || A.failuresOnly || A.text.trim() || A.thisThread);
    const filtered = AX.filterRows(rows, { phases: A.phases, failuresOnly: A.failuresOnly, text: A.text, threadId: A.thisThread ? S.threadId : null });
    const groups = AX.groupByRequest(filtered, { queries: A.queries });
    A.hasRunning = groups.some(g => g.status === 'running');
    const vis = AX.selectVisible(groups, g => actGroupOpen(g, groups[0]?.key, filtering), ACT_DOM_CAP);
    S.rowIndex.clear();

    const sum = $('activity-summary');
    if (sum) sum.textContent = rows.length
        ? `${groups.length} request${groups.length === 1 ? '' : 's'} · ${filtered.length}${filtered.length !== rows.length ? ' of ' + rows.length : ''} steps${vis.hidden ? ` · ${vis.hidden} oldest rows hidden (showing newest ${ACT_DOM_CAP})` : ''}`
        : 'No activity yet. Steps appear here live while the assistant works.';

    if (!groups.length) {
        list.innerHTML = `<div class="act-empty">${rows.length ? 'No steps match the current filters.' : 'No agent activity recorded yet.'}</div>`;
        return;
    }
    list.innerHTML = vis.groups.map(({ group: g, rows: gr, open, truncated }) => {
        const c = g.counts;
        const counts = `${c.total} step${c.total === 1 ? '' : 's'}${c.fail ? ` · <span class="act-fail-txt">${c.fail} failed</span>` : ''}${c.skip ? ` · ${c.skip} skipped` : ''}${c.open ? ` · ${c.open} running` : ''}`;
        const status = { running: ['running', 'Running'], fail: ['fail', 'Failed'], ok: ['ok', 'Done'] }[g.status];
        return `<section class="act-group" data-status="${status[0]}">
            <button type="button" class="act-group-h" data-act-group="${esc(g.key)}" aria-expanded="${open}" aria-label="${esc(g.label)}: ${status[1]}, ${c.total} steps">
                <span class="act-chevron" aria-hidden="true">▸</span>
                <span class="act-group-title">${esc(g.label)}</span>
                <span class="act-group-meta">${esc(fmtTs(new Date(g.startMs).toISOString()))}</span>
                <span class="act-group-meta">${esc(AX.fmtDuration(g.durationMs))}</span>
                <span class="act-badge act-badge-${status[0]}">${status[1]}</span>
                <span class="act-group-meta act-group-counts">${counts}</span>
            </button>
            ${open ? `<div class="act-rows">${truncated ? `<div class="act-empty">${truncated} older steps not shown</div>` : ''}${gr.map(r => actRowHtml(r, false)).join('')}</div>` : ''}
        </section>`;
    }).join('');
    if (focusKey) [...list.querySelectorAll('[data-act-row],[data-act-group]')].find(b => (b.dataset.actRow || b.dataset.actGroup) === focusKey)?.focus({ preventScroll: true });
}

function renderLive() {
    const L = S.live, A = S.act;
    A.liveDirty = false;
    const el = L && $(L.id + '_a');
    if (!el) return;
    const events = [...L.uids].map(u => A.events.get(u)).filter(Boolean);
    const rows = AX.mergeActionEvents(events);
    el.classList.toggle('hidden', !rows.length);
    if (!rows.length) return;
    const stick = el.scrollHeight - el.scrollTop - el.clientHeight < 24;
    const keep = rows.slice(-LIVE_ROWS);
    el.innerHTML = `<div class="act-live-h">Working · ${rows.length} step${rows.length === 1 ? '' : 's'}${rows.length > keep.length ? ' (latest ' + keep.length + ')' : ''}</div>` + keep.map(r => actRowHtml(r, true)).join('');
    if (stick) el.scrollTop = el.scrollHeight;
}

/* ---- Activity tab controls ---- */
function buildActivityControls() {
    const box = $('activity-phases');
    if (!box) return;
    box.innerHTML = AX.PHASES.map(p => `<button type="button" class="act-phase act-phase-btn" style="--ph:${AX.phaseHue(p)}" data-phase="${p}" aria-pressed="false">${p}</button>`).join('');
    box.addEventListener('click', e => {
        const b = e.target.closest('[data-phase]');
        if (!b) return;
        const on = !S.act.phases.has(b.dataset.phase);
        on ? S.act.phases.add(b.dataset.phase) : S.act.phases.delete(b.dataset.phase);
        b.setAttribute('aria-pressed', String(on));
        onActivityFilter();
    });
    $('activity-list').addEventListener('click', e => {
        const rb = e.target.closest('[data-act-row]');
        if (rb) return toggleActRow(rb);
        const gb = e.target.closest('[data-act-group]');
        if (gb) { S.act.groupOpen.set(gb.dataset.actGroup, gb.getAttribute('aria-expanded') !== 'true'); S.act.dirty = true; renderActivity(); }
    });
}

function onActivityFilter() {
    const A = S.act;
    A.failuresOnly = !!$('activity-failures')?.checked;
    A.thisThread = !!$('activity-thread')?.checked;
    A.text = $('activity-text')?.value || '';
    A.dirty = true; scheduleRender();
}
function clearActivity() { S.act.events.clear(); S.act.rowOpen.clear(); S.act.groupOpen.clear(); S.act.dirty = true; renderActivity(); }
async function reloadActivity() { await loadActionHistory(); showToast('Activity', 'Timeline refreshed.', 'info'); }

/* Expand/collapse one row anywhere (activity list, live panel, per-answer steps). Detail HTML is created lazily. */
function toggleActRow(btn) {
    const key = btn.dataset.actRow, open = btn.getAttribute('aria-expanded') !== 'true';
    btn.setAttribute('aria-expanded', String(open));
    open ? S.act.rowOpen.add(key) : S.act.rowOpen.delete(key);
    let d = btn.nextElementSibling;
    if (d && !d.classList.contains('act-row-d')) d = null;
    if (!d && open && S.rowIndex.has(key)) {
        btn.insertAdjacentHTML('afterend', actDetailHtml(S.rowIndex.get(key)));
        d = btn.nextElementSibling;
    }
    if (d) d.hidden = !open;
}
document.addEventListener('click', e => {         // rows inside the live panel and per-answer step panels
    const rb = e.target.closest('.act-live [data-act-row], .steps-panel [data-act-row]');
    if (rb) toggleActRow(rb);
});

/* ---- Per-answer "Steps (n)" ---- */
function stepRowsFor(rid) {
    const evs = [...S.act.events.values()].filter(ev => String(ev.request_id) === String(rid));
    return settleRows(AX.mergeActionEvents(evs));
}
function stepsPillHtml(rid) {
    const n = stepRowsFor(rid).length;
    return `<div class="steps-wrap"><button type="button" class="steps-pill" data-steps="${esc(rid)}" aria-expanded="false" aria-label="Show the steps taken for this answer"><i data-lucide="list-checks" class="w-3 h-3"></i><span>${n ? `Steps (${n})` : 'Steps'}</span></button>
        <div class="steps-panel hidden" role="region" aria-label="Steps taken"></div></div>`;
}
function renderStepsPanel(panel, rid) {
    const rows = stepRowsFor(rid);
    panel.innerHTML = rows.length ? rows.map(r => actRowHtml(r, true)).join('') : '<div class="act-empty">No recorded steps for this answer.</div>';
    const label = panel.parentElement.querySelector('.steps-pill span');
    if (label && rows.length) label.textContent = `Steps (${rows.length})`;
}
document.addEventListener('click', async e => {
    const btn = e.target.closest('[data-steps]');
    if (!btn) return;
    const panel = btn.parentElement.querySelector('.steps-panel'), rid = btn.dataset.steps;
    const open = btn.getAttribute('aria-expanded') !== 'true';
    btn.setAttribute('aria-expanded', String(open));
    panel.classList.toggle('hidden', !open);
    if (!open) return;
    renderStepsPanel(panel, rid);                       // whatever streamed in live shows immediately
    if (panel.dataset.loaded) return;
    panel.insertAdjacentHTML('afterbegin', '<div class="act-empty" data-loading>Loading steps…</div>');
    try {
        const res = await fetch('/api/v1/actions?' + new URLSearchParams({ request_id: rid, limit: '1000' }));
        if (!res.ok) throw new Error(`Server returned ${res.status}`);
        S.act.done.add(rid);
        addActionEvents((await res.json()).events || []);
        panel.dataset.loaded = '1';
    } catch (err) {
        panel.querySelector('[data-loading]')?.remove();
        panel.insertAdjacentHTML('afterbegin', `<div class="act-empty">Couldn't load steps: ${esc(err.message)}</div>`);
        return;
    }
    renderStepsPanel(panel, rid);
});

/* ================= Search mode + vectorize jobs ================= */
function repoLabel(id) { const r = S.repos.find(x => String(x.repo_id) === String(id)); return r ? r.title : id; }

async function fetchSearchMode(repoId) {
    if (!repoId) { S.searchMode = null; S.searchJob = null; return renderSearchMode(); }
    try {
        const res = await fetch(`/api/v1/repos/${encodeURIComponent(repoId)}/search-mode`);
        if (!res.ok) throw new Error(`Server returned ${res.status}`);
        const mode = await res.json();
        if (streamRepo() !== repoId) return;                                      // user moved on
        S.searchMode = mode; S.searchJob = null;
        const job = AX.normalizeJob(mode.job);
        if (job && job.job_id) S.jobs[job.job_id] = job;                           // known: no toast on later duplicates
    } catch { S.searchMode = null; }                                               // older backend: hide the badge
    renderSearchMode();
}

function renderSearchMode() {
    const box = $('search-mode');
    if (!box) return;
    const info = S.searchMode ? AX.searchModeInfo(S.searchMode, S.searchJob) : null;
    box.classList.toggle('hidden', !info);
    box.classList.toggle('flex', !!info);
    if (!info) return;
    box.dataset.kind = info.kind;
    $('search-mode-label').textContent = info.label;
    const failMsg = info.kind === 'failed' && info.job?.error ? ` — ${info.job.error}` : '';
    $('search-mode-badge').title = `Search mode: ${info.label}${failMsg}`;
    const bar = $('search-mode-bar');
    bar.classList.toggle('hidden', info.kind !== 'vectorizing');
    bar.dataset.indeterminate = String(info.kind === 'vectorizing' && info.pct == null);
    $('search-mode-fill').style.width = (info.pct == null ? 100 : info.pct) + '%';
    bar.setAttribute('aria-valuenow', String(info.pct == null ? 0 : info.pct));
    const btn = $('vectorize-btn');
    btn.classList.toggle('hidden', !info.canBuild);
    btn.classList.toggle('inline-flex', info.canBuild);
    btn.querySelector('span').textContent = info.kind === 'failed' ? 'Retry index' : 'Build vector index';
}

/* A job update arrived (SSE, history replay or the vectorize response). Replays never toast. */
function applyJob(raw, replay) {
    const job = AX.normalizeJob(parseMaybeJson(raw));
    if (!job || !job.job_id) return;
    const prev = S.jobs[job.job_id];
    S.jobs[job.job_id] = job;
    if (!replay) {
        const name = repoLabel(job.repo_id);
        if (!prev && AX.isActiveJob(job)) showToast('Vector index', `Vectorizing ${name} in the background${job.reason ? ' (' + job.reason + ')' : ''}. You'll be notified when it finishes.`, 'info');
        if (AX.isTerminalJob(job) && !(prev && prev.state === job.state)) {
            if (job.state === 'COMPLETED') showToast('Vector index ready', `Semantic search is now enabled for ${name}.`, 'success');
            else if (job.state === 'FAILED') showToast('Vectorize failed', `${name}: ${job.error || 'unknown error'}`, 'error');
            else showToast('Vectorize cancelled', name, 'warning');
        }
    }
    if (String(job.repo_id) !== String(streamRepo())) return;
    S.searchJob = job;
    if (!S.searchMode) S.searchMode = { vector_status: 'none' };
    renderSearchMode();
    if (!replay && AX.isTerminalJob(job)) fetchSearchMode(streamRepo());           // authoritative status after a job ends
}

async function buildVectorIndex() {
    const repoId = streamRepo();
    if (!repoId) return;
    if (!confirm(`Build a vector index for "${repoId}"?\n\nThis is a long-running background job: it can take several minutes on large repositories. You can keep chatting (agentic search keeps working) and you will be notified here when it finishes.`)) return;
    const btn = $('vectorize-btn');
    btn.disabled = true;
    try {
        const res = await fetch(`/api/v1/repos/${encodeURIComponent(repoId)}/vectorize`, { method: 'POST' });
        const job = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(job.detail || `Server returned ${res.status}`);
        applyJob(job, false);
    } catch (err) {
        showToast("Couldn't start vectorize", err.message, 'error');
    } finally {
        btn.disabled = false;
    }
}
