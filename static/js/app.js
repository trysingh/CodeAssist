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
            .replace(/([.:;!?])\s+(\*\s+\*\*)/g, "$1\n$2");

        const lines = this.escapeHtml(normalized).split("\n");
        const out = [];
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
                out.push(`<div class="flex items-start space-x-2 my-1.5 pl-2"><span class="text-blue-400 font-bold">•</span><div class="flex-1">${this.formatInline(ul[2])}</div></div>`);
                continue;
            }

            const ol = line.match(/^(\s*)(\d+\.)\s+(.*)$/);
            if (ol) {
                out.push(`<div class="flex items-start space-x-2 my-1.5 pl-2"><span class="text-blue-400 font-mono text-xs font-bold">${ol[2]}</span><div class="flex-1">${this.formatInline(ol[3])}</div></div>`);
                continue;
            }

            out.push(this.formatInline(line) + (i < lines.length - 1 ? "<br>" : ""));
        }
        if (inTable) out.push("</tbody></table></div>");
        return out.join("");
    },
    renderMarkdown: function (text) {
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
const TABS = ['chat', 'repos', 'ingest', 'telemetry'];
const TITLES = { chat: 'Chat', repos: 'Repositories', ingest: 'Ingest repository', telemetry: 'Telemetry & logs' };
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
    loadModels();
    syncLimitSelect(S.tlimit);
    setupNotificationStream();
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
        if (el.id === 'tab-telemetry') el.classList.toggle('flex', on);
    });
    document.querySelectorAll('#sidebar nav button').forEach(b => {
        const on = b.id === 'nav-' + tabId;
        b.classList.toggle('active', on);
        on ? b.setAttribute('aria-current', 'page') : b.removeAttribute('aria-current');
    });
    $('current-tab-title').textContent = TITLES[tabId];
    $('current-tab-title').classList.toggle('hidden', tabId === 'chat');
    $('chat-ctx').classList.toggle('hidden', tabId !== 'chat');
    $('chat-ctx').classList.toggle('flex', tabId === 'chat');
    $('focus-btn').classList.toggle('lg:block', tabId === 'chat');
    $('focus-btn').classList.toggle('lg:hidden', tabId !== 'chat');
    if (tabId !== 'chat') document.body.classList.remove('focus');
    if (location.hash !== '#' + tabId) history.replaceState(null, '', '#' + tabId);
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
        const n = { Digit1: 0, Digit2: 1, Digit3: 2, Digit4: 3 }[e.code];
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
function showToast(title, message, type = 'info') {
    const container = $('toast-container');
    if (!container) return;
    const bar = { success: 'bg-emerald-500', error: 'bg-red-500', warning: 'bg-amber-500' }[type] || 'bg-blue-500';
    const toast = document.createElement('div');
    toast.className = 'pointer-events-auto flex rounded-xl border border-slate-300 dark:border-slate-700 shadow-2xl bg-white dark:bg-slate-800 text-slate-900 dark:text-white overflow-hidden cursor-pointer';
    toast.innerHTML = `<span class="w-1 ${bar} shrink-0"></span><div class="flex-1 px-3.5 py-3"><h5 class="text-xs font-semibold">${esc(title)}</h5><p class="text-[11px] text-slate-600 dark:text-slate-300 mt-0.5 break-words">${esc(message)}</p></div>`;
    toast.onclick = () => toast.remove();
    container.appendChild(toast);
    setTimeout(() => toast.remove(), 4500);
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
            return `${t.phase} · ${t.model} · in ${t.input_tokens} / out ${t.output_tokens} / total ${t.total_tokens} tok · ${t.duration_sec}s`;
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

function setupNotificationStream() {
    const repo = streamRepo();
    const key = `${repo}|${S.tlimit}`;
    if (S.stream && S.streamKey === key) return;           // same scope: keep the open stream
    if (S.stream) S.stream.close();
    S.streamKey = key;
    renderScopeHint();

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
            (data.events || []).forEach(ev => frag.prepend(telemetryLine(ev)));    // oldest first in -> newest on top
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
            const level = String(data.level || 'info').toLowerCase();
            // showToast(data.title, prettyMessage(data), level);
            const st = S.liveStatus && $(S.liveStatus);
            if (st) {                                  // request in flight: show its latest step in the loader
                const msg = `${data.title} — ${prettyMessage(data)}`;
                st.textContent = msg.length > 160 ? msg.slice(0, 157) + '…' : msg;
                st.title = msg;                        // hover for the full text
            }
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
            <td class="py-3.5 px-5 font-mono text-xs"><span class="text-blue-600 dark:text-blue-400 font-semibold">${esc(r.repo_id)}</span><br><span class="text-slate-500 dark:text-slate-400">${esc(r.title)}</span></td>
            <td class="py-3.5 px-5">${statusChip(r.status)}</td>
            <td class="py-3.5 px-5 font-mono text-xs"><div>${esc(r.processed_files)}/${esc(r.total_files)}</div><div class="w-20 h-1 bg-slate-200 dark:bg-slate-800 rounded-full mt-1.5"><div class="h-1 bg-blue-500 rounded-full" style="width:${pct}%"></div></div></td>
            <td class="py-3.5 px-5 font-mono text-xs text-blue-600 dark:text-blue-400 font-semibold">${esc(r.total_chunks)}</td>
            <td class="py-3.5 px-5 font-mono text-xs text-slate-500 dark:text-slate-400 truncate max-w-xs" title="${esc(r.root_path)}">${esc(r.root_path)}</td>
            <td class="py-3.5 px-5 text-right"><button data-chat-repo="${esc(r.repo_id)}" class="bg-blue-600/20 hover:bg-blue-600/30 text-blue-600 dark:text-blue-400 border border-blue-500/30 px-3 py-1.5 rounded-lg text-xs">Open chat</button></td>
            <td class="py-3 px-4 text-right">
                <button data-reset-repo="${esc(r.repo_id)}" 
                        class="text-xs bg-red-900/30 hover:bg-red-900/50 text-red-400 border border-red-800/50 px-2 py-1 rounded transition-colors">
                    Reset Index
                </button>
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

});

async function loadRepositories() {
    try {
        const res = await fetch('/api/v1/repos');
        if (!res.ok) throw new Error(`Server returned ${res.status}`);
        S.repos = await res.json();
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
function msgHtml(role, content, meta = '', atts = []) {
    if (role === 'user') {
        const chips = atts.length ? `<div class="flex flex-wrap gap-1.5 ${content ? 'mb-2' : ''}">${atts.map(a => `<span class="inline-flex items-center gap-1 bg-blue-800/60 rounded-md px-2 py-0.5 text-xs"><i data-lucide="${a.kind === 'image' ? 'image' : 'file-text'}" class="w-3 h-3"></i>${esc(a.filename)}</span>`).join('')}</div>` : '';
        return `<div class="flex justify-end"><div class="bg-blue-600 text-white rounded-2xl rounded-br-md px-4 py-2.5 text-sm max-w-[85%] whitespace-pre-wrap break-words">${chips}${esc(content)}</div></div>`;
    }
    return `<div class="flex gap-3 group scroll-mt-4">
        <div class="w-7 h-7 mt-0.5 rounded-lg bg-blue-600/15 border border-blue-500/25 flex items-center justify-center text-blue-400 shrink-0"><i data-lucide="bot" class="w-4 h-4"></i></div>
        <div class="min-w-0 flex-1" data-raw="${esc(content)}">
            <div class="text-sm text-slate-700 dark:text-slate-200 leading-relaxed break-words">${Render.renderMarkdown(content)}</div>
            <div class="mt-1 flex items-center gap-3 text-[11px] text-slate-500 opacity-0 group-hover:opacity-100 focus-within:opacity-100 transition-opacity">
                <button data-copy="msg" class="hover:text-slate-900 dark:hover:text-slate-200 flex items-center gap-1"><i data-lucide="copy" class="w-3 h-3"></i><span>Copy</span></button>${meta}
            </div>
        </div></div>`;
}

function showEmpty(kind, title = '') {
    const col = $('chat-col');
    if (kind === 'thread') {
        col.innerHTML = `<div data-welcome class="pt-[12vh] text-center space-y-5">
            <div class="mx-auto w-11 h-11 rounded-xl bg-blue-600/15 border border-blue-500/25 flex items-center justify-center text-blue-400"><i data-lucide="sparkles" class="w-5 h-5"></i></div>
            <div><h3 class="text-lg font-semibold text-slate-900 dark:text-white">Ask about ${esc(title)}</h3><p class="text-sm text-slate-500 dark:text-slate-400 mt-1">${S.repoId === GENERAL ? 'General questions. The web is searched when needed (see the Web chip below).' : 'Answers are grounded in the ingested chunks of this repository.'}</p></div>
            <div class="flex flex-wrap justify-center gap-2">${(S.repoId === GENERAL ? GENERAL_SUGGESTIONS : SUGGESTIONS).map(s => `<button type="button" data-suggest class="text-xs text-slate-600 dark:text-slate-300 bg-slate-100 dark:bg-slate-900 hover:bg-slate-200 dark:hover:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-full px-3.5 py-2 transition-colors">${esc(s)}</button>`).join('')}</div></div>`;
    } else {
        const none = kind === 'none';
        col.innerHTML = `<div data-welcome class="pt-[14vh] text-center space-y-4">
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
        col.innerHTML = messages.map(m => msgHtml(m.role === 'user' ? 'user' : 'assistant', m.content, '', m.attachments || [])).join('');
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

async function loadModels() {
    const sel = $('model-select'); if (!sel) return;
    try {
        const r = await (await fetch('/api/v1/models')).json();
        sel.innerHTML = `<option value="">Default (${esc(r.default)})</option>` +
            r.models.map(m => `<option value="${esc(m)}">${esc(m)}</option>`).join('');
        if (!r.models.includes(S.model)) S.model = '';
        sel.value = S.model;
    } catch { /* stays on Default */ }
    sel.onchange = () => { S.model = sel.value; LS.set('model', S.model); };
}

async function sendChatMessage(e) {
    if (e) e.preventDefault();
    if (S.sending) return;

    S.abortController = new AbortController();
    const stopBtn = $('stop-chat-btn');
    if (stopBtn) {
        stopBtn.onclick = async () => {
            try {
                const params = new URLSearchParams({ thread_id: S.threadId });
                if (S.repoId) params.set('repo_id', S.repoId);
                const res = await fetch(`/api/v1/chat/stop?${params.toString()}`, { method: 'POST' });
                if (res.ok) {
                    S.abortController.abort();
                    showToast("Cancelled", "Chat request terminated.", "info");
                } else if (res.status === 404) {
                    S.abortController.abort();
                }
            } catch (err) {
                console.error("Stop request failed:", err);
                S.abortController.abort();
            }
        };
    }
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
    // col.insertAdjacentHTML('beforeend', `<div id="${id}" class="flex gap-3"><div class="w-7 h-7 rounded-lg bg-blue-600/15 border border-blue-500/25 flex items-center justify-center text-blue-400 shrink-0"><i data-lucide="bot" class="w-4 h-4"></i></div>
    //     <div class="flex items-center gap-2.5 text-sm text-slate-400 pt-0.5"><span class="think flex gap-1"><span></span><span></span><span></span></span>${files.length ? 'Reading attachments' : (S.repoId === GENERAL ? 'Thinking' : 'Retrieving relevant chunks')}${S.web !== 'off' ? ' · may search the web' : ''} <span id="${id}_t" class="font-mono text-xs text-slate-600">0s</span></div></div>`);
    col.insertAdjacentHTML('beforeend', `<div id="${id}" class="flex gap-3"><div class="w-7 h-7 rounded-lg bg-blue-600/15 border border-blue-500/25 flex items-center justify-center text-blue-400 shrink-0"><i data-lucide="bot" class="w-4 h-4"></i></div>
        <div class="flex items-center gap-2.5 text-sm text-slate-400 pt-0.5"><span class="think flex gap-1"><span></span><span></span><span></span></span><span id="${id}_s" class="min-w-0 break-words">${files.length ? 'Reading attachments' : (S.repoId === GENERAL ? 'Thinking' : 'Retrieving relevant chunks')}${S.web !== 'off' ? ' · may search the web' : ''}</span> <span id="${id}_t" class="font-mono text-xs text-slate-600 shrink-0">0s</span></div></div>`);
    icons(); scrollDown(true);
    S.liveStatus = id + '_s';
    const t0 = performance.now();
    const tick = setInterval(() => { const el = $(id + '_t'); if (el) el.textContent = Math.floor((performance.now() - t0) / 1000) + 's'; }, 1000);

    try {
        let res;
        const signal = S.abortController.signal;
        if (files.length) {
            const fd = new FormData();
            fd.append('repo_id', S.repoId); fd.append('thread_id', S.threadId); fd.append('query', query); fd.append('web', S.web); fd.append('model', S.model);
            files.forEach(f => fd.append('files', f, f.name));
            res = await fetch('/api/v1/chat/attach', { method: 'POST', body: fd, signal });
        } else {
            res = await fetch('/api/v1/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ repo_id: S.repoId, thread_id: S.threadId, query, web: S.web, model: S.model }),
                signal
            });
        }
        if (res.status === 499) {
            $(id)?.remove();
            showToast("Stopped", "Request cancelled by user.", "info");
            return;
        }
        if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || `Server returned ${res.status}`);
        const result = await res.json();
        files.forEach(f => f._url && URL.revokeObjectURL(f._url));
        $(id)?.remove();
        col.insertAdjacentHTML('beforeend', msgHtml('assistant', result.answer, `<span>${((performance.now() - t0) / 1000).toFixed(1)}s</span>`));
        icons();
        col.lastElementChild.scrollIntoView({ block: 'start' }); // start reading at the top of the answer
    } catch (err) {
        if (err.name === 'AbortError') {
            $(id)?.remove();
            showToast("Stopped", "Request cancelled by user.", "info");
        } else {
            $(id)?.remove();
            input.value = query; autosize();
            S.files = files; renderAttachStrip();
            showToast("Message failed", err.message, "error");
        }
    } finally {
        clearInterval(tick); S.liveStatus = null;
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
            return { ok: r.ok, status: r.status, body: await r.json().catch(() => ({})) };
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
            a.textContent = `Applied ${res.body.count || 0} files`;
            showToast('Success', `Successfully applied changes to the repository.`, 'success');
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
