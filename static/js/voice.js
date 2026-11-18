/* ==========================================================================
   Voice input for the prompt box (and optional read-aloud). Vanilla, no dependencies.
   Loaded after app.js; exposed as window.CodeAssistVoice (and via module.exports so tests/js/test_voice_trigger.mjs can test the pure parts).
   Server: services/speech/ + views/voice_routes.py. Setup: docs/VOICE_SETUP.md.

   Flow: mic click (or Ctrl/Cmd+Shift+M) -> record -> end of speech after `pause_ms` of silence -> text is APPENDED to #chat-input-query for review.
   Only a spoken trigger phrase at the END ("go ahead now", "ab bhejo", ...) also submits. Plain end-of-speech never submits.
   Server mode: record 16 kHz wav -> POST /api/v1/voice/transcribe (transcribe + translate to clean English). Browser mode: Web Speech API,
   then POST /api/v1/voice/refine for the translation. Everything here is fail-safe: an exception must never affect typing or sending.
   ========================================================================== */
(function (root, factory) {
    const api = factory(root);
    if (typeof module !== 'undefined' && module.exports) module.exports = api;
    root.CodeAssistVoice = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function (root) {
    'use strict';

    const warn = (...a) => { try { console.warn('[voice]', ...a); } catch (_) { /* no console */ } };

    /* ------------------------------------------------------------------ pure helpers (mirror services/speech/triggers.py) */
    const DEFAULT_PHRASES = ['go ahead now', 'process now', 'go ahead', 'submit now', 'send it now', 'ab bhejo', 'process karo', 'bhej do', 'अब भेजो', 'प्रोसेस करो'];
    const FILLERS = new Set(['please', 'pls', 'plz', 'thanks', 'thank', 'you', 'thankyou', 'kindly', 'ok', 'okay', 'ji', 'yaar', 'na', 'bas', 'जी', 'यार', 'ना', 'बस']);
    const TRIM = /[\s,;:\-–—…]+$/u;

    function tokens(text) {      // [{w, s, e}] words = letters / combining marks / digits, over the ORIGINAL string
        const out = [];
        const re = /[\p{L}\p{M}\p{N}]+/gu;
        let m;
        while ((m = re.exec(text)) !== null) out.push({ w: m[0].normalize('NFKC').toLowerCase(), s: m.index, e: m.index + m[0].length });
        return out;
    }

    function parsePhrases(phrases) {      // 'a b, c' or ['a b', 'c'] -> token lists, longest first
        const raw = Array.isArray(phrases) ? phrases : String(phrases == null ? '' : phrases).split(',');
        const seen = new Set(), out = [];
        raw.forEach(p => {
            const t = tokens(String(p)).map(x => x.w);
            const key = t.join('\u0001');
            if (t.length && !seen.has(key)) { seen.add(key); out.push(t); }
        });
        return out.sort((a, b) => b.length - a.length);
    }

    /** splitTrigger(text, phrases) -> {text, triggered}. Only a phrase at the END counts (fillers like "please" may follow it). */
    function splitTrigger(text, phrases) {
        text = String(text == null ? '' : text).trim();
        const plist = parsePhrases(phrases === undefined || phrases === null ? DEFAULT_PHRASES : phrases);
        if (!text || !plist.length) return { text, triggered: false };
        let cur = text, triggered = false;
        for (let pass = 0; pass < 3; pass++) {
            const tk = tokens(cur);
            let end = tk.length;
            while (end > 0 && FILLERS.has(tk[end - 1].w)) end--;
            let hit = 0;
            for (const ph of plist) {
                const k = ph.length;
                if (end >= k && tk.slice(end - k, end).every((t, i) => t.w === ph[i])) { hit = k; break; }
            }
            if (!hit) break;
            triggered = true;
            cur = cur.slice(0, tk[end - hit].s).replace(TRIM, '');
        }
        return { text: triggered ? cur.trim() : text, triggered };
    }

    /** Float32 mono at `inRate` -> 16 kHz mono 16-bit PCM wav (Uint8Array). Box-filter downsampling (averages each output sample's input span). */
    function encodeWav(samples, inRate, outRate) {
        outRate = outRate || 16000;
        let pcm;
        if (!inRate || inRate <= outRate) {
            pcm = samples; outRate = inRate || outRate;
        } else {
            const ratio = inRate / outRate, n = Math.floor(samples.length / ratio);
            pcm = new Float32Array(n);
            for (let i = 0; i < n; i++) {
                const a = Math.floor(i * ratio), b = Math.min(samples.length, Math.max(a + 1, Math.floor((i + 1) * ratio)));
                let sum = 0;
                for (let j = a; j < b; j++) sum += samples[j];
                pcm[i] = sum / (b - a);
            }
        }
        const buf = new ArrayBuffer(44 + pcm.length * 2), v = new DataView(buf);
        const str = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
        str(0, 'RIFF'); v.setUint32(4, 36 + pcm.length * 2, true); str(8, 'WAVE'); str(12, 'fmt ');
        v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
        v.setUint32(24, outRate, true); v.setUint32(28, outRate * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
        str(36, 'data'); v.setUint32(40, pcm.length * 2, true);
        for (let i = 0; i < pcm.length; i++) {
            const s = Math.max(-1, Math.min(1, pcm[i]));
            v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
        }
        return new Uint8Array(buf);
    }

    const rms = (buf) => { let s = 0; for (let i = 0; i < buf.length; i++) s += buf[i] * buf[i]; return Math.sqrt(s / (buf.length || 1)); };

    /** Text for read-aloud: no code blocks, links or markdown noise. */
    function speakable(md) {
        return String(md || '')
            .replace(/```[\s\S]*?```/g, ' (code omitted). ')
            .replace(/`([^`]*)`/g, '$1')
            .replace(/!\[[^\]]*\]\([^)]*\)/g, ' ')
            .replace(/\[([^\]]*)\]\([^)]*\)/g, '$1')
            .replace(/^\s{0,3}(#{1,6}|[-*+>]|\d+[.)])\s+/gm, '')
            .replace(/[*_~|]{1,3}/g, '')
            .replace(/\s+/g, ' ').trim().slice(0, 4000);
    }

    const pure = { splitTrigger, parsePhrases, encodeWav, rms, speakable, DEFAULT_PHRASES };
    if (typeof document === 'undefined') return pure;        // node: pure parts only

    /* ------------------------------------------------------------------ state */
    const $ = id => document.getElementById(id);
    const DEFAULT_CFG = { enabled: true, mode: 'browser', providers: [], pause_ms: 2500, max_seconds: 120, trigger_phrases: DEFAULT_PHRASES, translate: true, tts: { enabled: false, provider: 'off' } };
    const NO_SPEECH_MS = 8000, CALIBRATE_MS = 300, FETCH_TIMEOUT_MS = 60000;
    const ls = {
        get(k, d) { try { return (typeof LS !== 'undefined') ? LS.get(k, d) : (JSON.parse(localStorage.getItem('crag.' + k)) ?? d); } catch (_) { return d; } },
        set(k, v) { try { if (typeof LS !== 'undefined') LS.set(k, v); else localStorage.setItem('crag.' + k, JSON.stringify(v)); } catch (_) { /* storage unavailable */ } }
    };
    const V = { cfg: null, cfgRunner: null, state: 'idle', session: null, sessionBrowser: false, audio: null };
    const SR = root.SpeechRecognition || root.webkitSpeechRecognition || null;
    const hasGUM = () => !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia && (root.AudioContext || root.webkitAudioContext));
    const runnerName = () => { try { return (typeof S !== 'undefined' && S.runner) || ''; } catch (_) { return ''; } };
    const cfg = () => V.cfg || DEFAULT_CFG;
    const toast = (title, msg, type) => { try { if (typeof showToast === 'function') showToast(title, msg, type || 'info'); } catch (_) { /* ignore */ } };

    function mode() {          // 'server' | 'browser' | 'none'
        const c = cfg();
        if (c.mode === 'server' && !V.sessionBrowser && hasGUM()) return 'server';
        return SR ? 'browser' : 'none';
    }

    /* ------------------------------------------------------------------ UI */
    function setStatus(text, kind) {
        const el = $('voice-status');
        if (!el) return;
        el.textContent = text || '';
        el.classList.toggle('hidden', !text);
        el.dataset.kind = kind || 'info';
    }
    function setState(state) {
        V.state = state;
        const b = $('mic-btn');
        if (!b) return;
        b.dataset.state = state;
        b.setAttribute('aria-pressed', String(state === 'listening'));
        b.setAttribute('aria-label', state === 'listening' ? 'Stop voice input' : 'Voice input');
        $('mic-ico-idle')?.classList.toggle('hidden', state === 'processing');
        $('mic-ico-busy')?.classList.toggle('hidden', state !== 'processing');
        if (state !== 'listening') b.style.removeProperty('--lvl');
    }
    function applyAvailability() {
        const b = $('mic-btn');
        if (!b) return;
        const usable = cfg().enabled !== false && mode() !== 'none';
        b.classList.toggle('hidden', !usable);
        b.style.display = usable ? '' : 'none';
    }
    function level(v) { const b = $('mic-btn'); if (b) b.style.setProperty('--lvl', Math.min(1, v * 8).toFixed(2)); }

    function insertText(text) {
        const ta = $('chat-input-query');
        if (!ta || !text) return;
        const cur = ta.value || '';
        ta.value = cur + (cur && !/\s$/.test(cur) ? ' ' : '') + text;
        ta.dispatchEvent(new Event('input', { bubbles: true }));
        ta.focus();
        try { ta.setSelectionRange(ta.value.length, ta.value.length); } catch (_) { /* not focusable */ }
    }
    function submitPrompt() {
        const form = $('chat-form');
        if (!form) return;
        try {
            if (typeof S !== 'undefined' && S.sending) { toast('Voice', 'Still working on the previous request: your text is in the box, press Enter to send it.', 'warning'); return; }
        } catch (_) { /* S not visible: let the normal path decide */ }
        if (typeof form.requestSubmit === 'function') form.requestSubmit();
        else $('send-btn')?.click();
    }
    /** Final step of every path: put the text in the box; submit ONLY when a trigger phrase ended the utterance and text remains. */
    function deliver(text, triggered) {
        text = (text || '').trim();
        if (text) insertText(text);
        const ta = $('chat-input-query');
        if (triggered && text && ta && ta.value.trim()) { setStatus('Sending…', 'info'); submitPrompt(); }
        else if (triggered && !text) setStatus('Nothing to send: say your request first.', 'warn');
        else if (!text) setStatus("Didn't catch that. Click the mic and try again.", 'warn');
        else { setStatus('Added to the prompt. Edit it, then press Enter to send.', 'info'); setTimeout(clearInfoStatus, 4000); }
    }

    /* ------------------------------------------------------------------ config */
    async function loadConfig() {
        const runner = runnerName();
        try {
            const res = await fetch('/api/v1/voice/config' + (runner ? '?runner=' + encodeURIComponent(runner) : ''));
            if (!res.ok) throw new Error('HTTP ' + res.status);
            V.cfg = Object.assign({}, DEFAULT_CFG, await res.json());
        } catch (e) {
            warn('config failed, using browser defaults', e);
            V.cfg = Object.assign({}, DEFAULT_CFG);
        }
        V.cfgRunner = runner;
        V.sessionBrowser = false;                       // a new runner may have a working server provider
        applyAvailability();
    }
    const phrases = () => cfg().trigger_phrases || DEFAULT_PHRASES;
    const lang = () => ls.get('voiceLang', 'auto');
    const srLang = () => ({ auto: 'en-IN', en: 'en-IN', hi: 'hi-IN', pa: 'pa-IN' }[lang()] || 'en-IN');

    /* ------------------------------------------------------------------ the session: shared bookkeeping */
    function newSession(kind) {
        return { kind, cancelled: false, finishing: false, t0: Date.now(), speechAt: 0, timers: [], cleanup: [], preview: '', earlyTimer: null };
    }
    function endSessionResources(s) {
        s.timers.forEach(t => { try { clearInterval(t); clearTimeout(t); } catch (_) { /* gone */ } });
        s.timers = [];
        s.cleanup.splice(0).reverse().forEach(fn => { try { fn(); } catch (_) { /* best effort */ } });
    }

    async function start() {
        if (V.state !== 'idle' && V.state !== 'error') return;
        try {
            const ta = $('chat-input-query');
            if (!ta || ta.disabled) { toast('Voice input', 'Select a repository and thread first.', 'warning'); return; }
            stopSpeaking();
            if (!V.cfg || V.cfgRunner !== runnerName()) await loadConfig();
            if (root.isSecureContext === false) {   // http:// on a LAN IP / hostname: browsers hide the mic entirely
                const https = 'https://' + location.host + location.pathname;
                setStatus('Mic needs a secure page. Open ' + https + ' (or http://localhost on this machine). See docs/VOICE_SETUP.md \u2192 HTTPS.', 'error');
                toast('Voice needs HTTPS', 'Browsers only allow the microphone on https:// or localhost. This page is ' + location.origin + '.', 'error');
                return;
            }
            const m = mode();
            if (cfg().enabled === false || m === 'none') { setStatus('Voice unavailable — please type your prompt.', 'warn'); applyAvailability(); return; }
            setState('listening');
            if (m === 'server') await startServer(); else startBrowser();
        } catch (e) {
            warn('start failed', e);
            fail(e);
        }
    }

    function fail(e) {
        const s = V.session;
        V.session = null;
        if (s) { s.cancelled = true; endSessionResources(s); }
        const name = e && e.name;
        if (name === 'NotAllowedError' || name === 'SecurityError' || (e && e.message === 'not-allowed')) {
            setStatus('Microphone blocked. Allow microphone access for this site in the browser, then try again.', 'error');
            toast('Microphone blocked', 'Allow microphone access for this site (lock icon in the address bar).', 'error');
        } else if (name === 'NotFoundError' || (e && e.message === 'audio-capture')) {
            setStatus('No microphone found.', 'error');
        } else {
            setStatus('Voice unavailable — please type your prompt.', 'error');
        }
        setState('error');
        setTimeout(() => { if (V.state === 'error') setState('idle'); }, 2500);
    }

    /* ------------------------------------------------------------------ server mode: record + VAD + wav */
    async function startServer() {
        const s = newSession('server');
        V.session = s;
        const AC = root.AudioContext || root.webkitAudioContext;
        let stream = null, ctx = null;
        try {
            stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 } });
            if (V.session !== s) { stream.getTracks().forEach(t => t.stop()); return; }       // cancelled while the permission prompt was open
            s.cleanup.push(() => stream.getTracks().forEach(t => t.stop()));
            ctx = new AC();
            s.cleanup.push(() => { try { ctx.close(); } catch (_) { /* closed */ } });
            if (ctx.state === 'suspended') await ctx.resume();
            const src = ctx.createMediaStreamSource(stream);
            const analyser = ctx.createAnalyser();
            analyser.fftSize = 2048;
            src.connect(analyser);
            const chunks = [];
            let frames = 0;
            const sink = ctx.createGain();           // keeps the capture node pulled without playing the microphone
            sink.gain.value = 0;
            sink.connect(ctx.destination);
            const onChunk = (f32) => { chunks.push(f32); frames += f32.length; };
            let node = null;
            try {
                if (!ctx.audioWorklet) throw new Error('no AudioWorklet');
                const code = 'class P extends AudioWorkletProcessor{constructor(){super();this.b=[];this.n=0}process(i){const c=i[0]&&i[0][0];if(c){this.b.push(new Float32Array(c));this.n+=c.length;if(this.n>=4096){const o=new Float32Array(this.n);let k=0;for(const x of this.b){o.set(x,k);k+=x.length}this.port.postMessage(o,[o.buffer]);this.b=[];this.n=0}}return true}}registerProcessor("voice-capture",P);';
                const url = URL.createObjectURL(new Blob([code], { type: 'application/javascript' }));
                try { await ctx.audioWorklet.addModule(url); } finally { URL.revokeObjectURL(url); }
                node = new AudioWorkletNode(ctx, 'voice-capture');
                node.port.onmessage = ev => onChunk(ev.data);
            } catch (_) {                              // CSP / old browser: ScriptProcessor still works everywhere
                node = ctx.createScriptProcessor(4096, 1, 1);
                node.onaudioprocess = ev => onChunk(new Float32Array(ev.inputBuffer.getChannelData(0)));
            }
            src.connect(node);
            node.connect(sink);
            s.cleanup.push(() => { try { node.disconnect(); src.disconnect(); sink.disconnect(); } catch (_) { /* gone */ } });

            const sr = ctx.sampleRate, pauseMs = Math.max(500, +cfg().pause_ms || 2500), maxMs = Math.max(5, +cfg().max_seconds || 120) * 1000;
            const buf = new Float32Array(analyser.fftSize);
            let floor = 0, calN = 0, calSum = 0, loud = 0, lastVoice = 0, lastVoiceFrames = 0;
            setStatus(`Listening… say "${(phrases()[0] || 'go ahead now')}" to send`, 'info');
            startPreviewRecognition(s);
            const tick = setInterval(() => {
                try {
                    if (V.session !== s || s.finishing) return;
                    analyser.getFloatTimeDomainData(buf);
                    const r = rms(buf), now = Date.now(), el = now - s.t0;
                    level(r);
                    if (el < CALIBRATE_MS) { calSum += r; calN++; return; }
                    if (!floor) floor = Math.min(0.02, Math.max(0.002, calN ? calSum / calN : 0.004));
                    const thr = Math.max(0.012, floor * 2.8);
                    if (r > thr) { loud++; if (loud >= 2 && !s.speechAt) s.speechAt = now; if (s.speechAt) { lastVoice = now; lastVoiceFrames = frames; } }
                    else {
                        loud = 0;
                        if (r > thr * 0.7 && s.speechAt) { lastVoice = now; lastVoiceFrames = frames; }       // hysteresis: trailing words do not end the utterance
                        else if (!s.speechAt) floor = floor * 0.95 + r * 0.05;                                // adapt to a changing room before speech starts
                    }
                    if (s.speechAt && now - lastVoice >= pauseMs) return void finishServer(s, chunks, sr, lastVoiceFrames);
                    if (el >= maxMs) return void finishServer(s, chunks, sr, s.speechAt ? frames : 0);
                    if (!s.speechAt && el >= NO_SPEECH_MS) { cancel('No speech detected.'); }
                } catch (e) { warn('vad tick', e); }
            }, 50);
            s.timers.push(tick);
            s.stop = () => finishServer(s, chunks, sr, s.speechAt ? Math.max(lastVoiceFrames, frames - Math.round(sr * 0.2)) : 0);
        } catch (e) {
            if (V.session === s) V.session = null;
            endSessionResources(s);
            throw e;
        }
    }

    async function finishServer(s, chunks, sr, keepFrames) {
        if (s.finishing || s.cancelled) return;
        s.finishing = true;
        stopPreviewRecognition(s);
        endSessionResources(s);                       // mic released before the upload
        V.session = null;
        try {
            const want = Math.min(keepFrames + Math.round(sr * 0.6), Infinity);       // a little tail after the last voiced frame
            let total = 0;
            for (const c of chunks) total += c.length;
            if (!s.speechAt || total < sr * 0.3) { setState('idle'); setStatus(!s.speechAt ? 'No speech detected.' : "Didn't catch that. Click the mic and try again.", 'warn'); setTimeout(clearInfoStatus, 2500); return; }
            const pcm = new Float32Array(Math.min(total, want));
            let k = 0;
            for (const c of chunks) { if (k >= pcm.length) break; const part = c.length > pcm.length - k ? c.subarray(0, pcm.length - k) : c; pcm.set(part, k); k += part.length; }
            const wav = encodeWav(pcm, sr, 16000);
            setState('processing');
            setStatus(`Transcribing via ${(cfg().providers || [])[0] || 'server'}…`, 'info');
            await uploadAndDeliver(wav);
        } catch (e) {
            warn('finishServer', e);
            fail(e);
        }
    }

    async function uploadAndDeliver(wav) {
        const fd = new FormData();
        fd.append('audio', new Blob([wav], { type: 'audio/wav' }), 'speech.wav');
        fd.append('runner', runnerName());
        fd.append('lang', lang());
        const ac = new AbortController(), timer = setTimeout(() => ac.abort(), FETCH_TIMEOUT_MS);
        let data = null, httpStatus = 0;
        try {
            const res = await fetch('/api/v1/voice/transcribe', { method: 'POST', body: fd, signal: ac.signal });
            httpStatus = res.status;
            if (res.ok) data = await res.json();
        } catch (e) { warn('transcribe request failed', e); }
        finally { clearTimeout(timer); }
        if (data && data.ok) {
            const clientTrig = splitTrigger(data.text || '', phrases());          // belt and braces: the server already strips, this catches leftovers
            deliver(clientTrig.text, !!(data.triggered || clientTrig.triggered));
            setState('idle');
            return;
        }
        setState('idle');
        if (httpStatus === 413) { setStatus('Recording too long for the server. Say less at once.', 'error'); return; }
        // server could not transcribe (all providers failed, endpoint down, timeout): browser speech from now on in this session
        V.sessionBrowser = !!SR;
        applyAvailability();
        if (SR) toast('Voice', 'Server transcription failed — switching to browser speech. Click the mic and speak again.', 'warning', 6000);
        else toast('Voice', 'Voice unavailable — please type your prompt.', 'error');
        setStatus(SR ? 'Server transcription failed — switching to browser speech. Click the mic and speak again.' : 'Voice unavailable — please type your prompt.', 'warn');
    }

    /* ------------------------------------------------------------------ live preview + early trigger (server mode) */
    function startPreviewRecognition(s) {
        if (!SR) return;
        try {
            const rec = new SR();
            rec.continuous = true; rec.interimResults = true; rec.lang = srLang();
            s.rec = rec; s.recWanted = true;
            rec.onresult = ev => {
                try {
                    let t = '';
                    for (let i = 0; i < ev.results.length; i++) t += ev.results[i][0].transcript + ' ';
                    s.preview = t.trim();
                    if (s.preview) setStatus('“' + s.preview.slice(-140) + '”', 'info');
                    armEarlyTrigger(s, () => finishNowServer(s));
                } catch (_) { /* preview only */ }
            };
            rec.onerror = () => { s.recWanted = false; };
            rec.onend = () => { try { if (s.recWanted && V.session === s && !s.finishing) rec.start(); } catch (_) { s.recWanted = false; } };
            rec.start();
            s.cleanup.push(() => stopPreviewRecognition(s));
        } catch (e) { warn('preview recognition unavailable', e); }
    }
    function stopPreviewRecognition(s) {
        s.recWanted = false;
        if (s.rec) { try { s.rec.onend = null; s.rec.abort(); } catch (_) { /* not running */ } s.rec = null; }
    }
    /** A trigger phrase at the end of live text: stop after a short grace so "go ahead and ..." (more words coming) is not cut off. */
    function armEarlyTrigger(s, stopFn) {
        if (s.earlyTimer) { clearTimeout(s.earlyTimer); s.earlyTimer = null; }
        if (!splitTrigger(s.preview, phrases()).triggered) return;
        s.earlyTimer = setTimeout(() => { try { if (V.session === s && !s.finishing) { s.earlyTrigger = true; stopFn(); } } catch (e) { warn('early stop', e); } }, 900);
        s.timers.push(s.earlyTimer);
    }
    function finishNowServer(s) { if (s.stop) s.stop(); }

    /* ------------------------------------------------------------------ browser mode: Web Speech API */
    function startBrowser() {
        const s = newSession('browser');
        V.session = s;
        const pauseMs = Math.max(500, +cfg().pause_ms || 2500), maxMs = Math.max(5, +cfg().max_seconds || 120) * 1000;
        const rec = new SR();
        rec.continuous = true; rec.interimResults = true; rec.lang = srLang();
        s.rec = rec; s.recWanted = true; s.final = ''; s.lastActivity = Date.now(); s.restarts = 0;
        rec.onresult = ev => {
            try {
                let fin = '', tmp = '';
                for (let i = 0; i < ev.results.length; i++) { const r = ev.results[i]; if (r.isFinal) fin += r[0].transcript + ' '; else tmp += r[0].transcript + ' '; }
                s.final = fin.trim(); s.preview = (fin + tmp).trim();
                s.lastActivity = Date.now(); if (!s.speechAt) s.speechAt = s.lastActivity;
                if (s.preview) setStatus('“' + s.preview.slice(-140) + '”', 'info');
                armEarlyTrigger(s, () => finishBrowser(s));
            } catch (e) { warn('onresult', e); }
        };
        rec.onerror = ev => {
            const code = ev && ev.error;
            if (code === 'no-speech' || code === 'aborted') return;                // 'end' restarts it
            s.recWanted = false;
            if (code === 'not-allowed' || code === 'service-not-allowed') return fail(Object.assign(new Error('not-allowed'), { name: 'NotAllowedError' }));
            if (code === 'audio-capture') return fail(Object.assign(new Error('audio-capture'), { name: 'NotFoundError' }));
            if (code === 'network') { toast('Voice', 'Browser speech recognition needs an internet connection.', 'error'); return fail(new Error('network')); }
            warn('recognition error', code);
        };
        rec.onend = () => {                                                         // Chrome ends the stream after a few quiet seconds: keep listening
            try {
                if (!(s.recWanted && V.session === s && !s.finishing)) return;
                if (++s.restarts > 200) return finishBrowser(s);
                rec.start();
            } catch (_) { if (V.session === s) finishBrowser(s); }
        };
        try { rec.start(); } catch (e) { V.session = null; throw e; }
        s.cleanup.push(() => stopPreviewRecognition(s));
        s.stop = () => finishBrowser(s);
        setStatus(`Listening (browser speech)… say "${(phrases()[0] || 'go ahead now')}" to send`, 'info');
        const tick = setInterval(() => {
            try {
                if (V.session !== s || s.finishing) return;
                const now = Date.now();
                level(0.15 + 0.1 * Math.sin(now / 150));
                if (s.speechAt && now - s.lastActivity >= pauseMs) finishBrowser(s);
                else if (now - s.t0 >= maxMs) finishBrowser(s);
                else if (!s.speechAt && now - s.t0 >= NO_SPEECH_MS) cancel('No speech detected.');
            } catch (e) { warn('browser tick', e); }
        }, 200);
        s.timers.push(tick);
    }

    async function finishBrowser(s) {
        if (s.finishing || s.cancelled) return;
        s.finishing = true;
        const raw = (s.preview || '').trim();
        const wasSpeech = !!s.speechAt;
        endSessionResources(s);
        V.session = null;
        try {
            if (!wasSpeech || !raw) { setState('idle'); setStatus('No speech detected.', 'warn'); setTimeout(clearInfoStatus, 2500); return; }
            setState('processing');
            setStatus('Tidying up…', 'info');
            let out = null;
            try {
                const ac = new AbortController(), timer = setTimeout(() => ac.abort(), FETCH_TIMEOUT_MS);
                try {
                    const res = await fetch('/api/v1/voice/refine', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text: raw, runner: runnerName() }), signal: ac.signal });
                    if (res.ok) { const j = await res.json(); if (j && j.ok) out = j; }
                } finally { clearTimeout(timer); }
            } catch (e) { warn('refine failed, using the raw text', e); }
            if (out) { const c = splitTrigger(out.text || '', phrases()); deliver(c.text, !!(out.triggered || c.triggered)); }
            else { const c = splitTrigger(raw, phrases()); deliver(c.text, c.triggered); }
        } catch (e) { warn('finishBrowser', e); }
        finally { setState('idle'); }
    }

    /* ------------------------------------------------------------------ stop / cancel */
    function stopAndProcess() {
        const s = V.session;
        if (s && s.stop && !s.finishing) { try { s.stop(); } catch (e) { warn('stop', e); cancel(); } }
    }
    function cancel(message) {
        const s = V.session;
        V.session = null;
        if (s) { s.cancelled = true; stopPreviewRecognition(s); endSessionResources(s); }
        if (V.state === 'listening' || V.state === 'error') setState('idle');
        setStatus(message || '', message ? 'warn' : 'info');
        if (message) setTimeout(clearInfoStatus, 2500);
    }
    function clearInfoStatus() { const el = $('voice-status'); if (el && V.state === 'idle' && el.dataset.kind !== 'error') setStatus('', 'info'); }

    function toggle() {
        try {
            if (V.state === 'listening') stopAndProcess();
            else if (V.state === 'idle' || V.state === 'error') start();
        } catch (e) { warn('toggle', e); }
    }

    /* ------------------------------------------------------------------ read aloud */
    function stopSpeaking() {
        try { if (root.speechSynthesis) root.speechSynthesis.cancel(); } catch (_) { /* none */ }
        if (V.audio) { try { V.audio.pause(); URL.revokeObjectURL(V.audio.src); } catch (_) { /* gone */ } V.audio = null; }
        V.speakToken = (V.speakToken || 0) + 1;
    }
    function browserSpeak(text) {
        if (!root.speechSynthesis || typeof SpeechSynthesisUtterance === 'undefined') return;
        const u = new SpeechSynthesisUtterance(text);
        u.lang = 'en-IN';
        root.speechSynthesis.speak(u);
    }
    async function onAssistantReply(answer) {
        try {
            if (!ls.get('voiceTts', false)) return;
            const text = speakable(answer);
            if (!text) return;
            stopSpeaking();
            const token = V.speakToken;
            if (cfg().tts && cfg().tts.enabled && cfg().tts.provider === 'ollama') {
                try {
                    const res = await fetch('/api/v1/voice/speak', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text, runner: runnerName() }) });
                    if (res.ok) {
                        const blob = await res.blob();
                        if (token !== V.speakToken) return;               // the user started recording / sent a message meanwhile
                        const a = new Audio(URL.createObjectURL(blob));
                        V.audio = a;
                        a.onended = () => { try { URL.revokeObjectURL(a.src); } catch (_) { /* gone */ } if (V.audio === a) V.audio = null; };
                        await a.play();
                        return;
                    }
                } catch (e) { warn('server read-aloud failed, using the browser voice', e); }
                if (token !== V.speakToken) return;
            }
            browserSpeak(text);
        } catch (e) { warn('onAssistantReply', e); }
    }

    /* ------------------------------------------------------------------ wiring */
    function init() {
        try {
            const mic = $('mic-btn');
            if (mic) mic.addEventListener('click', e => { e.preventDefault(); toggle(); });
            document.addEventListener('keydown', e => {
                try {
                    if ((e.ctrlKey || e.metaKey) && e.shiftKey && (e.code === 'KeyM' || (e.key || '').toLowerCase() === 'm')) { e.preventDefault(); toggle(); }
                    else if (e.key === 'Escape' && V.state === 'listening') { e.preventDefault(); cancel('Voice input cancelled.'); }
                } catch (err) { warn('keydown', err); }
            });
            const ta = $('chat-input-query');
            if (ta) ta.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) stopSpeaking(); });      // Enter sends without a form submit event
            const form = $('chat-form');
            if (form) form.addEventListener('submit', () => { stopSpeaking(); if (V.state === 'listening') cancel(); });
            const sel = $('voice-lang');
            if (sel) { sel.value = lang(); sel.addEventListener('change', () => ls.set('voiceLang', sel.value)); }
            const tts = $('voice-tts');
            if (tts) { tts.checked = !!ls.get('voiceTts', false); tts.addEventListener('change', () => { ls.set('voiceTts', tts.checked); if (!tts.checked) stopSpeaking(); }); }
            const rs = $('runner-select');
            if (rs) rs.addEventListener('change', () => setTimeout(() => { loadConfig().catch(e => warn('config', e)); }, 0));
            applyAvailability();
            loadConfig().catch(e => warn('config', e));
        } catch (e) { warn('init', e); }
    }
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();

    return Object.assign({ onAssistantReply, toggle, cancel, start, stopSpeaking, state: () => V.state, reloadConfig: loadConfig }, pure);
});
