/*
 * live-code: run the notebook code cells of selected documentation pages
 * directly in the browser via Pyodide (CPython compiled to WebAssembly).
 *
 * Nothing is downloaded until the user clicks a run button. The runtime then
 * boots in a Web Worker (see live-code-worker.js), installs bayesloop from
 * PyPI and executes cells in a shared namespace, notebook-style: clicking a
 * cell first runs all not-yet-executed cells above it. Once the session is
 * ready, cells become editable in place (Shift+Enter re-runs).
 *
 * To enable a new page, add it to LIVE_PAGES below. Notebooks that download
 * data from third-party servers need their payloads snapshotted into
 * examples/data/web/ first (run docs/fetch_web_snapshots.py) — browsers
 * cannot reach most external servers (CORS); the session's urllib is patched
 * to serve those URLs from the snapshots, fetched lazily per request.
 * Packages without pure-Python wheels (hmmlearn, pyreadr, arch) are
 * unavailable in Pyodide, which is why animalmovement and marketvolatility
 * stay static.
 */
(function () {
    'use strict';

    if (typeof window === 'undefined' || !window.Worker || !window.WebAssembly) return;

    var PYODIDE_INDEX_URL = 'https://cdn.jsdelivr.net/pyodide/v0.28.3/full/';

    // `packages`: extra pure-Python PyPI packages installed at boot.
    // `data`: files fetched from the docs site into the in-browser filesystem
    //         (path is relative to the page URL and used verbatim as file path).
    var LIVE_PAGES = {
        'tutorials/firststeps': {},
        'tutorials/modelselection': {},
        'tutorials/priordistributions': {},
        'tutorials/hyperstudy': {},
        'tutorials/changepointstudy': {},
        'tutorials/onlinestudy': {},
        'tutorials/hyperparameteroptimization': {},
        'tutorials/customobservationmodels': {},
        'examples/anomalousdiffusion': {},
        'examples/stockmarketfluctuations': {},
        'examples/seizuredetection': {
            packages: ['seaborn'],
            data: ['data/eeg/bonn_FS.npz']
        },
        'examples/energydemand': {
            packages: ['seaborn'],
            data: ['data/energy/de_load_daily.csv', 'data/energy/berlin_temp_daily.csv']
        },
        'examples/baseball': { packages: ['seaborn'], web: 'data/web/baseball/manifest.json' },
        'examples/covidforecasting': { packages: ['seaborn'], web: 'data/web/covidforecasting/manifest.json' },
        'examples/earthquakes': { packages: ['seaborn'], web: 'data/web/earthquakes/manifest.json' },
        'examples/greatmoderation': { packages: ['seaborn'], web: 'data/web/greatmoderation/manifest.json' },
        'examples/hurricanes': { packages: ['seaborn'], web: 'data/web/hurricanes/manifest.json' },
        'examples/measles': { packages: ['seaborn'], web: 'data/web/measles/manifest.json' },
        'examples/sunspots': { packages: ['seaborn'], web: 'data/web/sunspots/manifest.json' }
    };

    var PLAY_SVG = '<svg width="11" height="11" viewBox="0 0 16 16" aria-hidden="true">' +
        '<path d="M3 1.5v13l11-6.5z" fill="currentColor"/></svg>';

    var SCRIPT_SRC = (function () {
        if (document.currentScript && document.currentScript.src) return document.currentScript.src;
        var scripts = document.getElementsByTagName('script');
        for (var i = 0; i < scripts.length; i++) {
            if (scripts[i].src && scripts[i].src.indexOf('live-code.js') !== -1) return scripts[i].src;
        }
        return '';
    })();

    var cfg = (function () {
        var m = window.location.pathname.match(/([\w-]+)\/([\w-]+?)(?:\.html?)?$/);
        return m ? LIVE_PAGES[m[1] + '/' + m[2]] || null : null;
    })();
    if (!cfg || !SCRIPT_SRC) return;

    var cells = [];
    var session = {
        state: 'idle',   // idle | booting | ready | error
        worker: null,
        busy: false,
        queue: [],
        current: null,
        execCount: 1,
        editable: false
    };
    var bootPromise = null;
    var statusEl = null, statusText = null;

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

    function init() {
        var inputs = document.querySelectorAll('div.nbinput');
        Array.prototype.forEach.call(inputs, function (inputDiv) {
            var area = inputDiv.querySelector('div.input_area');
            var pre = area && area.querySelector('pre');
            if (!pre) return;
            var promptPre = inputDiv.querySelector('div.prompt pre');
            var cell = {
                index: cells.length,
                inputDiv: inputDiv,
                area: area,
                pre: pre,
                promptPre: promptPre,
                originalPrompt: promptPre ? promptPre.textContent : null,
                wasNbLast: inputDiv.classList.contains('nblast'),
                outputDivs: collectOutputs(inputDiv),
                liveOutput: null,
                outArea: null,
                outPre: null,
                outLines: [],
                outCurrent: '',
                renderQueued: false,
                executed: false,
                btn: null
            };
            var btn = document.createElement('button');
            btn.className = 'live-run-btn';
            btn.type = 'button';
            btn.title = 'Run this cell in your browser\n(first use downloads the Python runtime, ~60 MB)';
            btn.setAttribute('aria-label', 'Run cell');
            btn.innerHTML = PLAY_SVG;
            btn.addEventListener('click', function () { requestRun(cell); });
            // attach to the (non-scrolling) cell row, not the scrollable input area
            inputDiv.appendChild(btn);
            cell.btn = btn;

            pre.addEventListener('input', function () { cell.executed = false; });
            pre.addEventListener('keydown', function (ev) {
                if (ev.key === 'Enter' && ev.shiftKey) {
                    ev.preventDefault();
                    requestRun(cell);
                }
            });
            cells.push(cell);
        });
        if (cells.length) document.body.classList.add('live-page');
    }

    function collectOutputs(inputDiv) {
        var outs = [];
        var el = inputDiv.nextElementSibling;
        while (el && el.classList.contains('nboutput')) {
            outs.push(el);
            el = el.nextElementSibling;
        }
        return outs;
    }

    /* ---------------- session ---------------- */

    function ensureBoot() {
        if (session.state === 'ready') return Promise.resolve();
        if (bootPromise) return bootPromise;
        bootPromise = new Promise(function (resolve, reject) {
            session.state = 'booting';
            session.bootResolve = resolve;
            session.bootReject = reject;
            showStatus('booting', 'Loading Python runtime… (~60 MB, first time only)');
            var worker;
            try {
                worker = new Worker(SCRIPT_SRC.replace(/live-code\.js.*$/, 'live-code-worker.js'));
            } catch (err) {
                reject(err);
                return;
            }
            session.worker = worker;
            worker.onmessage = onWorkerMessage;
            worker.onerror = function (ev) {
                var err = new Error(ev.message || 'Python runtime failed to start');
                if (session.state === 'booting') reject(err);
                fail(err.message);
            };
            worker.postMessage({
                type: 'boot',
                indexURL: PYODIDE_INDEX_URL,
                packages: ['bayesloop'].concat(cfg.packages || []),
                versionHint: (window.DOCUMENTATION_OPTIONS && window.DOCUMENTATION_OPTIONS.VERSION) || '',
                dataFiles: (cfg.data || []).map(function (p) {
                    return { url: new URL(p, window.location.href).href, path: p };
                }),
                webManifest: cfg.web ? new URL(cfg.web, window.location.href).href : null
            });
        });
        return bootPromise;
    }

    function onWorkerMessage(ev) {
        var msg = ev.data;
        switch (msg.type) {
            case 'status':
                if (session.state === 'booting') showStatus('booting', msg.text);
                break;
            case 'ready':
                session.state = 'ready';
                enableEditing();
                if (session.bootResolve) session.bootResolve();
                break;
            case 'stream':
                if (msg.id != null && cells[msg.id]) appendStream(cells[msg.id], msg.text);
                break;
            case 'result':
                finishCell(msg);
                break;
            case 'fatal':
                if (session.state === 'booting' && session.bootReject) session.bootReject(new Error(msg.text));
                fail(msg.text);
                break;
        }
    }

    function fail(text) {
        session.state = 'error';
        session.busy = false;
        session.queue.forEach(function (c) { setBtn(c, 'play'); });
        session.queue = [];
        if (session.current) {
            setBtn(session.current, 'error');
            session.current = null;
        }
        showStatus('error', 'Python session failed: ' + text);
    }

    function restart() {
        if (session.worker) {
            session.worker.terminate();
            session.worker = null;
        }
        bootPromise = null;
        session.state = 'idle';
        session.busy = false;
        session.queue = [];
        session.current = null;
        session.execCount = 1;
        session.editable = false;
        cells.forEach(function (c) {
            c.executed = false;
            if (c.liveOutput && c.liveOutput.parentNode) c.liveOutput.parentNode.removeChild(c.liveOutput);
            c.liveOutput = null;
            c.outArea = null;
            c.outPre = null;
            c.outputDivs.forEach(function (d) { d.classList.remove('live-hidden'); });
            if (c.promptPre && c.originalPrompt != null) c.promptPre.textContent = c.originalPrompt;
            if (c.wasNbLast) c.inputDiv.classList.add('nblast');
            c.pre.removeAttribute('contenteditable');
            setBtn(c, 'play');
        });
        if (statusEl) {
            statusEl.parentNode.removeChild(statusEl);
            statusEl = null;
        }
    }

    /* ---------------- execution ---------------- */

    function requestRun(cell) {
        if (session.busy || session.state === 'booting') return;
        var queue = [];
        cells.forEach(function (c) {
            if (c.index < cell.index && !c.executed) queue.push(c);
        });
        queue.push(cell);
        session.queue = queue;
        session.busy = true;
        queue.forEach(function (c) { setBtn(c, 'queued'); });
        ensureBoot().then(runNext).catch(function (err) {
            fail(err && err.message ? err.message : String(err));
        });
    }

    function runNext() {
        var cell = session.queue.shift();
        if (!cell) {
            session.busy = false;
            session.current = null;
            showStatus('ready', 'Python ready — edit any cell, Shift+Enter runs it');
            return;
        }
        session.current = cell;
        prepareLiveOutput(cell);
        setBtn(cell, 'running');
        if (cell.promptPre) cell.promptPre.textContent = '[*]:';
        showStatus('running', 'Running cell ' + (cell.index + 1) + ' of ' + cells.length + '…');
        var code = (cell.pre.innerText !== undefined ? cell.pre.innerText : cell.pre.textContent)
            .replace(/\u00a0/g, ' ')
            .replace(/\n$/, '');
        session.worker.postMessage({ type: 'run', id: cell.index, code: stripMagics(code) });
    }

    function stripMagics(code) {
        // IPython line/shell magics (e.g. "%matplotlib inline") are not Python
        return code.split('\n').filter(function (line) {
            return !/^\s*[%!]/.test(line);
        }).join('\n');
    }

    function finishCell(msg) {
        var cell = cells[msg.id];
        if (!cell) return;
        flushStream(cell);
        if (msg.repr) appendResultPre(cell, msg.repr, false);
        if (msg.html) {
            var div = document.createElement('div');
            div.className = 'live-html';
            div.innerHTML = msg.html;
            cell.outArea.appendChild(div);
            revealOutput(cell);
        }
        (msg.figs || []).forEach(function (b64) {
            var img = document.createElement('img');
            img.alt = 'live output figure';
            img.src = 'data:image/png;base64,' + b64;
            cell.outArea.appendChild(img);
            revealOutput(cell);
        });
        if (msg.ok) {
            cell.executed = true;
            setBtn(cell, 'done');
            if (cell.promptPre) cell.promptPre.textContent = '[' + session.execCount + ']:';
            session.execCount += 1;
            session.current = null;
            runNext();
        } else {
            appendResultPre(cell, msg.error || 'Execution failed.', true);
            setBtn(cell, 'error');
            if (cell.promptPre) cell.promptPre.textContent = '[!]:';
            session.queue.forEach(function (c) { setBtn(c, 'play'); });
            session.queue = [];
            session.busy = false;
            session.current = null;
            showStatus('ready', 'Cell raised an exception — fix it and run again');
        }
    }

    /* ---------------- live output rendering ---------------- */

    function prepareLiveOutput(cell) {
        if (!cell.liveOutput) {
            var div = document.createElement('div');
            div.className = 'nboutput nblast docutils container live-output';
            div.innerHTML = '<div class="prompt empty docutils container"></div>' +
                '<div class="output_area docutils container"></div>';
            cell.inputDiv.parentNode.insertBefore(div, cell.inputDiv.nextSibling);
            cell.liveOutput = div;
        }
        cell.outArea = cell.liveOutput.querySelector('.output_area');
        cell.outArea.innerHTML = '';
        cell.outPre = null;
        cell.outLines = [];
        cell.outCurrent = '';
        cell.liveOutput.style.display = 'none';
        if (cell.wasNbLast) cell.inputDiv.classList.add('nblast');
        cell.outputDivs.forEach(function (d) { d.classList.add('live-hidden'); });
    }

    function revealOutput(cell) {
        if (cell.liveOutput.style.display === 'none') {
            cell.liveOutput.style.display = '';
            cell.inputDiv.classList.remove('nblast');
        }
    }

    function ensureOutPre(cell) {
        if (!cell.outPre) {
            var hl = document.createElement('div');
            hl.className = 'highlight';
            var pre = document.createElement('pre');
            hl.appendChild(pre);
            cell.outArea.insertBefore(hl, cell.outArea.firstChild);
            cell.outPre = pre;
            revealOutput(cell);
        }
        return cell.outPre;
    }

    function appendResultPre(cell, text, isError) {
        var hl = document.createElement('div');
        hl.className = isError ? 'highlight live-traceback' : 'highlight';
        var pre = document.createElement('pre');
        pre.textContent = '\n' + text + '\n';
        hl.appendChild(pre);
        cell.outArea.appendChild(hl);
        revealOutput(cell);
    }

    function appendStream(cell, text) {
        // line-based buffering with carriage-return handling (tqdm progress bars)
        var parts = String(text).split('\n');
        for (var i = 0; i < parts.length; i++) {
            var seg = parts[i];
            var cr = seg.lastIndexOf('\r');
            if (cr >= 0) cell.outCurrent = seg.slice(cr + 1);
            else cell.outCurrent += seg;
            if (i < parts.length - 1) {
                cell.outLines.push(cell.outCurrent);
                cell.outCurrent = '';
            }
        }
        if (!cell.renderQueued) {
            cell.renderQueued = true;
            window.requestAnimationFrame(function () { renderStream(cell); });
        }
    }

    function renderStream(cell) {
        cell.renderQueued = false;
        var text = cell.outLines.join('\n');
        if (cell.outCurrent) text += (text ? '\n' : '') + cell.outCurrent;
        if (!text) return;
        ensureOutPre(cell).textContent = '\n' + text + '\n';
    }

    function flushStream(cell) {
        renderStream(cell);
    }

    /* ---------------- editing ---------------- */

    function enableEditing() {
        if (session.editable) return;
        session.editable = true;
        var probe = document.createElement('div');
        probe.setAttribute('contenteditable', 'plaintext-only');
        var mode = probe.contentEditable === 'plaintext-only' ? 'plaintext-only' : 'true';
        cells.forEach(function (c) {
            c.pre.setAttribute('contenteditable', mode);
            c.pre.setAttribute('spellcheck', 'false');
        });
    }

    /* ---------------- UI bits ---------------- */

    function setBtn(cell, state) {
        var btn = cell.btn;
        btn.className = 'live-run-btn live-state-' + state;
        switch (state) {
            case 'running':
            case 'queued':
                btn.innerHTML = '<span class="live-spinner"></span>';
                btn.title = state === 'running' ? 'Running…' : 'Queued…';
                break;
            case 'done':
                btn.innerHTML = '✓';
                btn.title = 'Run again';
                break;
            case 'error':
                btn.innerHTML = '!';
                btn.title = 'Cell raised an exception — run again';
                break;
            default:
                btn.innerHTML = PLAY_SVG;
                btn.title = 'Run this cell in your browser\n(first use downloads the Python runtime, ~60 MB)';
        }
    }

    function showStatus(kind, text) {
        if (!statusEl) {
            statusEl = document.createElement('div');
            statusEl.id = 'live-status';
            statusEl.innerHTML = '<span class="live-dot"></span><span class="live-text"></span>';
            var btn = document.createElement('button');
            btn.className = 'live-restart';
            btn.type = 'button';
            btn.textContent = 'restart';
            btn.title = 'Discard the Python session and restore the original outputs';
            btn.addEventListener('click', restart);
            statusEl.appendChild(btn);
            document.body.appendChild(statusEl);
            statusText = statusEl.querySelector('.live-text');
        }
        statusEl.className = 'live-' + kind;
        statusText.textContent = text;
        statusText.title = text;
    }
})();
