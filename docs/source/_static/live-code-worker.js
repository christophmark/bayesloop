/*
 * live-code-worker: hosts the Pyodide runtime for live-code.js.
 *
 * Runs in a Web Worker so that long fits never block the page. Receives
 * {type:'boot', ...} once and {type:'run', id, code} per cell; emits
 * {type:'status'|'ready'|'stream'|'result'|'fatal'} messages.
 */
'use strict';

var pyodide = null;
var runCell = null;
var currentId = null;

function post(msg) { self.postMessage(msg); }
function status(text) { post({ type: 'status', text: text }); }

/* Synchronous binary-safe fetch for the Python urllib shim (sync XHR is
 * allowed in workers; responseType cannot be set in sync mode, so binary
 * data is smuggled through a user-defined charset). */
self._liveSyncFetch = function (url) {
    try {
        var xhr = new XMLHttpRequest();
        xhr.open('GET', url, false);
        xhr.overrideMimeType('text/plain; charset=x-user-defined');
        xhr.send(null);
        if (xhr.status < 200 || xhr.status >= 300) return { status: xhr.status, data: null };
        var t = xhr.responseText, out = new Uint8Array(t.length);
        for (var i = 0; i < t.length; i++) out[i] = t.charCodeAt(i) & 0xff;
        return { status: xhr.status, data: out };
    } catch (e) {
        return { status: 0, data: null };
    }
};

self.onmessage = function (ev) {
    var msg = ev.data;
    if (msg.type === 'boot') boot(msg);
    else if (msg.type === 'run') run(msg);
};

async function boot(msg) {
    try {
        status('Loading Python runtime… (~60 MB, first time only)');
        importScripts(msg.indexURL + 'pyodide.js');
        pyodide = await self.loadPyodide({ indexURL: msg.indexURL });
        attachStreams();

        status('Installing ' + msg.packages.join(', ') + '…');
        await pyodide.loadPackage('micropip');
        var micropip = pyodide.pyimport('micropip');
        // try to match the documented version, fall back to the latest release
        var pinned = msg.packages.slice();
        if (msg.versionHint) pinned[0] += '==' + msg.versionHint;
        try {
            for (var i = 0; i < pinned.length; i++) await micropip.install(pinned[i]);
        } catch (err) {
            for (var j = 0; j < msg.packages.length; j++) await micropip.install(msg.packages[j]);
        }
        micropip.destroy();

        if (msg.dataFiles && msg.dataFiles.length) {
            status('Fetching example data…');
            for (var k = 0; k < msg.dataFiles.length; k++) {
                var f = msg.dataFiles[k];
                var resp = await fetch(f.url);
                if (!resp.ok) throw new Error('could not fetch ' + f.url + ' (HTTP ' + resp.status + ')');
                var buf = new Uint8Array(await resp.arrayBuffer());
                var dir = f.path.split('/').slice(0, -1).join('/');
                if (dir) pyodide.FS.mkdirTree(dir);
                pyodide.FS.writeFile(f.path, buf);
            }
        }

        status('Preparing session…');
        pyodide.runPython(PRELUDE);
        runCell = pyodide.globals.get('_live_run');

        if (msg.webManifest) {
            // register the page's dataset snapshots with the urllib shim;
            // payloads themselves are fetched lazily, per request
            var mResp = await fetch(msg.webManifest);
            if (!mResp.ok) throw new Error('could not fetch ' + msg.webManifest + ' (HTTP ' + mResp.status + ')');
            var manifest = await mResp.json();
            var mapping = {};
            (manifest.snapshots || []).forEach(function (s) {
                mapping[s.url] = { url: new URL(s.file, msg.webManifest).href, gzip: !!s.gzip };
            });
            var register = pyodide.globals.get('_register_web_snapshots');
            register(JSON.stringify(mapping));
            register.destroy();
        }
        post({ type: 'ready' });
    } catch (err) {
        post({ type: 'fatal', text: String((err && err.message) || err) });
    }
}

function attachStreams() {
    function streamHandler() {
        var decoder = new TextDecoder();
        return {
            write: function (buf) {
                post({ type: 'stream', id: currentId, text: decoder.decode(buf, { stream: true }) });
                return buf.length;
            },
            isatty: true  // tqdm then redraws via \r instead of printing new lines
        };
    }
    try {
        pyodide.setStdout(streamHandler());
        pyodide.setStderr(streamHandler());
    } catch (err) {
        // older Pyodide API fallback: line-buffered, no \r handling
        pyodide.setStdout({ batched: function (s) { post({ type: 'stream', id: currentId, text: s + '\n' }); } });
        pyodide.setStderr({ batched: function (s) { post({ type: 'stream', id: currentId, text: s + '\n' }); } });
    }
}

async function run(msg) {
    if (!pyodide || !runCell) {
        post({ type: 'fatal', text: 'Python runtime is not ready' });
        return;
    }
    currentId = msg.id;
    try {
        // auto-load any Pyodide-bundled packages the cell imports (pandas, statsmodels, …)
        try {
            await pyodide.loadPackagesFromImports(msg.code, {
                messageCallback: function () {},
                errorCallback: function () {}
            });
        } catch (e) { /* best effort */ }
        var result = JSON.parse(runCell(msg.code));
        result.type = 'result';
        result.id = msg.id;
        post(result);
    } catch (err) {
        post({ type: 'result', id: msg.id, ok: false, error: String((err && err.message) || err), figs: [] });
    } finally {
        currentId = null;
    }
}

/* Python session setup and per-cell runner. */
var PRELUDE = [
    "import os",
    "os.environ['MPLBACKEND'] = 'agg'",
    "import warnings",
    "warnings.filterwarnings('ignore', message='.*non-interactive.*')",
    "import matplotlib",
    "matplotlib.use('agg')",
    "# match the inline-backend defaults used to execute the static notebooks",
    "matplotlib.rcParams['figure.figsize'] = (6.0, 4.0)",
    "matplotlib.rcParams['savefig.bbox'] = 'tight'",
    "",
    "# WebAssembly cannot spawn subprocesses: force joblib to run sequentially",
    "import joblib",
    "_orig_parallel_init = joblib.Parallel.__init__",
    "def _sequential_init(self, *args, **kwargs):",
    "    if args:",
    "        args = (1,) + tuple(args[1:])",
    "    else:",
    "        kwargs['n_jobs'] = 1",
    "    _orig_parallel_init(self, *args, **kwargs)",
    "joblib.Parallel.__init__ = _sequential_init",
    "",
    "import ast, base64, io, json, sys, traceback",
    "",
    "# urllib shim: the notebooks' dataset downloads are served from snapshots",
    "# bundled with the docs (fetched lazily, one synchronous request per URL).",
    "# Browsers cannot reach most third-party servers directly (CORS); unknown",
    "# URLs are still attempted and work for hosts that allow cross-origin GETs.",
    "import email.message, gzip, urllib.error, urllib.request",
    "",
    "_web_snapshots = {}",
    "",
    "def _register_web_snapshots(mapping_json):",
    "    _web_snapshots.update(json.loads(mapping_json))",
    "",
    "class _SnapshotResponse(io.BytesIO):",
    "    def __init__(self, data, url, status=200):",
    "        super().__init__(data)",
    "        self.url, self.status, self.code = url, status, status",
    "        self.headers = email.message.Message()",
    "    def geturl(self): return self.url",
    "    def getcode(self): return self.status",
    "    def info(self): return self.headers",
    "",
    "def _browser_get(url):",
    "    import js",
    "    res = js._liveSyncFetch(url)",
    "    if res.data is None:",
    "        return None, int(res.status)",
    "    return bytes(res.data.to_py()), int(res.status)",
    "",
    "def _live_urlopen(url, *args, **kwargs):",
    "    full = getattr(url, 'full_url', url)",
    "    entry = _web_snapshots.get(full)",
    "    if entry is not None:",
    "        data, http_status = _browser_get(entry['url'])",
    "        if data is None:",
    "            raise urllib.error.URLError(",
    "                f'snapshot fetch failed (HTTP {http_status}) for {full}')",
    "        if entry.get('gzip'):",
    "            data = gzip.decompress(data)",
    "        return _SnapshotResponse(data, full)",
    "    data, http_status = _browser_get(full)",
    "    if data is not None:",
    "        return _SnapshotResponse(data, full, http_status)",
    "    raise urllib.error.URLError(",
    "        f'cannot reach {full} from the browser: no bundled snapshot, and direct '",
    "        'requests are blocked unless the server allows cross-origin access')",
    "",
    "urllib.request.urlopen = _live_urlopen",
    "",
    "_live_ns = {'__name__': '__main__'}",
    "",
    "def _live_run(code):",
    "    import matplotlib.pyplot as plt",
    "    out = {'ok': True, 'repr': None, 'html': None, 'figs': [], 'error': None}",
    "    try:",
    "        tree = ast.parse(code)",
    "        trailing = None",
    "        # IPython convention: a trailing semicolon suppresses the result display",
    "        if tree.body and isinstance(tree.body[-1], ast.Expr) and not code.rstrip().endswith(';'):",
    "            trailing = ast.Expression(tree.body[-1].value)",
    "            tree.body = tree.body[:-1]",
    "        exec(compile(tree, '<live-cell>', 'exec'), _live_ns)",
    "        if trailing is not None:",
    "            value = eval(compile(trailing, '<live-cell>', 'eval'), _live_ns)",
    "            if value is not None:",
    "                html = getattr(value, '_repr_html_', None)",
    "                if callable(html):",
    "                    try:",
    "                        out['html'] = html()",
    "                    except Exception:",
    "                        pass",
    "                if out['html'] is None:",
    "                    out['repr'] = repr(value)",
    "    except BaseException:",
    "        out['ok'] = False",
    "        out['error'] = traceback.format_exc(limit=20)",
    "    try:",
    "        for num in plt.get_fignums():",
    "            buf = io.BytesIO()",
    "            plt.figure(num).savefig(buf, format='png')",
    "            out['figs'].append(base64.b64encode(buf.getvalue()).decode('ascii'))",
    "        plt.close('all')",
    "    finally:",
    "        sys.stdout.flush()",
    "        sys.stderr.flush()",
    "    return json.dumps(out)"
].join('\n');
