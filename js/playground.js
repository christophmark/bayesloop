'use strict';

/* ============================================================
   bayesloop.com — in-browser playground (Pyodide)
   ============================================================ */

const PYODIDE_INDEX_URL = 'https://cdn.jsdelivr.net/pyodide/v0.28.2/full/';
const WHEEL_URL = new URL('assets/bayesloop-2.0.0.dev0-py3-none-any.whl', window.location.href).href;

const DEFAULT_CODE = `import bayesloop as bl
import matplotlib.pyplot as plt

S = bl.Study()
S.load_example_data()          # UK coal mining disasters, 1852-1961

# The yearly number of disasters follows a Poisson distribution, with a
# rate that may drift gradually from one year to the next:
L = bl.om.Poisson('rate', bl.oint(0, 6, 180))
T = bl.tm.GaussianRandomWalk('sigma', 0.25, target='rate')

S.set(L, T)
S.fit()

# Plot the data together with the inferred disaster rate:
plt.figure(figsize=(9, 4))
plt.bar(S.raw_timestamps, S.raw_data, align='center',
        facecolor='#e0605e', alpha=.6, label='disasters per year')
S.plot('rate', color='#287fb9', label='inferred rate')
plt.xlim([1851, 1962])
plt.xlabel('year')
plt.legend()

# Things to try:
#   * loosen or stiffen the dynamics: sigma = 0.05 ... 0.5
#   * an abrupt change instead of a gradual one:
#       T = bl.tm.ChangePoint('t_change', 1890)
#   * sudden parameter jumps at random times:
#       T = bl.tm.RegimeSwitch('log10p_min', -7)
`;

const CAPTURE_FIGURES = `
import io as _io, base64 as _b64, json as _json
import matplotlib.pyplot as _plt
_figs = []
for _n in _plt.get_fignums():
    _buf = _io.BytesIO()
    _plt.figure(_n).savefig(_buf, format="png", dpi=140, bbox_inches="tight")
    _figs.append(_b64.b64encode(_buf.getvalue()).decode())
_plt.close("all")
_json.dumps(_figs)
`;

const runBtn = document.getElementById('run-btn');
const resetBtn = document.getElementById('reset-btn');
const statusEl = document.getElementById('py-status');
const consoleEl = document.getElementById('console');
const figuresEl = document.getElementById('figures');

/* ----------------------------- editor ----------------------------- */

const editor = CodeMirror.fromTextArea(document.getElementById('code'), {
  mode: 'python',
  theme: 'material-darker',
  lineNumbers: true,
  indentUnit: 4,
  viewportMargin: Infinity,
  extraKeys: {
    'Ctrl-Enter': run,
    'Cmd-Enter': run,
  },
});
editor.setValue(DEFAULT_CODE);

resetBtn.addEventListener('click', () => {
  editor.setValue(DEFAULT_CODE);
  editor.focus();
});

/* ----------------------------- status & console ----------------------------- */

function setStatus(text, busy) {
  statusEl.textContent = text;
  statusEl.classList.toggle('busy', Boolean(busy));
}

let progressEl = null;

function clearConsole() {
  consoleEl.textContent = '';
  progressEl = null;
}

function appendConsole(text, className) {
  const line = document.createElement('span');
  if (className) line.className = className;
  line.textContent = text + '\n';
  consoleEl.appendChild(line);
  consoleEl.scrollTop = consoleEl.scrollHeight;
}

// tqdm redraws its progress bar with carriage returns; render it as a
// single line that updates in place instead of hundreds of lines.
function handleStderr(text) {
  const line = text.split('\r').filter(Boolean).pop();
  if (!line) return;
  const isProgress = /\d+%\|/.test(line) || /it\/s\]?\s*$/.test(line);
  if (isProgress) {
    if (!progressEl || progressEl !== consoleEl.lastChild) {
      progressEl = document.createElement('span');
      progressEl.className = 'progress';
      consoleEl.appendChild(progressEl);
    }
    progressEl.textContent = line + '\n';
  } else {
    appendConsole(line, 'progress');
  }
  consoleEl.scrollTop = consoleEl.scrollHeight;
}

/* ----------------------------- figures ----------------------------- */

function renderFigures(base64List) {
  if (!base64List.length) {
    appendConsole('(no figure produced — call plt.figure()/plt.plot() to draw one)', 'progress');
    return;
  }
  figuresEl.textContent = '';
  for (const b64 of base64List) {
    const img = document.createElement('img');
    img.src = 'data:image/png;base64,' + b64;
    img.alt = 'bayesloop analysis result';
    figuresEl.appendChild(img);
  }
}

/* ----------------------------- pyodide ----------------------------- */

let pyodidePromise = null;

function ensurePyodide() {
  if (!pyodidePromise) {
    pyodidePromise = initPyodide().catch((err) => {
      pyodidePromise = null; // allow retrying after e.g. a network hiccup
      throw err;
    });
  }
  return pyodidePromise;
}

async function initPyodide() {
  setStatus('Downloading Python runtime — only needed once…', true);
  const pyodide = await loadPyodide({ indexURL: PYODIDE_INDEX_URL });

  setStatus('Loading NumPy, SciPy, SymPy & matplotlib…', true);
  await pyodide.loadPackage(['numpy', 'scipy', 'sympy', 'matplotlib', 'micropip']);

  setStatus('Installing bayesloop…', true);
  await pyodide.runPythonAsync(`
import matplotlib
matplotlib.use("Agg")
import micropip
await micropip.install("${WHEEL_URL}")
import warnings
from tqdm import TqdmMonitorWarning
warnings.filterwarnings("ignore", category=TqdmMonitorWarning)  # WebAssembly has no threads
import bayesloop
`);
  return pyodide;
}

/* ----------------------------- run ----------------------------- */

let running = false;

async function run() {
  if (running) return;
  running = true;
  runBtn.disabled = true;
  clearConsole();

  const t0 = performance.now();
  try {
    const pyodide = await ensurePyodide();
    pyodide.setStdout({ batched: (s) => appendConsole(s) });
    pyodide.setStderr({ batched: (s) => handleStderr(s) });

    setStatus('Running analysis…', true);
    await pyodide.runPythonAsync(editor.getValue());

    const figures = await pyodide.runPythonAsync(CAPTURE_FIGURES);
    renderFigures(JSON.parse(figures));

    const seconds = ((performance.now() - t0) / 1000).toFixed(1);
    setStatus(`Done in ${seconds} s`, false);
  } catch (err) {
    appendConsole(String((err && err.message) || err), 'err');
    setStatus('Error — see console output.', false);
  } finally {
    running = false;
    runBtn.disabled = false;
  }
}

runBtn.addEventListener('click', run);

/* ----------------------------- page chrome ----------------------------- */

document.querySelectorAll('[data-copy]').forEach((btn) => {
  btn.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(btn.dataset.copy);
      btn.classList.add('copied');
      setTimeout(() => btn.classList.remove('copied'), 1200);
    } catch (e) {
      /* clipboard unavailable (e.g. http) — ignore */
    }
  });
});

const yearEl = document.getElementById('year');
if (yearEl) yearEl.textContent = String(new Date().getFullYear());
