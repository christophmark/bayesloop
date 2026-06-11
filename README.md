# bayesloop.com

This branch contains the source of [bayesloop.com](http://bayesloop.com), served via GitHub Pages from the `gh-pages` branch.

The site is fully static — no build step. The interactive playground runs bayesloop directly in the browser via [Pyodide](https://pyodide.org) (Python compiled to WebAssembly) and installs the bayesloop wheel from `assets/`.

## Local preview

```
python3 -m http.server 8000
```

Then open [http://localhost:8000](http://localhost:8000). A plain `file://` open won't work — Pyodide and the wheel must be fetched over HTTP.

## Updating the bayesloop wheel

When a new version is released, rebuild the wheel from `master` and replace the one in `assets/`:

```
git checkout master && uv build
cp dist/bayesloop-*.whl <this-branch>/assets/
```

Then update `WHEEL_URL` in `js/playground.js` to match the new filename.

## Deploying

Merge this branch into `gh-pages` — GitHub Pages serves that branch at bayesloop.com (custom domain via `CNAME`).
