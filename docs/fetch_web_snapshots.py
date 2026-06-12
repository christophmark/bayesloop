#!/usr/bin/env python3
"""Snapshot the external datasets used by the case-study notebooks.

The live-code runner (docs/source/_static/live-code-worker.js) patches
``urllib.request.urlopen`` in the in-browser Python session so that the
notebooks' download URLs are served from these snapshots — third-party
servers cannot be reached from a browser (CORS). Each page directory under
``docs/source/examples/data/web/<page>/`` holds gzipped payloads plus a
``manifest.json`` mapping the original URLs to the payload files.

Re-run this script to refresh the snapshots, then re-execute the affected
notebooks so static outputs and live results stay in sync.
"""

import gzip
import json
import time
import urllib.request
from pathlib import Path

UA = {"User-Agent": "Mozilla/5.0 (bayesloop docs)"}
OUT_ROOT = Path(__file__).parent / "source" / "examples" / "data" / "web"


def usgs_decade_urls():
    """Replicate the URL construction in earthquakes.ipynb exactly."""
    urls, start = [], 1900
    while start <= 2025:
        end = min(start + 9, 2025)
        urls.append(
            "https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv"
            f"&starttime={start}-01-01&endtime={end}-12-31"
            "&minmagnitude=7.0&orderby=time-asc"
        )
        start = end + 1
    return urls


# page -> list of snapshot groups; a group is a list of alternative URLs the
# notebook tries in order (all alternatives map to the same payload file).
PAGES = {
    "baseball": [
        [
            "https://raw.githubusercontent.com/orrski/baseballdatabank/master/core/Teams.csv",
            "https://raw.githubusercontent.com/cbwinslow/baseballdatabank/master/core/Teams.csv",
        ],
    ],
    "covidforecasting": [
        [
            "https://raw.githubusercontent.com/owid/covid-19-data/master/"
            "public/data/cases_deaths/full_data.csv",
        ],
    ],
    "earthquakes": [[u] for u in usgs_decade_urls()],
    "greatmoderation": [
        # FRED is flaky; the notebook falls back to the GitHub mirror on its
        # own, so a missing FRED snapshot only changes which code path runs.
        {"urls": ["https://fred.stlouisfed.org/graph/fredgraph.csv?id=GDPC1"],
         "optional": True},
        ["https://raw.githubusercontent.com/datasets/gdp-us/main/data/quarter.csv"],
    ],
    "hurricanes": [
        [
            "https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2023-051124.txt",
            "https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2024-040425.txt",
            "https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2022-050423.txt",
        ],
    ],
    "measles": [
        ["https://ourworldindata.org/grapher/number-of-measles-cases.csv?csvType=full"],
    ],
    "sunspots": [
        [
            "https://www.sidc.be/SILSO/INFO/snytotcsv.php",
            "https://www.sidc.be/silso/INFO/snytotcsv.php",
        ],
    ],
}


def fetch_first(urls, attempts=2):
    last_err = None
    for url in urls:
        for _ in range(attempts):
            try:
                req = urllib.request.Request(url, headers=UA)
                data = urllib.request.urlopen(req, timeout=120).read()
                return url, data
            except Exception as err:  # noqa: BLE001 - retry, then next mirror
                print(f"    ! {url}: {err}")
                last_err = err
                time.sleep(2.0)
    raise RuntimeError(f"all alternatives failed: {urls}") from last_err


def main():
    for page, groups in PAGES.items():
        out_dir = OUT_ROOT / page
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest = []
        print(f"== {page}")
        for i, group in enumerate(groups):
            urls = group["urls"] if isinstance(group, dict) else group
            optional = isinstance(group, dict) and group.get("optional")
            try:
                fetched_url, data = fetch_first(urls)
            except RuntimeError:
                if optional:
                    print("    (optional, skipped)")
                    continue
                raise
            name = f"{i:02d}.gz"
            payload = gzip.compress(data, 9)
            (out_dir / name).write_bytes(payload)
            for url in urls:
                manifest.append({"url": url, "file": name, "gzip": True})
            print(
                f"    {name}: {len(data) / 1024:.0f} KB raw -> "
                f"{len(payload) / 1024:.0f} KB gz   ({fetched_url})"
            )
            time.sleep(1.0)
        (out_dir / "manifest.json").write_text(
            json.dumps({"snapshots": manifest}, indent=1) + "\n"
        )
    total = sum(f.stat().st_size for f in OUT_ROOT.rglob("*.gz"))
    print(f"\ntotal snapshot size: {total / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
