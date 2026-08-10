#!/usr/bin/env python3
"""Stage 5 corpus puller — bulk, resumable, idempotent document downloader.

RUN THIS ON YOUR OWN MACHINE (normal internet). The Cowork cloud sandbox sits
behind a proxy allowlist that blocks arxiv/sec/wikipedia/archive.org/data.gov,
and the device bridge has no network at all — so this cannot run from Cowork.
It is written to run in your terminal with plain `python corpus_puller.py`.

Design (this IS the Stage-5 "ingestion pipeline is the product" step):
  * ONE manifest (corpus/manifest.json) is the source of truth. Every doc is a
    row keyed by a stable id: {source, id, url, path, format, status, bytes,
    sha256, err}. Re-running skips anything already `done` with a non-empty file
    on disk — so a Ctrl-C or a network drop resumes for free.
  * Per-HOST rate limiting + retry-with-backoff. SEC and arXiv have explicit
    politeness policies; we honor them (and send a real User-Agent with your
    contact, which SEC REQUIRES or it 403s).
  * Each source is an independent fetcher. One source failing never blocks the
    others. `--source all` runs them in sequence; `--source arxiv` runs one.

Six sources (counts in TARGETS below):
  arxiv     arXiv API, mixed CS/physics/econ/math/bio categories -> PDF
  sec       SEC EDGAR 10-K primary documents (HTML) via data.sec.gov submissions
  wikipedia Wikipedia REST render endpoint -> PDF, top members per category
  archive   Internet Archive scanned texts (needs `pip install internetarchive`)
  datagov   data.gov CKAN package_search, resources with format=PDF
  rtd       ReadTheDocs PDF builds (mkdocs/sphinx manuals) — see note below

NOTE on source 6: "GitHub docs repos -> PDF" as literally stated means building
100 mkdocs/sphinx sites (sphinx needs a full LaTeX toolchain) — fragile at
scale. ReadTheDocs already auto-builds a PDF for most such projects and exposes
it at a stable URL, so `rtd` pulls those finished PDFs instead of building
sites. Same documents, none of the build pain. Edit RTD_PROJECTS to taste.

Usage:
    pip install requests internetarchive
    export CORPUS_CONTACT="Deepak Katukuri deepakkatukuri@gmail.com"   # for SEC UA

    python corpus_puller.py --source all
    python corpus_puller.py --source arxiv --limit 50      # smaller trial
    python corpus_puller.py --source sec                   # one source
    python corpus_puller.py --status                       # counts per source
    python corpus_puller.py --source all --dry-run         # plan only, no download
"""

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("Missing dependency: pip install requests")

# --------------------------------------------------------------------- config

ROOT = Path(__file__).resolve().parent
CORPUS_DIR = ROOT.parent / "corpus"
MANIFEST_PATH = CORPUS_DIR / "manifest.json"

# load .env (no python-dotenv dependency, KEY=VALUE lines only)
env_file = ROOT.parent / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

# SEC and good-citizen scraping REQUIRE a real UA with contact info.
CONTACT = os.environ["CORPUS_CONTACT"]
UA = {"User-Agent": CONTACT}

# how many docs per source
TARGETS = {
    "arxiv": 300,
    "sec": 200,
    "wikipedia": 200,
    "archive": 100,
    "datagov": 100,
    "rtd": 100,
}

# arXiv: mixed categories so parent/section STRUCTURE varies (the point of the
# pull). Count is split evenly across these.
ARXIV_CATEGORIES = [
    "cs.CL", "cs.LG", "cs.CV", "cs.DC",
    "physics.optics", "cond-mat.stat-mech", "astro-ph.GA",
    "econ.EM", "q-fin.ST", "math.PR", "stat.ML", "q-bio.NC",
]

# Wikipedia: top members of these categories, rendered to PDF.
WIKI_CATEGORIES = [
    "Machine learning", "Physics", "Economics", "History",
    "Biology", "Philosophy", "Mathematics", "Medicine",
    "Law", "Geology",
]

# ReadTheDocs projects that publish a PDF build. Over-provision past the target
# so dead/pdf-less projects don't starve the count. slug -> download path.
RTD_PROJECTS = [
    "requests", "flask", "numpy", "pandas", "scikit-learn", "django",
    "pytest", "sphinx", "click", "sqlalchemy", "pillow", "scrapy",
    "matplotlib", "networkx", "beautiful-soup-4", "urllib3", "celery",
    "jinja", "werkzeug", "pygments", "arrow-py", "attrs", "black",
    "boto3", "cryptography", "cython", "dask", "fastapi", "h5py",
    "httpx", "hypothesis", "lxml", "mkdocs", "opencv-python-tutroals",
    "paramiko", "plotly", "poetry", "psycopg", "pydantic-docs", "pymongo",
    "python-socketio", "pytorch", "rich", "seaborn", "setuptools",
    "statsmodels", "sympy", "tox", "tqdm", "typer", "uvicorn", "virtualenv",
    "xarray", "yaml", "aiohttp", "alembic", "ansible", "asyncpg", "bokeh",
    "conda", "coverage", "flake8", "gensim", "gunicorn", "keras", "kombu",
    "loguru", "mypy", "nltk", "notebook", "pip", "pluggy", "poetry-core",
    "prometheus-client", "pybind11", "pyarrow", "pygame", "pyinstaller",
    "pyproj", "pyserial", "pytest-django", "python-dateutil", "pytz",
    "redis-py", "requests-oauthlib", "scipy-lectures", "selenium-python",
    "spacy", "starlette", "tenacity", "transformers", "twisted", "watchdog",
    "websockets", "wtforms", "xgboost", "zarr", "boltons", "cachetools",
    "docutils", "flask-sqlalchemy", "invoke", "joblib", "markdown", "more-itertools",
]

RETRIES = 3
CHUNK = 1 << 16   # 64 KiB streaming


# ----------------------------------------------------------------- manifest io

def load_manifest():
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {}


def save_manifest(man):
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = MANIFEST_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(man, indent=2, ensure_ascii=False))
    tmp.replace(MANIFEST_PATH)   # atomic: a crash mid-write never corrupts it


def _key(source, doc_id):
    return f"{source}:{doc_id}"


def already_done(man, source, doc_id):
    """True if this doc is recorded done AND its file is still on disk."""
    rec = man.get(_key(source, doc_id))
    if not rec or rec.get("status") != "done":
        return False
    p = ROOT.parent / rec["path"]
    return p.exists() and p.stat().st_size > 0


def already_failed(man, source, doc_id):
    """True if a previous run already recorded this doc as failed — don't retry it."""
    rec = man.get(_key(source, doc_id))
    return bool(rec and rec.get("status") == "failed")


# --------------------------------------------------------------- http helpers

class RateLimiter:
    """Minimum seconds between requests, per host."""

    def __init__(self):
        self._last = {}

    def wait(self, host, min_interval):
        now = time.time()
        prev = self._last.get(host, 0.0)
        gap = now - prev
        if gap < min_interval:
            time.sleep(min_interval - gap)
        self._last[host] = time.time()


RL = RateLimiter()


def http_get(url, min_interval=0.3, stream=False, accept=None, params=None):
    """GET with per-host throttle + retry/backoff. Returns a requests.Response."""
    host = urllib.parse.urlparse(url).netloc
    headers = dict(UA)
    if accept:
        headers["Accept"] = accept
    last_err = None
    for attempt in range(RETRIES):
        RL.wait(host, min_interval)
        try:
            r = requests.get(url, headers=headers, params=params,
                             stream=stream, timeout=60)
            if r.status_code == 200:
                return r
            # 429/5xx are worth retrying; 4xx (except 429) are not
            if r.status_code not in (429, 500, 502, 503, 504):
                r.raise_for_status()
            last_err = f"HTTP {r.status_code}"
        except requests.RequestException as e:
            last_err = str(e)[:120]
        time.sleep(2 ** attempt)   # 1s, 2s, 4s backoff
    raise RuntimeError(f"GET failed after {RETRIES} tries ({last_err}): {url}")


def download_to(url, dest, min_interval=0.3, accept=None):
    """Stream a URL to dest; returns (bytes, sha256)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = http_get(url, min_interval=min_interval, stream=True, accept=accept)
    h = hashlib.sha256()
    n = 0
    tmp = dest.with_suffix(dest.suffix + ".part")
    with open(tmp, "wb") as f:
        for block in r.iter_content(CHUNK):
            if block:
                f.write(block)
                h.update(block)
                n += len(block)
    if n == 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"empty download: {url}")
    tmp.replace(dest)
    return n, h.hexdigest()


def record(man, source, doc_id, path, fmt, url, nbytes, sha, meta=None):
    man[_key(source, doc_id)] = {
        "source": source, "id": doc_id,
        "path": str(path.relative_to(ROOT.parent)), "format": fmt, "url": url,
        "status": "done", "bytes": nbytes, "sha256": sha, "meta": meta or {},
    }


def record_fail(man, source, doc_id, url, err):
    man[_key(source, doc_id)] = {
        "source": source, "id": doc_id, "url": url,
        "status": "failed", "err": str(err)[:200],
    }


# ------------------------------------------------------------------- fetchers
# Each fetcher yields work and downloads until `limit` NEW docs succeed. They
# all take (man, limit, dry) and update the manifest as they go.

def fetch_arxiv(man, limit, dry):
    """arXiv Atom API per category -> PDF. Honors arXiv's ~3s API cadence."""
    ns = {"a": "http://www.w3.org/2005/Atom"}
    per_cat = max(1, limit // len(ARXIV_CATEGORIES) + 1)
    got = 0
    for cat in ARXIV_CATEGORIES:
        if got >= limit:
            break
        api = ("http://export.arxiv.org/api/query"
               f"?search_query=cat:{cat}&start=0&max_results={per_cat}"
               "&sortBy=submittedDate&sortOrder=descending")
        try:
            resp = http_get(api, min_interval=3.0)   # arXiv API politeness
        except RuntimeError as e:
            print(f"  [arxiv] {cat} query failed: {e}")
            continue
        root = ET.fromstring(resp.text)
        for entry in root.findall("a:entry", ns):
            if got >= limit:
                break
            raw_id = entry.find("a:id", ns).text          # .../abs/2401.12345v1
            aid = raw_id.rsplit("/", 1)[-1]
            title = (entry.find("a:title", ns).text or "").strip().replace("\n", " ")
            if already_done(man, "arxiv", aid) or already_failed(man, "arxiv", aid):
                continue
            pdf_url = f"https://arxiv.org/pdf/{aid}.pdf"
            dest = CORPUS_DIR / "arxiv" / f"{aid.replace('/', '_')}.pdf"
            if dry:
                print(f"  [arxiv] would fetch {aid}  ({cat})")
                got += 1
                continue
            try:
                n, sha = download_to(pdf_url, dest, min_interval=1.0)
                record(man, "arxiv", aid, dest, "pdf", pdf_url, n, sha,
                       {"category": cat, "title": title})
                got += 1
                print(f"  [arxiv] {got}/{limit}  {aid}  {n//1024}KB")
            except RuntimeError as e:
                record_fail(man, "arxiv", aid, pdf_url, e)
                print(f"  [arxiv] FAIL {aid}: {e}")
            save_manifest(man)
    return got


def fetch_sec(man, limit, dry):
    """SEC EDGAR 10-K primary docs via the submissions API. Needs a real UA."""
    tickers_url = "https://www.sec.gov/files/company_tickers.json"
    try:
        companies = http_get(tickers_url, min_interval=0.2).json()
    except RuntimeError as e:
        print(f"  [sec] company list failed: {e}")
        return 0
    ciks = [str(v["cik_str"]).zfill(10) for v in companies.values()]
    got = 0
    for cik in ciks:
        if got >= limit:
            break
        sub_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        try:
            sub = http_get(sub_url, min_interval=0.15).json()
        except RuntimeError:
            continue
        recent = sub.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        accns = recent.get("accessionNumber", [])
        docs = recent.get("primaryDocument", [])
        for form, adsh, primary in zip(forms, accns, docs):
            if form != "10-K" or not primary:
                continue
            doc_id = adsh                                  # unique per filing
            if already_failed(man, "sec", doc_id):
                continue
            if already_done(man, "sec", doc_id):
                got += 1
                break
            adsh_nodash = adsh.replace("-", "")
            url = (f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
                   f"{adsh_nodash}/{primary}")
            ext = Path(primary).suffix or ".htm"
            dest = CORPUS_DIR / "sec" / f"{doc_id}{ext}"
            if dry:
                print(f"  [sec] would fetch 10-K {doc_id} (CIK {cik})")
                got += 1
                break
            try:
                n, sha = download_to(url, dest, min_interval=0.15)
                record(man, "sec", doc_id, dest, ext.lstrip("."), url, n, sha,
                       {"cik": cik})
                got += 1
                print(f"  [sec] {got}/{limit}  {doc_id}  {n//1024}KB")
            except RuntimeError as e:
                record_fail(man, "sec", doc_id, url, e)
                print(f"  [sec] FAIL {doc_id}: {e}")
            save_manifest(man)
            break                                          # one 10-K per company
    return got


def fetch_wikipedia(man, limit, dry):
    """Top members per category -> Wikipedia REST PDF render endpoint."""
    api = "https://en.wikipedia.org/w/api.php"
    per_cat = max(1, limit // len(WIKI_CATEGORIES) + 1)
    got = 0
    for cat in WIKI_CATEGORIES:
        if got >= limit:
            break
        params = {
            "action": "query", "list": "categorymembers",
            "cmtitle": f"Category:{cat}", "cmlimit": per_cat * 2,
            "cmtype": "page", "format": "json",
        }
        try:
            data = http_get(api, min_interval=0.3, params=params).json()
        except RuntimeError as e:
            print(f"  [wiki] {cat} listing failed: {e}")
            continue
        members = data.get("query", {}).get("categorymembers", [])
        taken = 0
        for m in members:
            if got >= limit or taken >= per_cat:
                break
            title = m["title"]
            doc_id = title.replace(" ", "_")
            if already_done(man, "wikipedia", doc_id) or already_failed(man, "wikipedia", doc_id):
                continue
            enc = urllib.parse.quote(doc_id, safe="")
            pdf_url = f"https://en.wikipedia.org/api/rest_v1/page/pdf/{enc}"
            dest = CORPUS_DIR / "wikipedia" / f"{doc_id}.pdf"
            if dry:
                print(f"  [wiki] would render {title}  ({cat})")
                got += 1
                taken += 1
                continue
            try:
                n, sha = download_to(pdf_url, dest, min_interval=0.5,
                                     accept="application/pdf")
                record(man, "wikipedia", doc_id, dest, "pdf", pdf_url, n, sha,
                       {"category": cat, "title": title})
                got += 1
                taken += 1
                print(f"  [wiki] {got}/{limit}  {title}  {n//1024}KB")
            except RuntimeError as e:
                record_fail(man, "wikipedia", doc_id, pdf_url, e)
                print(f"  [wiki] FAIL {title}: {e}")
            save_manifest(man)
    return got


def fetch_archive(man, limit, dry):
    """Internet Archive scanned texts (mediatype:texts, has a PDF derivative).
    Uses the `internetarchive` package requested in the plan."""
    try:
        from internetarchive import search_items, get_item
    except ImportError:
        print("  [archive] pip install internetarchive — skipping")
        return 0
    query = "mediatype:texts AND format:(Text PDF) AND year:[1850 TO 1960]"
    got = 0
    try:
        results = search_items(query, params={"rows": limit * 3})
        ids = [r["identifier"] for r in results]
    except Exception as e:
        print(f"  [archive] search failed: {e}")
        return 0
    for ident in ids:
        if got >= limit:
            break
        if already_failed(man, "archive", ident):
            continue
        if already_done(man, "archive", ident):
            got += 1
            continue
        if dry:
            print(f"  [archive] would fetch {ident}")
            got += 1
            continue
        try:
            item = get_item(ident)
            pdfs = [f for f in item.files if f["name"].lower().endswith(".pdf")]
            if not pdfs:
                continue
            f0 = min(pdfs, key=lambda f: int(f.get("size", 0) or 0))  # smallest PDF
            url = f"https://archive.org/download/{ident}/{urllib.parse.quote(f0['name'])}"
            dest = CORPUS_DIR / "archive" / f"{ident}.pdf"
            n, sha = download_to(url, dest, min_interval=1.0)
            record(man, "archive", ident, dest, "pdf", url, n, sha,
                   {"orig_name": f0["name"]})
            got += 1
            print(f"  [archive] {got}/{limit}  {ident}  {n//1024//1024}MB")
        except Exception as e:
            record_fail(man, "archive", ident, "", e)
            print(f"  [archive] FAIL {ident}: {str(e)[:80]}")
        save_manifest(man)
    return got


def fetch_datagov(man, limit, dry):
    """data.gov CKAN package_search, PDF resources."""
    api = "https://catalog.data.gov/api/3/action/package_search"
    got = 0
    start = 0
    page = 50
    while got < limit and start < 2000:
        params = {"q": "report OR form OR statistics", "rows": page,
                  "start": start, "fq": "res_format:PDF"}
        try:
            data = http_get(api, min_interval=0.5, params=params).json()
        except RuntimeError as e:
            print(f"  [datagov] search failed: {e}")
            break
        pkgs = data.get("result", {}).get("results", [])
        if not pkgs:
            break
        for pkg in pkgs:
            if got >= limit:
                break
            pdfs = [r for r in pkg.get("resources", [])
                    if (r.get("format", "").upper() == "PDF" and r.get("url"))]
            if not pdfs:
                continue
            res = pdfs[0]
            doc_id = res.get("id") or pkg.get("id")
            if already_failed(man, "datagov", doc_id):
                continue
            if already_done(man, "datagov", doc_id):
                got += 1
                continue
            url = res["url"]
            dest = CORPUS_DIR / "datagov" / f"{doc_id}.pdf"
            if dry:
                print(f"  [datagov] would fetch {doc_id}")
                got += 1
                continue
            try:
                n, sha = download_to(url, dest, min_interval=0.5)
                record(man, "datagov", doc_id, dest, "pdf", url, n, sha,
                       {"package": pkg.get("name")})
                got += 1
                print(f"  [datagov] {got}/{limit}  {doc_id}  {n//1024}KB")
            except RuntimeError as e:
                record_fail(man, "datagov", doc_id, url, e)
                print(f"  [datagov] FAIL {doc_id}: {str(e)[:80]}")
            save_manifest(man)
        start += page
    return got


def fetch_rtd(man, limit, dry):
    """ReadTheDocs PDF builds (technical manuals). Over-provisioned list."""
    got = 0
    for slug in RTD_PROJECTS:
        if got >= limit:
            break
        if already_failed(man, "rtd", slug):
            continue
        if already_done(man, "rtd", slug):
            got += 1
            continue
        url = f"https://{slug}.readthedocs.io/_/downloads/en/latest/pdf/"
        dest = CORPUS_DIR / "rtd" / f"{slug}.pdf"
        if dry:
            print(f"  [rtd] would fetch {slug}")
            got += 1
            continue
        try:
            n, sha = download_to(url, dest, min_interval=0.5,
                                 accept="application/pdf")
            record(man, "rtd", slug, dest, "pdf", url, n, sha, {})
            got += 1
            print(f"  [rtd] {got}/{limit}  {slug}  {n//1024}KB")
        except RuntimeError as e:
            record_fail(man, "rtd", slug, url, e)
            print(f"  [rtd] skip {slug}: {str(e)[:60]}")
        save_manifest(man)
    return got


FETCHERS = {
    "arxiv": fetch_arxiv, "sec": fetch_sec, "wikipedia": fetch_wikipedia,
    "archive": fetch_archive, "datagov": fetch_datagov, "rtd": fetch_rtd,
}


# ------------------------------------------------------------------- status

def print_status(man):
    from collections import Counter
    done = Counter()
    failed = Counter()
    bytes_by = Counter()
    for rec in man.values():
        if rec.get("status") == "done":
            done[rec["source"]] += 1
            bytes_by[rec["source"]] += rec.get("bytes", 0)
        elif rec.get("status") == "failed":
            failed[rec["source"]] += 1
    print(f"{'source':12}{'done':>6}{'target':>8}{'failed':>8}{'size':>12}")
    print("-" * 46)
    for s in TARGETS:
        mb = bytes_by[s] / 1e6
        print(f"{s:12}{done[s]:>6}{TARGETS[s]:>8}{failed[s]:>8}{mb:>10.1f}MB")
    total_mb = sum(bytes_by.values()) / 1e6
    print("-" * 46)
    print(f"{'TOTAL':12}{sum(done.values()):>6}{sum(TARGETS.values()):>8}"
          f"{sum(failed.values()):>8}{total_mb:>10.1f}MB")


# ---------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=list(FETCHERS) + ["all"], default="all")
    ap.add_argument("--limit", type=int, default=None,
                    help="override per-source target (for a quick trial)")
    ap.add_argument("--dry-run", action="store_true",
                    help="plan only: hit listing APIs, print what would download")
    ap.add_argument("--status", action="store_true",
                    help="print per-source counts from the manifest and exit")
    args = ap.parse_args()

    man = load_manifest()

    if args.status:
        print_status(man)
        return

    sources = list(FETCHERS) if args.source == "all" else [args.source]
    print(f"contact/User-Agent: {CONTACT!r}")
    if CONTACT.startswith("RAGdev research"):
        print("WARN: set CORPUS_CONTACT to your real name+email "
              "(SEC 403s a generic UA).")
    print(f"corpus dir: {CORPUS_DIR}\n")

    for s in sources:
        target = args.limit if args.limit is not None else TARGETS[s]
        print(f"== {s} (target {target}) ==")
        try:
            got = FETCHERS[s](man, target, args.dry_run)
        except Exception as e:
            print(f"  [{s}] aborted: {e}")
            got = 0
        finally:
            save_manifest(man)
        print(f"== {s}: {got} this run ==\n")

    print_status(man)
    print(f"\nmanifest -> {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
