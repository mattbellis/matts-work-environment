"""
Pull every paper by an INSPIRE author in a date window and build a
de-duplicated co-author list (name, latest affiliation, INSPIRE ID).
Optionally fold in the author list of the most recent collaboration
paper(s) and/or specific papers by INSPIRE record ID.

Usage:
    python inspire_coauthors.py
    python inspire_coauthors.py --collab CMS --collab-papers 3
    python inspire_coauthors.py --recids 2789123 2791456
    python inspire_coauthors.py --max-authors 30 --collab CMS
    python inspire_coauthors.py --check 3218942 --find Hogan   # diagnose one paper
    python inspire_coauthors.py --force-add-authors CMS 3218942
        # every author of every CMS paper in the window, plus that record,
        # whether or not those papers are linked to your INSPIRE profile

The CSV has both `name` (INSPIRE's "Last, First") and `display_name`
("First Last"), so search for either form.
"""
import argparse
import csv
import time

import requests

BAI = "M.Bellis.1"
MY_NAME = "Bellis, M"   # catches papers where your signature isn't linked to your profile
START, END = "2022-09-01", "2026-12-31"
API = "https://inspirehep.net/api/literature"
FIELDS = ",".join([
    "titles.title",
    "control_number",
    "earliest_date",
    "author_count",
    "collaborations.value",
    "authors.full_name",
    "authors.ids",
    "authors.record",
    "authors.affiliations.value",
])


MIN_INTERVAL = 0.4     # seconds between ANY two requests; INSPIRE allows ~15 / 5 s
MAX_RETRIES = 6        # attempts per request before giving up
RETRY_STATUS = {429, 500, 502, 503, 504}
_session = requests.Session()
_last_request = 0.0


def get_json(url, params=None):
    """GET with global pacing plus back-off-and-retry on rate limits and server errors."""
    global _last_request
    for attempt in range(MAX_RETRIES):
        # Pacing: applies across every request the script makes, not just pages
        wait = MIN_INTERVAL - (time.monotonic() - _last_request)
        if wait > 0:
            time.sleep(wait)
        _last_request = time.monotonic()

        try:
            r = _session.get(url, params=params, timeout=120,
                             headers={"Accept": "application/json"})
        except (requests.ConnectionError, requests.Timeout) as e:
            delay = 2 ** attempt
            print(f"  network error ({e.__class__.__name__}); retrying in {delay} s")
            time.sleep(delay)
            continue

        if r.status_code in RETRY_STATUS:
            # Honour Retry-After when INSPIRE sends it, else exponential back-off
            try:
                delay = float(r.headers.get("Retry-After", ""))
            except ValueError:
                delay = 2 ** attempt * (5 if r.status_code == 429 else 1)
            print(f"  HTTP {r.status_code}; retrying in {delay:.0f} s "
                  f"(attempt {attempt + 1}/{MAX_RETRIES})")
            time.sleep(delay)
            continue

        r.raise_for_status()   # other errors (400 bad query, 404) are not retried
        return r.json()

    raise RuntimeError(f"INSPIRE request failed after {MAX_RETRIES} attempts: {url}")


def fetch(q, size=25, max_papers=None):
    """Yield paper metadata for an INSPIRE query, following pagination."""
    url = API
    params = {"q": q, "size": size, "sort": "mostrecent", "fields": FIELDS}
    n = 0
    while url:
        data = get_json(url, params)
        for hit in data["hits"]["hits"]:
            yield hit["metadata"]
            n += 1
            if max_papers and n >= max_papers:
                return
        url = data.get("links", {}).get("next")
        params = None          # the "next" link already carries the query


def is_me(a):
    """Skip yourself whether or not the signature is linked to your profile."""
    if author_key(a) == BAI:
        return True
    return a.get("full_name", "").lower().startswith(MY_NAME.lower())


def display_name(full_name):
    """'Hogan, Julie M.' -> 'Julie M. Hogan' (INSPIRE stores Last, First)."""
    if "," in full_name:
        last, first = [s.strip() for s in full_name.split(",", 1)]
        return f"{first} {last}"
    return full_name


def author_key(a):
    """Best available unique ID: INSPIRE BAI > author record > name."""
    for i in a.get("ids", []):
        if i.get("schema") == "INSPIRE BAI":
            return i["value"]
    ref = a.get("record", {}).get("$ref")
    return ref.rsplit("/", 1)[-1] if ref else a["full_name"]


def add_paper(paper, coauthors, source):
    date = paper.get("earliest_date", "")
    for a in paper.get("authors", []):
        if is_me(a):
            continue
        key = author_key(a)
        affs = [x["value"] for x in a.get("affiliations", [])]
        entry = coauthors.setdefault(key, {
            "name": a["full_name"], "display_name": display_name(a["full_name"]),
            "inspire_id": key, "affiliation": "",
            "last_date": "", "n_papers": 0, "source": set()})
        entry["n_papers"] += 1
        entry["source"].add(source)
        if date >= entry["last_date"]:          # keep most recent affiliation
            entry["last_date"] = date
            entry["affiliation"] = "; ".join(affs) or entry["affiliation"]


def check(recid, own_q, find):
    """Explain why a given paper's authors do or don't end up in the CSV."""
    paper = next(fetch(f"recid:{recid}"), None)
    if paper is None:
        print(f"recid {recid}: not found on INSPIRE")
        return
    authors = paper.get("authors", [])
    print(f"Title:        {paper.get('titles', [{}])[0].get('title', '')[:80]}")
    print(f"Earliest date: {paper.get('earliest_date')}  (window {START} .. {END})")
    print(f"author_count: {paper.get('author_count')}   authors returned: {len(authors)}")

    me = [a for a in authors if is_me(a)]
    for a in me:
        print(f"You appear as: {a['full_name']!r}, linked id = {author_key(a)!r}")
    if not me:
        print("You do NOT appear in this paper's author list under the name/ID the script uses.")

    in_bai = next(fetch(f"recid:{recid} and a {BAI}"), None) is not None
    in_own = next(fetch(f"recid:{recid} and {own_q}"), None) is not None
    print(f"Matched by BAI-only search ('a {BAI}'):  {in_bai}")
    print(f"Matched by the script's own-paper search: {in_own}")

    if find:
        hits = [a["full_name"] for a in authors if find.lower() in a["full_name"].lower()]
        print(f"Authors matching '{find}': {hits or 'none'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-authors", type=int, default=None,
                    help="for YOUR papers, only include those with at most this many authors")
    ap.add_argument("--collab", default=None,
                    help="also include authors of the most recent paper(s) from this collaboration, e.g. CMS")
    ap.add_argument("--collab-papers", type=int, default=1,
                    help="how many of the most recent collaboration papers to union (default 1)")
    ap.add_argument("--recids", nargs="*", default=[],
                    help="INSPIRE record IDs of extra papers to include")
    ap.add_argument("--force-add-authors", nargs="*", default=[], metavar="CMS|RECID",
                    help="add every author of these, whether or not the papers are linked to you: "
                         "a record ID adds that paper; a collaboration name adds all its "
                         "full-author-list papers in the date window")
    ap.add_argument("--force-max-papers", type=int, default=None,
                    help="cap on collaboration papers pulled by --force-add-authors (default: all)")
    ap.add_argument("--out", default="coauthors.csv")
    ap.add_argument("--check", metavar="RECID",
                    help="diagnose one paper: is it found by the search, and who is on it?")
    ap.add_argument("--find", default="",
                    help="with --check: a name fragment to look for, e.g. Hogan")
    args = ap.parse_args()

    # Linked signatures (BAI) OR name match, so unclaimed papers are not missed
    own_q = f'(a {BAI} or a "{MY_NAME}") and de >= {START} and de <= {END}'
    if args.max_authors:
        own_q += f" and ac 1->{args.max_authors}"

    if args.check:
        check(args.check, own_q, args.find)
        return

    coauthors = {}

    # 1. Your own papers in the window
    n_mine = 0
    for paper in fetch(own_q):
        add_paper(paper, coauthors, "own papers")
        n_mine += 1
    print(f"{n_mine} of your papers in {START} .. {END}")

    # 2. Current collaboration author list (most recent full-author-list papers)
    if args.collab:
        cq = f"collaboration:{args.collab} and ac > 500 and doc_type:article"
        for paper in fetch(cq, size=args.collab_papers, max_papers=args.collab_papers):
            title = paper.get("titles", [{}])[0].get("title", "")
            print(f"{args.collab} paper {paper['control_number']} "
                  f"({paper.get('earliest_date')}, {paper.get('author_count')} authors): {title[:70]}")
            add_paper(paper, coauthors, f"{args.collab} author list")

    # 3. Any specific papers you name
    if args.recids:
        rq = " or ".join(f"recid:{r}" for r in args.recids)
        for paper in fetch(rq):
            add_paper(paper, coauthors, f"recid {paper['control_number']}")

    # 4. Forced additions: no link to your profile required
    for target in args.force_add_authors:
        if target.isdigit():                      # an INSPIRE record ID
            paper = next(fetch(f"recid:{target}"), None)
            if paper is None:
                print(f"force-add: recid {target} not found, skipped")
                continue
            add_paper(paper, coauthors, f"forced: recid {target}")
            print(f"force-add: recid {target} ({len(paper.get('authors', []))} authors)")
        else:                                     # a collaboration name
            fq = (f"collaboration:{target} and ac > 500 "
                  f"and de >= {START} and de <= {END}")
            n = 0
            # small pages: each full-author-list paper is a large response
            for paper in fetch(fq, size=5, max_papers=args.force_max_papers):
                add_paper(paper, coauthors, f"forced: {target}")
                n += 1
                if n % 25 == 0:
                    print(f"  force-add {target}: {n} papers so far...")
            print(f"force-add: {n} {target} papers in {START} .. {END}")

    rows = sorted(coauthors.values(), key=lambda e: e["name"].lower())
    for r in rows:
        r["source"] = "; ".join(sorted(r["source"]))
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["name"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} unique co-authors -> {args.out}")


if __name__ == "__main__":
    main()
