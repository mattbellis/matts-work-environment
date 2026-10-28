# inspire_coauthors.py

Build a de-duplicated list of your co-authors from [INSPIRE-HEP](https://inspirehep.net) over a date window, written to a CSV.

## Why this exists

NSF's Collaborators and Other Affiliations (COA) form asks for everyone you have co-authored with over the past 48 months. For anyone in a large collaboration like CMS, that is thousands of people. Collecting them by hand isn't realistic, and the INSPIRE web interface won't export them in a usable form.

This script queries the INSPIRE REST API and produces one row per person. It handles three complications:

- **Unclaimed papers.** On large collaboration papers, INSPIRE often fails to link your author signature to your profile. Those papers drop out of a search on your INSPIRE ID.
- **New collaboration members.** If you joined recently, you may appear on few or no papers yet. You still need to list the current collaboration.
- **Duplicates.** The same person appears on hundreds of papers, sometimes under slightly different spellings.

## Setup

Requires Python 3.8+ and `requests`:

```
pip install requests
```

Edit the constants at the top of the script:

| Constant  | Meaning | Example |
|-----------|---------|---------|
| `BAI`     | Your INSPIRE author ID (shown on your INSPIRE profile) | `"M.Bellis.1"` |
| `MY_NAME` | Your name as INSPIRE writes it, "Last, F" | `"Bellis, M"` |
| `START`, `END` | Date window, `YYYY-MM-DD` | `"2022-09-01"`, `"2026-12-31"` |

## Usage

```
# Your own papers in the window
python inspire_coauthors.py

# Plus every author of every CMS paper in the window (recommended for COA)
python inspire_coauthors.py --force-add-authors CMS

# Plus specific papers by INSPIRE record ID
python inspire_coauthors.py --force-add-authors 3218942 3201177

# Only direct collaborators: skip papers with more than 30 authors
python inspire_coauthors.py --max-authors 30

# Diagnose why a particular paper or person is or isn't showing up
python inspire_coauthors.py --check 3218942 --find Hogan
```

The record ID is the number at the end of an INSPIRE paper URL, e.g. `https://inspirehep.net/literature/3218942`.

## Flags

| Flag | What it does |
|------|--------------|
| *(none)* | Finds your papers in the date window. A paper counts as yours if your signature is linked to `BAI` **or** the author name matches `MY_NAME`. |
| `--force-add-authors X [Y ...]` | Adds every author of the given targets, whether or not the papers are linked to you. A **number** is an INSPIRE record ID. A **name** (e.g. `CMS`) means all of that collaboration's papers with more than 500 authors inside the date window. |
| `--force-max-papers N` | Caps how many collaboration papers `--force-add-authors` pulls. Useful for a quick test run, e.g. `--force-max-papers 10`. |
| `--collab NAME` | Adds the authors of the *most recent* paper(s) from that collaboration, regardless of the date window. This is a snapshot of the current roster. |
| `--collab-papers N` | How many recent papers `--collab` unions (default 1; 2–3 is safer). |
| `--recids ID [ID ...]` | Adds specific papers by record ID. Same effect as giving numbers to `--force-add-authors`; kept for compatibility. |
| `--max-authors N` | Restricts **your own** papers to those with at most N authors. It does not affect `--force-add-authors`, `--collab` or `--recids`. |
| `--check RECID` | Diagnostic mode; writes no CSV. Reports the paper's date, how many authors INSPIRE returned versus its stated count, how you appear on the paper and whether that signature is linked, and whether each search finds the paper. |
| `--find TEXT` | Used with `--check`: lists the paper's authors whose names contain `TEXT`. |
| `--out FILE` | Output filename (default `coauthors.csv`). |

### `--force-add-authors CMS` vs. `--collab CMS`

- `--force-add-authors CMS` collects **everyone who was on any CMS paper during the window**. This matches the COA lookback period. It is slower and gives the longest list.
- `--collab CMS` collects **whoever is on the latest paper or papers**. It is fast, but it misses people who left during the window.

## Output

`coauthors.csv`, sorted by last name, one row per person:

| Column | Meaning |
|--------|---------|
| `name` | INSPIRE's form, `Last, First` |
| `display_name` | `First Last` |
| `inspire_id` | INSPIRE ID where available, otherwise the author record number or, failing that, the name |
| `affiliation` | Affiliation on the most recent paper the person appears on |
| `last_date` | Date of that most recent paper |
| `n_papers` | Number of collected papers the person appears on |
| `source` | Where the person came from: `own papers`, `forced: CMS`, `forced: recid …`, `CMS author list`, or several of these |

You are excluded from the list automatically, whether your signature is linked or matched by name.

## Rate limiting and timing

INSPIRE allows roughly **15 requests per 5 seconds** per client and returns HTTP 429 if you exceed it. Every request goes through a single helper, `get_json()`, which handles this in two layers.

**Staying under the limit**

1. **Global pacing.** The script waits at least `MIN_INTERVAL` (0.4 s) between *any* two requests: pages of one search, separate record-ID lookups, and `--check` lookups alike. That caps it at about 2.5 requests per second, below INSPIRE's limit of about 3 per second, however you combine the flags.
2. **Small pages for collaboration papers.** A CMS paper's author list is a large response, so `--force-add-authors` asks for 5 papers per page instead of the default 25.
3. **Fewer fields.** Every request asks only for the fields the script uses, not the full record. This cuts the download size several-fold.
4. **One connection.** A shared `requests.Session` reuses the connection to INSPIRE instead of opening a new one for each request.

**Recovering when something goes wrong**

| Problem | What the script does |
|---------|----------------------|
| HTTP 429 (rate limited) | Waits for the time given in INSPIRE's `Retry-After` header. Without the header, it waits 5 s, 10 s, 20 s, 40 s and so on, then retries. |
| HTTP 500/502/503/504 (server trouble) | Waits 1 s, 2 s, 4 s, 8 s and so on, then retries. |
| Connection error or timeout (120 s) | Waits 1 s, 2 s, 4 s and so on, then retries. |
| Other errors (e.g. 400 from a malformed query, 404) | Stops immediately. Retrying won't help. |

Each retry prints a line such as `HTTP 429; retrying in 10 s (attempt 2/6)`. After `MAX_RETRIES` (6) failed attempts on the same request, the script stops with an error. Six attempts with doubling waits ride out a few minutes of trouble; if INSPIRE is down for longer than that, rerun the script later. Both constants are at the top of the script.

In practice, `--force-add-authors CMS` over a four-year window takes several minutes. Most of that time is spent downloading author lists, not waiting on the pauses. Progress is printed every 25 papers.

## Edge effects and caveats

- **Name-match false positives.** Matching on `MY_NAME` ("Bellis, M") catches your unlinked papers, but it would also catch papers by any other author named "Bellis, M." in the window. Check the "of your papers" count, or use `--check` on anything surprising. The long-term fix is to claim your papers on your INSPIRE profile so that the ID match is enough.
- **Duplicate people.** Authors are de-duplicated by INSPIRE ID. If a co-author's signature isn't linked to a profile, the script falls back to the record number or the raw name. Spelling variants ("Smith, J." vs "Smith, John") then show up as separate rows. Sort the CSV by `name` and scan for near-duplicates.
- **Publication lag.** INSPIRE dates papers by their earliest date (usually arXiv submission), and collaboration author lists trail actual membership by months. People who joined recently may not appear yet, and people who left may still appear. For an authoritative current roster, ask the collaboration secretariat.
- **The 500-author threshold.** Collaboration papers are identified as those with more than 500 authors, which selects full-collaboration CMS papers and skips small-author-list notes. For a smaller collaboration, change `ac > 500` in the script.
- **Window edges.** The window uses INSPIRE's earliest date. A paper submitted to arXiv before `START` but published afterwards is excluded.
- **Affiliations.** Only the affiliation on a person's most recent collected paper is kept. People with multiple affiliations get them joined with `;`.
- **What NSF actually requires.** NSF has specific guidance on listing large-collaboration co-authors on the COA form. Check the current PAPPG or your sponsored-programs office before deciding between the full list and `--max-authors` (direct collaborators only).
