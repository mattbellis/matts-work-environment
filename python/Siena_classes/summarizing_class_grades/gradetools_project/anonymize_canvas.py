#!/usr/bin/env python3
"""
Anonymize a Canvas gradebook export (.csv) so it can be shared safely.

What it changes (student rows only):
    Student       -> "Lastname07, Student07"   (keeps the "Last, First" format)
    ID            -> 900007                    (fake Canvas ID)
    SIS User ID   -> 000000007                 (fake, same width as typical SIDs)
    SIS Login ID  -> student07@siena.example   (.example TLD is reserved; undeliverable)

What it leaves alone:
    - The header row (assignment names and Canvas assignment IDs)
    - The "Points Possible" row (and any other non-student metadata rows)
    - Canvas's built-in "Student, Test" row (not a real person; useful for testing skip logic)
    - Section and every grade/score cell, byte-for-byte

Usage:
    python anonymize_canvas.py GRADES.csv                    # writes GRADES_anon.csv
    python anonymize_canvas.py GRADES.csv -o out.csv
    python anonymize_canvas.py GRADES.csv --shuffle          # also randomize row order
    python anonymize_canvas.py GRADES.csv --key key.csv      # save real<->fake map LOCALLY

Do NOT upload the --key file anywhere; it contains the real names.
"""

import argparse
import csv
import random
import sys
from pathlib import Path

IDENTITY_COLS = ["Student", "ID", "SIS User ID", "SIS Login ID"]
TEST_STUDENT_NAMES = {"Student, Test", "Test Student"}


def is_metadata_row(row, col):
    """Rows like 'Points Possible' or 'Muted' that aren't students."""
    name = row[col["Student"]].strip()
    has_ids = any(row[col[c]].strip() for c in ("ID", "SIS User ID", "SIS Login ID"))
    return (not has_ids) and name in {"Points Possible", "Muted", ""}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("infile")
    ap.add_argument("-o", "--outfile")
    ap.add_argument("--shuffle", action="store_true",
                    help="randomize student row order (so roster order can't identify anyone)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--key", help="optional path for a real->fake mapping CSV (keep private)")
    args = ap.parse_args()

    inpath = Path(args.infile)
    outpath = Path(args.outfile) if args.outfile else inpath.with_name(inpath.stem + "_anon.csv")

    # utf-8-sig strips a BOM if Canvas added one
    with open(inpath, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))

    header = rows[0]
    missing = [c for c in IDENTITY_COLS if c not in header]
    if missing:
        sys.exit(f"ERROR: expected columns not found: {missing}")
    col = {c: header.index(c) for c in IDENTITY_COLS}

    body = [r for r in rows[1:] if any(cell.strip() for cell in r)]  # drop blank lines
    meta = [r for r in body if is_metadata_row(r, col)]
    tests = [r for r in body if r[col["Student"]].strip() in TEST_STUDENT_NAMES]
    students = [r for r in body if r not in meta and r not in tests]

    if args.shuffle:
        random.Random(args.seed).shuffle(students)

    width = max(2, len(str(len(students))))
    key_rows = []
    for i, r in enumerate(students, start=1):
        n = f"{i:0{width}d}"
        real = [r[col[c]] for c in IDENTITY_COLS]
        r[col["Student"]] = f"Lastname{n}, Student{n}"
        r[col["ID"]] = str(900000 + i)
        r[col["SIS User ID"]] = f"{i:09d}"
        r[col["SIS Login ID"]] = f"student{n}@siena.example"
        key_rows.append(real + [r[col[c]] for c in IDENTITY_COLS])

    with open(outpath, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(meta + students + tests)

    if args.key:
        with open(args.key, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([f"real {c}" for c in IDENTITY_COLS] + [f"anon {c}" for c in IDENTITY_COLS])
            w.writerows(key_rows)

    # Sanity check: make sure no original identity value survives anywhere in the output
    leaked = []
    out_text = outpath.read_text(encoding="utf-8")
    for real_vals in (k[:4] for k in key_rows):
        for v in real_vals:
            v = v.strip()
            if len(v) >= 3 and v in out_text:
                leaked.append(v)

    print(f"Read {len(students)} students, {len(meta)} metadata row(s), "
          f"{len(tests)} test-student row(s)")
    print(f"Wrote {outpath}")
    if args.key:
        print(f"Wrote key {args.key}  (keep this private)")
    if leaked:
        print(f"WARNING: {len(leaked)} original value(s) still appear in the output, e.g. "
              f"{leaked[:3]!r} -- check before sharing.")
    else:
        print("Check passed: no original names/IDs/emails found in the output.")


if __name__ == "__main__":
    main()
