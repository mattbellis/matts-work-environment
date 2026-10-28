"""
Load a Canvas gradebook export into clean, separate tables.

A Canvas export has:
  * identity columns (Student, ID, SIS User ID, SIS Login ID, Section)
  * one column per assignment, named "<title> (<canvas id>)"
  * Canvas's own computed columns ("Labs Current Score", "Final Score", ...)
  * a "Points Possible" row (and sometimes a "Muted" row) above the students
  * Canvas's built-in "Student, Test" row

Cells are kept as three things per student x assignment:
  points   : float, NaN when blank or excused
  excused  : True where Canvas shows "EX"
  blank    : True where the cell is empty (not graded / not submitted)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

IDENTITY_COLS = ["Student", "ID", "SIS User ID", "SIS Login ID", "Section"]
ASSIGNMENT_RE = re.compile(r"^(?P<title>.*\S)\s*\((?P<cid>\d+)\)$")
META_ROW_NAMES = {"Points Possible", "Muted"}
TEST_STUDENT_NAMES = {"Student, Test", "Test Student"}


@dataclass
class Gradebook:
    path: Path
    students: pd.DataFrame        # index = Canvas ID; first, last, name, email, section
    assignments: pd.DataFrame     # index = column name; title, canvas_id, points_possible (sorted by canvas_id)
    points: pd.DataFrame          # students x assignments, float, NaN = blank or excused
    excused: pd.DataFrame         # students x assignments, bool
    blank: pd.DataFrame           # students x assignments, bool
    canvas_scores: pd.DataFrame   # Canvas's own computed columns, numeric
    skipped: list = field(default_factory=list)      # (name, reason)
    unparsed: list = field(default_factory=list)     # (student name, assignment, raw text)

    def __repr__(self):
        return (f"Gradebook({self.path.name}: {len(self.students)} students, "
                f"{len(self.assignments)} assignments)")


def _split_name(full: str):
    if "," in full:
        last, first = full.split(",", 1)
        return first.strip(), last.strip()
    parts = full.split()
    return (" ".join(parts[:-1]), parts[-1]) if len(parts) > 1 else (full, "")


def load_canvas(path, skip_students=()) -> Gradebook:
    """Read a Canvas gradebook CSV.

    skip_students: names (as in the Student column, e.g. "Lastname, First")
                   or emails of real students to leave out (withdrawals, auditors).
    """
    path = Path(path)
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in IDENTITY_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: not a Canvas gradebook? Missing columns {missing}")
    df["Student"] = df["Student"].str.strip()

    # ---- separate metadata rows, test student, skipped students, real students
    skip_set = {s.strip().lower() for s in skip_students}
    pts_row = None
    keep, skipped = [], []
    for i, row in df.iterrows():
        name = row["Student"]
        if name == "Points Possible":
            pts_row = row
        elif name in META_ROW_NAMES or not name:
            continue
        elif name in TEST_STUDENT_NAMES:
            skipped.append((name, "Canvas test student"))
        elif name.lower() in skip_set or row["SIS Login ID"].strip().lower() in skip_set:
            skipped.append((name, "listed in skip_students"))
        else:
            keep.append(i)
    if pts_row is None:
        raise ValueError(f"{path.name}: no 'Points Possible' row found")
    sdf = df.loc[keep]

    # ---- assignment columns vs. Canvas computed columns
    acols = [c for c in df.columns if c not in IDENTITY_COLS and ASSIGNMENT_RE.match(c)]
    ccols = [c for c in df.columns if c not in IDENTITY_COLS and c not in acols]

    arows = []
    for c in acols:
        m = ASSIGNMENT_RE.match(c)
        try:
            pp = float(pts_row[c])
        except ValueError:
            pp = np.nan
        arows.append(dict(column=c, title=m["title"], canvas_id=int(m["cid"]), points_possible=pp))
    assignments = (pd.DataFrame(arows).set_index("column")
                   .sort_values("canvas_id"))          # Canvas IDs increase in creation order
    acols = list(assignments.index)

    # ---- students
    ids = sdf["ID"].str.strip()
    first_last = [_split_name(n) for n in sdf["Student"]]
    students = pd.DataFrame({
        "first": [f for f, _ in first_last],
        "last": [l for _, l in first_last],
        "name": list(sdf["Student"]),
        "email": list(sdf["SIS Login ID"].str.strip()),
        "section": list(sdf["Section"].str.strip()),
    }, index=pd.Index(ids, name="canvas_id"))
    if students.index.duplicated().any():
        raise ValueError("Duplicate student IDs in the gradebook")

    # ---- scores
    raw = sdf[acols].apply(lambda s: s.str.strip())
    raw.index = students.index
    blank = raw.eq("")
    excused = raw.apply(lambda s: s.str.upper().eq("EX"))
    points = raw.apply(pd.to_numeric, errors="coerce")
    unparsed_mask = points.isna() & ~blank & ~excused
    unparsed = [(students.at[sid, "name"], col, raw.at[sid, col])
                for sid, col in zip(*np.where(unparsed_mask))
                for sid, col in [(raw.index[sid], raw.columns[col])]]
    blank = blank | unparsed_mask          # treat unreadable cells as blank (and report them)

    canvas_scores = sdf[ccols].apply(pd.to_numeric, errors="coerce")
    canvas_scores.index = students.index

    return Gradebook(path, students, assignments, points, excused, blank,
                     canvas_scores, skipped, unparsed)
