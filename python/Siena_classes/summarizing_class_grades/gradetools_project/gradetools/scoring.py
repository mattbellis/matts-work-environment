"""
Turn points into category averages and a course grade.

Rules:
  * Only assignments with used == True (categorized and graded) count.
  * EX (excused) cells never count.
  * Blank cells in a graded assignment count as 0 if cfg["blank_as_zero"], else are skipped.
  * Drops remove the lowest N percentages in a category, but always keep at least one.
  * method "mean": average of percentages; "points": sum(points) / sum(possible).
  * A category with no scores yet is left out and the remaining weights are rescaled,
    so the grade is "the grade so far".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Results:
    cfg: dict
    students: pd.DataFrame
    assignments: pd.DataFrame       # the used assignments, in Canvas order, with category
    pct: pd.DataFrame               # students x assignments, fraction (1.0 = 100%), NaN = not counted
    counted_zero: pd.DataFrame      # True where a blank was counted as 0
    dropped: pd.DataFrame           # True where a score was dropped
    cat_drops: pd.DataFrame         # students x categories, fraction
    cat_nodrops: pd.DataFrame
    grade_drops: pd.Series          # fraction
    grade_nodrops: pd.Series
    running: pd.DataFrame           # students x assignments: course grade (no drops) after each one

    @property
    def grade(self) -> pd.Series:
        return self.grade_drops if self.cfg["apply_drops"] else self.grade_nodrops

    @property
    def categories(self) -> pd.DataFrame:
        return self.cat_drops if self.cfg["apply_drops"] else self.cat_nodrops

    def letter(self, grade=None) -> pd.Series:
        g = self.grade if grade is None else grade
        return g.apply(lambda x: to_letter(100 * x, self.cfg["letter_scale"]))

    def summary(self) -> pd.DataFrame:
        """One row per student: name, course grade(s), letter, category averages (percent)."""
        labels = {k: c["label"] for k, c in self.cfg["categories"].items()}
        df = pd.DataFrame({
            "last": self.students["last"], "first": self.students["first"],
            "grade": 100 * self.grade,
            "grade (no drops)": 100 * self.grade_nodrops,
            "grade (drops)": 100 * self.grade_drops,
            "letter": self.letter(),
        })
        cats = (100 * self.categories).rename(columns=labels)
        return df.join(cats).sort_values(["last", "first"]).round(2)


def to_letter(percent, scale):
    if percent != percent:
        return ""
    for cutoff, letter in scale:
        if percent >= cutoff - 1e-9:
            return letter
    return scale[-1][1]


def _category_average(p, pts, drop, method):
    """p: fractions for one student in one category (NaN = not counted), pts: points possible.
    Returns (average or NaN, boolean mask of dropped entries)."""
    ok = ~np.isnan(p)
    dropped = np.zeros_like(ok)
    if not ok.any():
        return np.nan, dropped
    idx = np.where(ok)[0]
    n_drop = min(drop, len(idx) - 1)
    if n_drop > 0:
        lowest = idx[np.argsort(p[idx], kind="stable")[:n_drop]]
        dropped[lowest] = True
        idx = np.setdiff1d(idx, lowest)
    if method == "points":
        return np.sum(p[idx] * pts[idx]) / np.sum(pts[idx]), dropped
    return np.mean(p[idx]), dropped


def _course_grade(cat_row, weights):
    have = cat_row.notna()
    if not have.any():
        return np.nan
    w = weights[have]
    return float((cat_row[have] * w).sum() / w.sum()) if w.sum() > 0 else np.nan


def compute(gb, classified, cfg) -> Results:
    used = classified[classified["used"]]
    cols = list(used.index)
    pts = used["points_possible"].to_numpy(float)

    raw = gb.points[cols].copy()
    counted_zero = gb.blank[cols] & ~gb.excused[cols] if cfg["blank_as_zero"] else gb.blank[cols] & False
    raw = raw.mask(counted_zero, 0.0)
    pct = raw / pts                                    # NaN stays NaN (excused, or blank when not zeroed)

    cats = list(cfg["categories"])
    weights = pd.Series({k: c["weight"] for k, c in cfg["categories"].items()})
    cat_d = pd.DataFrame(np.nan, index=pct.index, columns=cats)
    cat_n = cat_d.copy()
    dropped = pd.DataFrame(False, index=pct.index, columns=cols)

    for key, c in cfg["categories"].items():
        in_cat = (used["category"] == key).to_numpy()
        if not in_cat.any():
            continue
        ccols = [col for col, m in zip(cols, in_cat) if m]
        cpts = pts[in_cat]
        for sid in pct.index:
            p = pct.loc[sid, ccols].to_numpy(float)
            cat_n.at[sid, key], _ = _category_average(p, cpts, 0, c["method"])
            cat_d.at[sid, key], d = _category_average(p, cpts, c["drop"], c["method"])
            dropped.loc[sid, ccols] = d

    grade_d = cat_d.apply(_course_grade, axis=1, weights=weights)
    grade_n = cat_n.apply(_course_grade, axis=1, weights=weights)

    # running course grade (no drops) after each assignment, in Canvas order
    running = pd.DataFrame(np.nan, index=pct.index, columns=cols)
    for sid in pct.index:
        p = pct.loc[sid].to_numpy(float)
        for j in range(len(cols)):
            row = {}
            for key, c in cfg["categories"].items():
                m = (used["category"].to_numpy()[: j + 1] == key)
                if m.any():
                    row[key] = _category_average(p[: j + 1][m], pts[: j + 1][m], 0, c["method"])[0]
            running.iat[running.index.get_loc(sid), j] = _course_grade(
                pd.Series(row, dtype=float), weights[list(row)]) if row else np.nan

    return Results(cfg, gb.students, used, pct, counted_zero, dropped,
                   cat_d, cat_n, grade_d, grade_n, running)


def canvas_check(gb, classified, cfg) -> pd.DataFrame:
    """For categories with `canvas_group` set, recompute the category the way Canvas does
    (total points, blanks skipped, no drops) and compare with Canvas's "<group> Current Score".
    A mismatch usually means an assignment is in a different category here than in Canvas.
    Returns one row per category: Canvas group, n students compared, max |difference|."""
    rows = []
    for key, c in cfg["categories"].items():
        grp = c["canvas_group"]
        if not grp:
            continue
        colname = f"{grp} Current Score"
        if colname not in gb.canvas_scores:
            rows.append(dict(category=key, canvas_group=grp, compared=0, max_diff=np.nan,
                             note=f"no column {colname!r} in export"))
            continue
        sel = classified[(classified["category"] == key) & classified["graded"]]
        if sel.empty:
            rows.append(dict(category=key, canvas_group=grp, compared=0, max_diff=np.nan,
                             note="no graded assignments here"))
            continue
        p = gb.points[sel.index]
        mine = 100 * p.sum(axis=1, min_count=1) / (p.notna() * sel["points_possible"]).sum(axis=1)
        theirs = gb.canvas_scores[colname]
        both = mine.notna() & theirs.notna()
        diff = (mine - theirs)[both].abs()
        rows.append(dict(category=key, canvas_group=grp, compared=int(both.sum()),
                         max_diff=round(diff.max(), 2) if len(diff) else np.nan,
                         note="ok" if len(diff) and diff.max() < 0.05 else "MISMATCH"))
        rows[-1]["_scores"] = mine

    # Overall: combine the Canvas-style category scores with the config weights and compare
    # with Canvas's "Current Score". A mismatch here (with categories ok) means the weights differ.
    if rows and "Current Score" in gb.canvas_scores and len(rows) == len(cfg["categories"]):
        cat = pd.DataFrame({r["category"]: r.get("_scores") for r in rows if "_scores" in r})
        w = pd.Series({k: cfg["categories"][k]["weight"] for k in cat})
        mine = cat.apply(lambda r: (r[r.notna()] * w[r.notna()]).sum() / w[r.notna()].sum()
                         if r.notna().any() else np.nan, axis=1)
        theirs = gb.canvas_scores["Current Score"]
        both = mine.notna() & theirs.notna()
        diff = (mine - theirs)[both].abs()
        rows.append(dict(category="(course)", canvas_group="Current Score", compared=int(both.sum()),
                         max_diff=round(diff.max(), 2) if len(diff) else np.nan,
                         note="ok" if len(diff) and diff.max() < 0.05
                         else "MISMATCH: weights differ from Canvas's group weights"))
    out = pd.DataFrame(rows)
    return out.drop(columns="_scores", errors="ignore")
