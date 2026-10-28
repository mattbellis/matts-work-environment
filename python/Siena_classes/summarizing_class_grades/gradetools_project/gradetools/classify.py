"""
Put each assignment into a category, and decide whether it has been graded.

Order of rules for each assignment (first one that applies wins):
  1. overrides     - by Canvas ID (int) or exact title; value is a category or None (= ignore)
  2. ignore        - title contains any keyword in cfg["ignore"]
  3. zero points   - points possible is 0 or missing (can't be turned into a percent)
  4. categories    - title contains one of the category's `include` words and none of its
                     `exclude` words. If several categories match, the first one in the
                     config wins and the assignment is flagged as AMBIGUOUS.
  5. otherwise     - UNCLASSIFIED (flagged)

Then "graded" = at least cfg["graded_threshold"] of students have a score or EX.
"""

from __future__ import annotations

import pandas as pd


def _contains(title, word, case_sensitive):
    return word in title if case_sensitive else word.lower() in title.lower()


def classify(gb, cfg) -> pd.DataFrame:
    """Return gb.assignments with extra columns:
    category (str or None), status, matched (all categories that matched),
    frac_entered, graded, used (= has a category and is graded)
    """
    cs = cfg["case_sensitive"]
    rows = []
    for col, a in gb.assignments.iterrows():
        title = a["title"]
        cat, status, matched = None, "", []

        if a["canvas_id"] in cfg["overrides"] or title in cfg["overrides"]:
            cat = cfg["overrides"].get(a["canvas_id"], cfg["overrides"].get(title))
            status = "override" if cat else "ignored (override)"
        elif any(_contains(title, w, cs) for w in cfg["ignore"]):
            status = "ignored (keyword)"
        elif not a["points_possible"] > 0:
            status = "ignored (0 points possible)"
        else:
            for key, c in cfg["categories"].items():
                if (any(_contains(title, w, cs) for w in c["include"])
                        and not any(_contains(title, w, cs) for w in c["exclude"])):
                    matched.append(key)
            if len(matched) == 1:
                cat, status = matched[0], "ok"
            elif len(matched) > 1:
                cat, status = matched[0], "AMBIGUOUS"
            else:
                status = "UNCLASSIFIED"

        entered = (~gb.blank[col]).mean() if len(gb.students) else 0.0
        rows.append(dict(column=col, category=cat, status=status,
                         matched=", ".join(matched), frac_entered=entered))

    info = pd.DataFrame(rows).set_index("column")
    out = gb.assignments.join(info)
    out["graded"] = out["frac_entered"] >= cfg["graded_threshold"]
    out["used"] = out["category"].notna() & out["graded"]
    return out
