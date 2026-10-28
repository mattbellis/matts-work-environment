"""
Check everything before grades are computed, and say plainly what looks wrong.

diagnose() prints a report and returns a list of (level, message) so a notebook cell
can stop if there are errors:  assert not [m for lvl, m in issues if lvl == "ERROR"]
"""

from __future__ import annotations

import pandas as pd

from .config import weight_sum

ERROR, WARN, NOTE = "ERROR", "WARN", "NOTE"


def diagnose(gb, classified, cfg, verbose=True):
    issues = []
    add = lambda lvl, msg: issues.append((lvl, msg))
    cats = cfg["categories"]

    # --- config
    ws = weight_sum(cfg)
    if abs(ws - 1) > 1e-6:
        add(WARN, f"Category weights sum to {ws:.3f}, not 1. Grades are still rescaled "
                  f"by the weights actually present, but check this is intended.")

    # --- roster
    for name, why in gb.skipped:
        add(NOTE, f"Skipped student {name!r}: {why}")
    for name, col, raw in gb.unparsed:
        add(WARN, f"Couldn't read score {raw!r} for {name} in {col!r}; treated as blank")

    # --- classification
    for col, a in classified.iterrows():
        t = a["title"]
        if a["status"] == "UNCLASSIFIED":
            add(WARN, f"UNCLASSIFIED: {t!r} ({a['canvas_id']}) matches no category and is not counted. "
                      f"Add a keyword or an override.")
        elif a["status"] == "AMBIGUOUS":
            add(WARN, f"AMBIGUOUS: {t!r} matches {a['matched']}; using {a['category']!r}. "
                      f"Add an exclude word or an override.")
        elif a["status"].startswith("ignored") and a["status"] != "ignored (0 points possible)":
            add(NOTE, f"Ignored {t!r} ({a['status']})")
        elif a["status"] == "ignored (0 points possible)":
            add(NOTE, f"Ignored {t!r}: 0 points possible")

        if a["category"] is not None and not a["graded"] and a["frac_entered"] > 0:
            add(WARN, f"Partly graded, NOT counted yet: {t!r} has scores for "
                      f"{a['frac_entered']:.0%} of students (threshold {cfg['graded_threshold']:.0%})")
        if a["category"] is not None and a["graded"] and a["frac_entered"] < 1:
            n = int(round((1 - a['frac_entered']) * len(gb.students)))
            how = "counted as 0" if cfg["blank_as_zero"] else "skipped"
            add(NOTE, f"{t!r}: {n} blank cell(s), {how}")

    ungraded = classified[classified["category"].notna() & (classified["frac_entered"] == 0)]
    if len(ungraded):
        add(NOTE, f"Not graded yet, not counted ({len(ungraded)}): "
                  + "; ".join(ungraded["title"]))

    # --- categories
    used = classified[classified["used"]]
    for key, c in cats.items():
        n = int((used["category"] == key).sum())
        if n == 0:
            add(NOTE, f"Category {key!r} has no graded assignments yet; it is left out of the grade "
                      f"(its {c['weight']:.0%} weight is shared out among the others)")
        elif c["drop"] >= n:
            add(WARN, f"Category {key!r}: drop={c['drop']} but only {n} graded; keeping 1")

    # --- excused
    n_ex = int(gb.excused[used.index].to_numpy().sum()) if len(used) else 0
    if n_ex:
        add(NOTE, f"{n_ex} excused (EX) cell(s) in counted assignments; these never count")

    if verbose:
        print_report(gb, classified, cfg, issues)
    return issues


def assignment_table(classified) -> pd.DataFrame:
    t = classified[["canvas_id", "title", "points_possible", "category", "status",
                    "frac_entered", "graded", "used"]].copy()
    t["frac_entered"] = (100 * t["frac_entered"]).round(0).astype(int).astype(str) + "%"
    return t.reset_index(drop=True)


def print_report(gb, classified, cfg, issues):
    used = classified[classified["used"]]
    print(f"{cfg['name']}  |  {gb.path.name}")
    print(f"{len(gb.students)} students, {len(gb.assignments)} assignment columns, "
          f"{len(used)} counted\n")

    print("Counted assignments by category")
    print("-" * 60)
    for key, c in cfg["categories"].items():
        sel = used[used["category"] == key]
        drop = f", drop {c['drop']}" if c["drop"] else ""
        print(f"{c['label']}  [{key}]  weight {c['weight']:.0%}{drop}, {c['method']}"
              f"  ->  {len(sel)} assignment(s)")
        for _, a in sel.iterrows():
            print(f"     {a['points_possible']:6.1f} pts  {a['title']}")
    print()

    for lvl in (ERROR, WARN, NOTE):
        msgs = [m for l, m in issues if l == lvl]
        if msgs:
            print(f"{lvl} ({len(msgs)})")
            for m in msgs:
                print(f"  - {m}")
            print()
    if not any(l in (ERROR, WARN) for l, _ in issues):
        print("No warnings.")
