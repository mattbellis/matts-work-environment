"""
Course configuration: a plain Python dict, checked and filled with defaults here.

See courses/TEMPLATE.py for an annotated example of every key.
"""

from __future__ import annotations

import copy
import difflib

DEFAULT_LETTER_SCALE = [  # (minimum percent, letter), highest first
    (93, "A"), (90, "A-"), (87, "B+"), (83, "B"), (80, "B-"), (77, "C+"),
    (73, "C"), (70, "C-"), (67, "D+"), (63, "D"), (60, "D-"), (0, "F"),
]

TOP_DEFAULTS = dict(
    name="(unnamed course)",
    gradebook=None,
    categories={},
    ignore=[],                 # keywords: any assignment whose title contains one is ignored
    overrides={},              # {canvas_id or exact title: "category" or None}
    case_sensitive=True,       # keyword matching
    graded_threshold=0.5,      # an assignment counts once this fraction of students have a score
    blank_as_zero=True,        # in a graded assignment, a blank cell counts as 0 (EX is always excluded)
    apply_drops=True,          # True: the reported course grade uses drops
    letter_scale=DEFAULT_LETTER_SCALE,
    skip_students=[],          # names or emails to leave out entirely
    output_dir="output",
    email={},
)

CATEGORY_DEFAULTS = dict(
    label=None,                # how it is written in emails/plots; defaults to the key
    weight=0.0,
    include=[],                # title must contain at least one of these...
    exclude=[],                # ...and none of these
    drop=0,                    # drop this many lowest scores (by percent)
    method="mean",             # "mean": each assignment counts equally; "points": total points (Canvas-style)
    canvas_group=None,         # name of the matching Canvas assignment group, for cross-checking
)

EMAIL_DEFAULTS = dict(
    sender=None,               # the address you log in with
    cc=[],                     # list of addresses copied on every message
    reply_to=None,
    smtp_host="smtp.gmail.com",
    smtp_port=465,
    subject="{course}: grade summary for {first} {last}",
    greeting="Hi {first},",
    intro="Here is a summary of your grades so far. The attached plot shows each of your "
          "scores (red stars) against the spread of the class (boxes), and your running "
          "course average (black line).",
    signoff="Let me know if you see anything that looks wrong.",
    signature="",
    show_letter=False,         # include a letter grade in the email
    show_no_drops=True,        # also show the averages before dropping
)


class ConfigError(ValueError):
    pass


def _check_keys(d, allowed, where):
    for k in d:
        if k not in allowed:
            hint = difflib.get_close_matches(k, allowed, n=1)
            raise ConfigError(f"Unknown key {k!r} in {where}"
                              + (f" (did you mean {hint[0]!r}?)" if hint else ""))


def load_config(cfg: dict) -> dict:
    """Return a copy of cfg with defaults filled in. Raises ConfigError on mistakes."""
    _check_keys(cfg, TOP_DEFAULTS, "course config")
    out = copy.deepcopy(TOP_DEFAULTS)
    out.update(copy.deepcopy(cfg))

    if not out["categories"]:
        raise ConfigError("No categories defined")
    cats = {}
    for key, c in out["categories"].items():
        _check_keys(c, CATEGORY_DEFAULTS, f"category {key!r}")
        full = copy.deepcopy(CATEGORY_DEFAULTS)
        full.update(c)
        full["label"] = full["label"] or key
        if full["method"] not in ("mean", "points"):
            raise ConfigError(f"category {key!r}: method must be 'mean' or 'points'")
        for kw in ("include", "exclude"):
            if isinstance(full[kw], str):
                full[kw] = [full[kw]]
        cats[key] = full
    out["categories"] = cats

    for target in out["overrides"].values():
        if target is not None and target not in cats:
            raise ConfigError(f"override points to unknown category {target!r}")

    _check_keys(out["email"], EMAIL_DEFAULTS, "email settings")
    em = copy.deepcopy(EMAIL_DEFAULTS)
    em.update(out["email"])
    if isinstance(em["cc"], str):
        em["cc"] = [em["cc"]]
    out["email"] = em

    out["letter_scale"] = sorted(out["letter_scale"], key=lambda t: -t[0])
    return out


def weight_sum(cfg) -> float:
    return sum(c["weight"] for c in cfg["categories"].values())
