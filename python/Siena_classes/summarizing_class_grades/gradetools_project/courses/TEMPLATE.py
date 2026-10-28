"""
Annotated template. Copy to courses/<course>_<term>.py and edit.
Only `name`, `gradebook` and `categories` are required; everything else has a default.
"""

COURSE = dict(
    name="PHYS 999: Course Name",                 # used in subjects, plots, file names
    gradebook="2026-12-15T0900_Grades-PHYS999.csv",

    categories={
        # key: short name used in code.  label: what students see.
        "hw": dict(
            label="Homework and quizzes",
            weight=0.20,
            include=["HOMEWORK", "QUIZ"],          # title must contain one of these...
            exclude=["REFLECTION", "PRE-READING"], # ...and none of these
            drop=2,                                # drop the 2 lowest (by percent); keeps at least 1
            method="mean",                         # "mean" = each assignment equal; "points" = sum/sum
            canvas_group=None,                     # e.g. "Homework" to cross-check with Canvas
        ),
        "hwref": dict(label="Homework reflections", weight=0.10, include=["REFLECTION"], drop=1),
        "midterm": dict(label="Midterm exam", weight=0.30, include=["MIDTERM"]),
        "final": dict(label="Final exam", weight=0.40, include=["FINAL EXAM"]),
    },
    # A title that matches more than one category goes to the FIRST one listed and is
    # flagged AMBIGUOUS in the diagnostics, so order categories from most to least specific
    # if you rely on that, or (better) add an exclude word.

    ignore=["OPTIONAL", "REMINDER"],   # never count titles containing these
    case_sensitive=True,               # keyword matching

    # One-off exceptions. Key: Canvas ID (the number in parentheses) or exact title.
    # Value: a category key, or None to leave the assignment out.
    overrides={
        # 123456: None,
        # "HOMEWORK - Chapter 3a": None,
        # 234567: "hw",
    },

    graded_threshold=0.5,   # count an assignment once this fraction of students have a score
                            # (replaces hand-listing ungraded assignments to drop)
    blank_as_zero=True,     # blank cell in a counted assignment = 0. EX (excused) never counts.
    apply_drops=True,       # which grade is "the" grade; both are always computed
    skip_students=[],       # e.g. ["Rivera, Ana"] or emails: withdrawals, auditors

    letter_scale=[(93, "A"), (90, "A-"), (87, "B+"), (83, "B"), (80, "B-"), (77, "C+"),
                  (73, "C"), (70, "C-"), (67, "D+"), (63, "D"), (60, "D-"), (0, "F")],

    output_dir="output",

    email=dict(
        sender="matthew.bellis@gmail.com",    # login address (Gmail App Password)
        cc=["mbellis@siena.edu"],
        reply_to="mbellis@siena.edu",
        subject="{course}: grade summary for {first} {last}",
        greeting="Hi {first},",
        intro="Here is a summary of your grades so far. ...",
        signoff="Let me know if you see anything that looks wrong.",
        signature="Prof. Bellis",
        show_letter=False,      # put a letter grade in the email
        show_no_drops=True,     # also show averages before drops
    ),
)
