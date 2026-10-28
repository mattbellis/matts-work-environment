"""PHYS 310: Mechanics I, Fall 2026"""

COURSE = dict(
    name="PHYS 310: Mechanics I",
    #gradebook="2026-10-28T0145_Grades-PHYS310.10_anon.csv",
    gradebook="phys310_grades.csv",

    # Keyword rules: an assignment goes in the FIRST category whose `include` words
    # appear in its title and whose `exclude` words don't. Matching is case-sensitive,
    # which suits the ALL-CAPS prefixes you use in Canvas titles.
    categories={
        "labs": dict(
            label="Labs", weight=0.15,                     # CHECK weight
            include=["LAB"],
            canvas_group="Labs",
        ),
        "hw": dict(
            label="Homework, reflections, quizzes", weight=0.10,
            include=["HOMEWORK", "QUIZ", "ACTIVITY"],
            drop=0,                                        # CHECK
            canvas_group="Homeworks, reflections, quizzes, activities",
        ),
        "midterms": dict(
            label="Midterm exams", weight=0.45,
            include=["MIDTERM exam #0", "WITH EXTRA POINTS"],
            method="points",                               # Canvas-style: total points across exams
            canvas_group="Midterm exams",
        ),
        "final": dict(
            label="Final exam", weight=0.30,               # CHECK weight
            include=["FINAL EXAM"],
            canvas_group="Final exam",
        ),
    },

    ignore=["OPTIONAL"],           # titles containing these are never counted

    # Per-assignment exceptions, by Canvas ID (the number in parentheses) or exact title.
    # Value: a category key, or None to leave it out.
    overrides={
        490943: None,   # MIDTERM #1 - supplement: already folded into "...WITH EXTRA POINTS"
        # 489656: None, # MIDTERM exam #1 (original) -- uncomment if "WITH EXTRA POINTS" replaces it
    },

    graded_threshold=0.5,
    blank_as_zero=True,
    apply_drops=True,
    skip_students=[],

    email=dict(
        sender="matthew.bellis@gmail.com",
        cc=["mbellis@siena.edu"],
        reply_to="mbellis@siena.edu",
        signature="Prof. Bellis",
    ),
)
