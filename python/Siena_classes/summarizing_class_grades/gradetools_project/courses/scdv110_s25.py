"""SCDV 110: Exploratory Data Analysis and Visualization, Spring 2025 (converted from the old notebook)

In the old notebook, OPTIONAL EXTRA WORK was added as one more quiz score. Listing it in the
quiz category's `include` does the same thing (it can then also be dropped like any quiz).
"""

COURSE = dict(
    name="SCDV 110: Exploratory Data Analysis and Visualization",
    gradebook="scdv110s25.csv",
    categories={
        "quiz": dict(label="Quizzes", weight=0.30, drop=1,
                     include=["QUIZ", "OPTIONAL EXTRA WORK"], exclude=["PRE-READING", "REMINDER"]),
        "hw": dict(label="Homework", weight=0.20, drop=1, include=["HOMEWORK"]),
        "midtermproject": dict(label="Midterm project", weight=0.125, include=["MIDTERM project"]),
        "midtermexam": dict(label="Midterm exam", weight=0.125, include=["MIDTERM exam"]),
        "finalproject": dict(label="Final project", weight=0.125, include=["FINAL PROJECT"]),
        "finalexam": dict(label="Final exam", weight=0.125, include=["FINAL EXAM"]),
    },
    email=dict(
        sender="matthew.bellis@gmail.com",
        cc=["mbellis@siena.edu"],
        reply_to="mbellis@siena.edu",
    ),
)
