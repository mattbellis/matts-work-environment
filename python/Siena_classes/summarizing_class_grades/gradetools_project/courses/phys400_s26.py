"""PHYS 400: Nuclear and Particle Physics, Spring 2026 (converted from the old notebook)"""

COURSE = dict(
    name="PHYS 400: Nuclear and Particle Physics",
    gradebook="phys400s26.csv",
    categories={
        "hwandquiz": dict(
            label="Homework, quizzes, etc.", weight=0.20, drop=3,
            include=["QUIZ", "HOMEWORK", "DOCUMENTATION", "PRESENTATION", "EXTRA CREDIT"],
            exclude=["PRE-READING", "REMINDER", "REFLECTION"],
        ),
        "hwref": dict(label="Homework reflections", weight=0.10, drop=1, include=["REFLECTION"]),
        "paper": dict(label="Paper", weight=0.15, include=["JOURNAL ARTICLE"]),
        "midtermexam": dict(label="Midterm exam", weight=0.25, include=["MIDTERM exam"]),
        "finalexam": dict(label="Final exam", weight=0.30, include=["FINAL EXAM"]),
    },
    email=dict(
        sender="matthew.bellis@gmail.com",
        cc=["mbellis@siena.edu"],
        reply_to="mbellis@siena.edu",
    ),
)
