"""PHYS 440: Quantum Physics, Spring 2026 (converted from the old notebook)"""

COURSE = dict(
    name="PHYS 440: Quantum Physics",
    gradebook="phys440s26.csv",
    categories={
        "hw": dict(
            label="Homework, quizzes, etc.", weight=0.10, drop=2,
            include=["QUIZ", "HOMEWORK", "DOCUMENTATION", "PRESENTATION", "EXTRA CREDIT"],
            exclude=["PRE-READING", "REMINDER", "REFLECTION"],
        ),
        "hwref": dict(label="Homework reflections", weight=0.10, drop=1, include=["REFLECTION"]),
        "midterm1": dict(label="Midterm exam #1", weight=0.25, include=["MIDTERM #1"]),
        "midterm2": dict(label="Midterm exam #2", weight=0.25, include=["MIDTERM #2"]),
        "final": dict(label="Final exam", weight=0.30, include=["FINAL EXAM"]),
    },
    email=dict(
        sender="matthew.bellis@gmail.com",
        cc=["mbellis@siena.edu"],
        reply_to="mbellis@siena.edu",
    ),
)
