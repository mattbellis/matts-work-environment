# gradetools

Canvas gradebook export → category averages → class plots → per-student emails.

```
gradetools/          the code (you shouldn't need to edit this per course)
  canvas.py          read the Canvas CSV (handles Points Possible row, test student, EX, odd cells)
  config.py          defaults + checks for the course dict (catches typos like drops= vs drop=)
  classify.py        keyword rules -> category; decides which assignments are graded
  diagnostics.py     the pre-flight report
  scoring.py         category averages, drops, course grade, running average, Canvas cross-check
  plots.py           class overview + per-student figure
  emailer.py         outbox (figures, bodies, manifest), preview, send
courses/             one dict per course-term; TEMPLATE.py documents every option
grades.ipynb         the notebook you run
anonymize_canvas.py  scrub a gradebook before sharing it
```

Requirements: pandas, numpy, matplotlib (seaborn is no longer needed).

## Each time you run it

1. Export the gradebook from Canvas into this folder; set `gradebook=` in the course file.
2. Open `grades.ipynb`, change the `from courses.... import COURSE` line, run all.
3. Read the diagnostics. Fix UNCLASSIFIED / AMBIGUOUS / partly-graded items in the course
   file (a keyword, an `exclude` word, or an `overrides` entry), and re-run.
4. Check `canvas_check`: if you set `canvas_group` on categories it should say `ok`.
5. Look at the plots and `res.summary()`.
6. Build the outbox, `preview()` a few students, dry run, send one to yourself with
   `test_to=`, then send for real.

## Categorizing assignments

Your include/exclude keyword scheme is kept as is. It's a good fit for the ALL-CAPS
prefixes you already use in Canvas titles. Four changes:

- **Ambiguity is flagged.** Before, a title matching two categories went silently to
  whichever was checked first.
- **Unmatched titles are reported**, so a typo in a title doesn't silently drop an assignment.
- **`overrides`** handles one-off exceptions by Canvas ID, so you don't have to invent
  keywords for them.
- **`ignore`** is a global "never count" list (OPTIONAL, REMINDER, ...).

Only the assignment title is matched. Canvas's own "... Current Score" columns are
recognized and set aside automatically, so the old `exclude=['Current Score']` entries
are no longer needed.

## What changed from the old notebook, and why

- **Ungraded assignments.** The old notebook ran `df.fillna(0)`, so an assignment nobody
  had been graded on yet counted as a 0 for everyone. That's why you had to keep the
  hand-maintained `hnames_to_drop` list. Now an assignment counts only once at least
  `graded_threshold` (default 50%) of students have a score. After that, an individual
  blank counts as 0, and every such blank is listed. `EX` never counts.
- **Cc copies were never delivered.** The old code set a `Cc:` header, but `sendmail()`
  was only given the student's address, so the copy to mbellis@siena.edu was never
  actually sent. You'd only have seen these in Gmail's Sent folder. `send_message()` now
  delivers to both To and Cc.
- **Attachments.** Messages were built as `multipart/alternative` with the PNG inside,
  and some mail clients hide attachments in that structure. They're now
  `multipart/mixed`. Each email also carries an HTML copy in a monospace font, so the
  score columns line up in Gmail, which is what the commented-out Courier attempt was for.
- **Dates.** The fake dates (`day_org + 20 s × Canvas ID`) were only used for ordering.
  Assignments are now ordered by Canvas ID directly, and the emails don't show dates.
- **No double sends.** `manifest.csv` records each successful send. Re-running skips
  those students unless you pass `resend=True`. A real send refuses anonymized or
  invalid addresses.
- **Password.** Set the environment variable `GRADETOOLS_SMTP_PASSWORD`, or you'll be
  prompted. It's never stored in the notebook.

Grading math that is unchanged: the default `method="mean"` gives each assignment equal
weight within a category, drops remove the lowest percentages, and categories with
nothing graded yet are left out with the remaining weights rescaled.
`method="points"` (total points, as Canvas computes it) is available per category.
