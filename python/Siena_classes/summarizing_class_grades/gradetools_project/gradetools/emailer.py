"""
Write grade-summary emails to an outbox folder for review, then send them.

Nothing goes out until you call send_emails(..., really_send=True).

Workflow:
    outbox = build_outbox(res, gb)            # writes figures, .txt bodies, manifest.csv
    preview(outbox, "Lastname03")             # look at one
    send_emails(outbox, cfg, test_to="you@siena.edu", only=["Lastname03"], really_send=True)
    send_emails(outbox, cfg, really_send=True)  # everyone not already marked sent

The manifest records each successful send, so re-running never emails a student twice
(pass resend=True to override).

Password: set the environment variable GRADETOOLS_SMTP_PASSWORD, or you'll be prompted.
For Gmail this must be an App Password (Google Account > Security > App passwords).
"""

from __future__ import annotations

import datetime as dt
import getpass
import html
import os
import re
import smtplib
import ssl
import time
from email.message import EmailMessage
from email.utils import formatdate, make_msgid
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plots import student_figure

PASSWORD_ENV = "GRADETOOLS_SMTP_PASSWORD"


def _as_of(gb):
    m = re.match(r"(\d{4}-\d{2}-\d{2})", gb.path.name)
    d = dt.date.fromisoformat(m[1]) if m else dt.date.today()
    return d.strftime("%B %-d, %Y")


def _fmt_pts(v):
    return f"{v:g}" if v == round(v, 1) else f"{v:.2f}"


def compose(res, gb, sid) -> str:
    """Plain-text body for one student."""
    cfg, em = res.cfg, res.cfg["email"]
    s = res.students.loc[sid]
    fields = dict(first=s["first"], last=s["last"], course=cfg["name"])
    L = []
    if em["greeting"]:
        L += [em["greeting"].format(**fields), ""]
    if em["intro"]:
        L += [em["intro"].format(**fields), ""]
    L += [f"{cfg['name']}: {s['first']} {s['last']}, as of {_as_of(gb)}", ""]

    for key, c in cfg["categories"].items():
        cols = list(res.assignments.index[res.assignments["category"] == key])
        if not cols:
            continue
        head = c["label"]
        if c["drop"]:
            head += f"  (lowest {c['drop']} dropped)" if res.cfg["apply_drops"] else ""
        L += [head, "-" * len(head)]
        for col in cols:
            a = res.assignments.loc[col]
            pp = a["points_possible"]
            pct = res.pct.at[sid, col]
            if gb.excused.at[sid, col]:
                line, tag = f"{'EX':>7}  {'':>7} / {_fmt_pts(pp):<6}", "excused"
            elif pct != pct:
                line, tag = f"{'--':>7}  {'--':>7} / {_fmt_pts(pp):<6}", "not graded"
            else:
                got = gb.points.at[sid, col]
                got = 0.0 if got != got else got
                line, tag = f"{100*pct:6.1f}%  {_fmt_pts(got):>7} / {_fmt_pts(pp):<6}", ""
                if res.counted_zero.at[sid, col]:
                    tag = "missing, counted as 0"
                if cfg["apply_drops"] and res.dropped.at[sid, col]:
                    tag = (tag + ", " if tag else "") + "dropped"
            L.append(f"  {line}  {a['title']}" + (f"   [{tag}]" if tag else ""))
        L.append("")

    L += ["Summary", "-------"]
    w = max(len(c["label"]) for c in cfg["categories"].values())
    show_nd = em["show_no_drops"] and cfg["apply_drops"] and any(
        c["drop"] for c in cfg["categories"].values())
    hdr = f"  {'':<{w}}  weight   average" + ("   before drops" if show_nd else "")
    L.append(hdr)
    for key, c in cfg["categories"].items():
        v, vn = res.categories.at[sid, key], res.cat_nodrops.at[sid, key]
        if v != v:
            L.append(f"  {c['label']:<{w}}  {100*c['weight']:5.1f}%   (none yet)")
            continue
        extra = f"   {100*vn:6.2f}%" if show_nd and c["drop"] else ""
        L.append(f"  {c['label']:<{w}}  {100*c['weight']:5.1f}%   {100*v:6.2f}%{extra}")
    g = res.grade.at[sid]
    L += ["", f"  Course grade so far: {100*g:.2f}%"
          + (f"  ({res.letter().at[sid]})" if em["show_letter"] else "")]
    if show_nd:
        L.append(f"  Before dropping lowest scores: {100*res.grade_nodrops.at[sid]:.2f}%")
    missing = [c["label"] for k, c in cfg["categories"].items() if res.categories[k].isna().all()]
    if missing:
        L.append(f"  (Not yet included: {', '.join(missing)}. The weights of the "
                 f"categories above are scaled up to fill in for them.)")
    L.append("")
    if em["signoff"]:
        L += [em["signoff"].format(**fields), ""]
    if em["signature"]:
        L += [em["signature"]]
    return "\n".join(L).rstrip() + "\n"


def _tag(cfg):
    return re.sub(r"[^A-Za-z0-9]+", "_", cfg["name"]).strip("_")


def build_outbox(res, gb, outdir=None, dpi=150) -> Path:
    """Write one figure + body per student, and manifest.csv. Returns the outbox folder.
    Re-building keeps the 'sent' records of an existing manifest."""
    cfg = res.cfg
    out = Path(outdir or Path(cfg["output_dir"]) / _tag(cfg) / "outbox")
    out.mkdir(parents=True, exist_ok=True)
    old = _read_manifest(out)

    rows = []
    for sid, s in res.students.iterrows():
        stem = re.sub(r"[^A-Za-z0-9]+", "_", f"{s['last']}_{s['first']}")
        fig = student_figure(res, sid)
        fig_path = out / f"{stem}.png"
        fig.savefig(fig_path, dpi=dpi)
        plt.close(fig)
        body_path = out / f"{stem}.txt"
        body_path.write_text(compose(res, gb, sid), encoding="utf-8")
        subject = cfg["email"]["subject"].format(first=s["first"], last=s["last"], course=cfg["name"])
        prev = old.loc[sid] if old is not None and sid in old.index else None
        rows.append(dict(canvas_id=sid, last=s["last"], first=s["first"], email=s["email"],
                         grade=round(100 * res.grade.at[sid], 2), subject=subject,
                         body=body_path.name, figure=fig_path.name,
                         sent_at=prev["sent_at"] if prev is not None else ""))
    man = pd.DataFrame(rows).set_index("canvas_id")
    man.to_csv(out / "manifest.csv")
    n_sent = (man["sent_at"].fillna("") != "").sum()
    print(f"Wrote {len(man)} emails to {out}" + (f"  ({n_sent} already marked sent)" if n_sent else ""))
    return out


def _read_manifest(outbox):
    p = Path(outbox) / "manifest.csv"
    if not p.exists():
        return None
    return pd.read_csv(p, dtype={"canvas_id": str}, keep_default_na=False).set_index("canvas_id")


def _select(man, only):
    if not only:
        return man
    only = [only] if isinstance(only, str) else list(only)
    hit = pd.Series(False, index=man.index)
    for q in only:
        q = str(q).lower()
        hit |= (man.index.str.lower() == q) | man["last"].str.lower().str.contains(q) \
            | man["first"].str.lower().str.contains(q) | (man["email"].str.lower() == q)
    return man[hit]


def preview(outbox, who=None):
    """Show the email text and figure for one student (default: the first) in a notebook."""
    man = _read_manifest(outbox)
    row = _select(man, who).iloc[0] if who else man.iloc[0]
    print(f"To: {row['email']}\nSubject: {row['subject']}\n")
    print((Path(outbox) / row["body"]).read_text())
    try:
        from IPython.display import Image, display
        display(Image(filename=str(Path(outbox) / row["figure"])))
    except ImportError:
        print(f"[figure: {Path(outbox) / row['figure']}]")


def _message(cfg, row, outbox, to, test):
    em = cfg["email"]
    body = (Path(outbox) / row["body"]).read_text(encoding="utf-8")
    msg = EmailMessage()
    msg["Subject"] = ("[TEST] " if test else "") + row["subject"]
    msg["From"] = em["sender"]
    msg["To"] = to
    if em["cc"] and not test:
        msg["Cc"] = ", ".join(em["cc"])
    if em["reply_to"]:
        msg["Reply-To"] = em["reply_to"]
    msg["Date"] = formatdate(localtime=True)
    msg["Message-ID"] = make_msgid()
    msg.set_content(body)
    # HTML twin in a monospace font so the columns line up in Gmail/Outlook
    msg.add_alternative(
        '<pre style="font-family: Menlo, Consolas, \'Courier New\', monospace; font-size: 13px;">'
        + html.escape(body) + "</pre>", subtype="html")
    fig = Path(outbox) / row["figure"]
    msg.add_attachment(fig.read_bytes(), maintype="image", subtype="png", filename=fig.name)
    return msg


def send_emails(outbox, cfg, only=None, test_to=None, really_send=False, resend=False, pause=1.0):
    """Send the emails in an outbox.

    only        : a name, email, Canvas ID, or list of them (default: everyone)
    test_to     : send everything to this address instead (subject gets [TEST], no Cc,
                  and the manifest is not marked as sent)
    really_send : False (default) just lists what would be sent
    resend      : also send to students already marked as sent
    """
    em = cfg["email"]
    if not em["sender"]:
        raise ValueError("Set cfg['email']['sender'] first")
    man = _read_manifest(outbox)
    todo = _select(man, only)
    if not resend and not test_to:
        already = todo["sent_at"].fillna("") != ""
        if already.any():
            print(f"Skipping {already.sum()} already sent (resend=True to override)")
        todo = todo[~already]
    if todo.empty:
        print("Nothing to send.")
        return

    bad = todo[~todo["email"].str.contains("@") | todo["email"].str.endswith(".example")]
    if len(bad) and not test_to and not really_send:
        print(f"WARNING: {len(bad)} address(es) look invalid or anonymized "
              f"(e.g. {bad['email'].iloc[0]!r}); a real send would refuse.")
    elif len(bad) and not test_to:
        raise ValueError(f"{len(bad)} address(es) look invalid or anonymized, e.g. "
                         f"{bad['email'].iloc[0]!r}. Use test_to= for anonymized data.")

    print(("SENDING" if really_send else "DRY RUN (really_send=False)")
          + (f" -- all to {test_to}" if test_to else "") + f": {len(todo)} message(s)")
    for sid, row in todo.iterrows():
        to = test_to or row["email"]
        cc = "" if test_to or not em["cc"] else f"  cc {', '.join(em['cc'])}"
        print(f"  {row['first']} {row['last']:<20} -> {to}{cc}   ({row['grade']:.2f}%)")
    if not really_send:
        return

    password = os.environ.get(PASSWORD_ENV) or getpass.getpass(f"SMTP password for {em['sender']}: ")
    ctx = ssl.create_default_context()
    sent = 0
    with smtplib.SMTP_SSL(em["smtp_host"], em["smtp_port"], context=ctx) as server:
        server.login(em["sender"], password)
        for sid, row in todo.iterrows():
            to = test_to or row["email"]
            msg = _message(cfg, row, outbox, to, test=bool(test_to))
            server.send_message(msg)   # delivers to To and Cc
            sent += 1
            if not test_to:
                man.at[sid, "sent_at"] = dt.datetime.now().isoformat(timespec="seconds")
                man.to_csv(Path(outbox) / "manifest.csv")   # save after each, in case of a crash
            time.sleep(pause)
    print(f"Sent {sent} message(s).")
