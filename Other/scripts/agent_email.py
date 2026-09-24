#!/usr/bin/env python3
"""Send progress mail to the user and read replies, using the account in email.txt.

Usage:
  python scripts/agent_email.py send "subject" "body text"
  python scripts/agent_email.py send "subject" --body-file path
  python scripts/agent_email.py inbox [--unseen] [--limit N]
"""

from __future__ import annotations

import argparse
import email
import email.header
import imaplib
import smtplib
import ssl
from email.message import EmailMessage
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
USER_ADDRESS = "yodharinmonplub@actuarialtutor.org"  # per AGENTS.md (changed 2026-09-20; was the gmail address)
SMTP_HOST, SMTP_PORT = "smtp.purelymail.com", 465
IMAP_HOST, IMAP_PORT = "imap.purelymail.com", 993


def credentials() -> tuple[str, str]:
    values = {}
    for line in (ROOT / "email.txt").read_text(encoding="utf-8").splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            values[key.strip().lower()] = value.strip()
    return values["username"], values["password"]


def send(subject: str, body: str) -> None:
    user, password = credentials()
    msg = EmailMessage()
    msg["From"] = user
    msg["To"] = USER_ADDRESS
    msg["Subject"] = subject
    msg.set_content(body)
    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=ssl.create_default_context(), timeout=60) as smtp:
        smtp.login(user, password)
        smtp.send_message(msg)
    print(f"sent: {subject}")


def decode(value: str | None) -> str:
    if not value:
        return ""
    parts = email.header.decode_header(value)
    return "".join(p.decode(enc or "utf-8", "replace") if isinstance(p, bytes) else p for p, enc in parts)


def inbox(unseen: bool, limit: int) -> None:
    user, password = credentials()
    with imaplib.IMAP4_SSL(IMAP_HOST, IMAP_PORT, ssl_context=ssl.create_default_context(), timeout=30) as imap:
        imap.login(user, password)
        imap.select("INBOX")
        _, data = imap.search(None, "UNSEEN" if unseen else "ALL")
        ids = data[0].split()[-limit:]
        for uid in reversed(ids):
            _, raw = imap.fetch(uid, "(RFC822)")
            msg = email.message_from_bytes(raw[0][1])
            body = ""
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    body = part.get_payload(decode=True).decode(part.get_content_charset() or "utf-8", "replace")
                    break
            print("=" * 70)
            print(f"From: {decode(msg['From'])}\nDate: {msg['Date']}\nSubject: {decode(msg['Subject'])}\n")
            print(body.strip()[:4000])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("send")
    s.add_argument("subject")
    s.add_argument("body", nargs="?")
    s.add_argument("--body-file", type=Path)
    i = sub.add_parser("inbox")
    i.add_argument("--unseen", action="store_true")
    i.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    if args.cmd == "send":
        body = args.body_file.read_text(encoding="utf-8") if args.body_file else (args.body or "")
        send(args.subject, body)
    else:
        inbox(args.unseen, args.limit)


if __name__ == "__main__":
    main()
