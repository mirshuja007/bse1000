"""Email the current scan results, tracked picks, and recommendation
history as CSV attachments, on demand, via Gmail SMTP.

Setup (one-time, per your choice of Gmail SMTP + App Password):
1. Enable 2-Step Verification on the Gmail account you want to send from
   (required before Google will issue an App Password).
2. Generate an App Password at https://myaccount.google.com/apppasswords
   (a 16-character code, unrelated to your normal Gmail password).
3. Set in .env locally, or Streamlit Cloud's Secrets for the deployed app:
     EMAIL_SENDER=youraddress@gmail.com
     EMAIL_APP_PASSWORD=<the 16-character app password, no spaces>

Uses only the standard library (smtplib/email) - no new dependency.
"""
from __future__ import annotations

import os
import smtplib
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587


def is_configured() -> bool:
    return bool(os.environ.get("EMAIL_SENDER") and os.environ.get("EMAIL_APP_PASSWORD"))


def send_report_email(recipient: str, subject: str, body: str, attachments: dict[str, str]) -> None:
    """attachments: {filename: csv_text_content}. Raises on any failure -
    misconfiguration, an invalid recipient, or an SMTP error - rather than
    swallowing it, so a failed send is never mistaken for a sent one."""
    sender = os.environ.get("EMAIL_SENDER")
    app_password = os.environ.get("EMAIL_APP_PASSWORD")
    if not sender or not app_password:
        raise RuntimeError("Email not configured - set EMAIL_SENDER and EMAIL_APP_PASSWORD")
    if not recipient or "@" not in recipient:
        raise ValueError(f"{recipient!r} doesn't look like a valid email address")

    msg = MIMEMultipart()
    msg["From"] = sender
    msg["To"] = recipient
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    for filename, content in attachments.items():
        part = MIMEApplication(content.encode("utf-8"), Name=filename)
        part["Content-Disposition"] = f'attachment; filename="{filename}"'
        msg.attach(part)

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20) as server:
        server.starttls()
        server.login(sender, app_password)
        server.sendmail(sender, [recipient], msg.as_string())
