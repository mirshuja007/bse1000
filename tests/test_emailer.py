"""Offline unit tests for src/emailer.py - smtplib is mocked, no real
email is ever sent by the test suite."""
import pytest

from src import emailer


@pytest.fixture(autouse=True)
def configured_env(monkeypatch):
    monkeypatch.setenv("EMAIL_SENDER", "sender@gmail.com")
    monkeypatch.setenv("EMAIL_APP_PASSWORD", "fake-app-password")


def test_is_configured_requires_both_env_vars(monkeypatch):
    assert emailer.is_configured() is True
    monkeypatch.delenv("EMAIL_APP_PASSWORD", raising=False)
    assert emailer.is_configured() is False


def test_send_report_email_raises_when_not_configured(monkeypatch):
    monkeypatch.delenv("EMAIL_SENDER", raising=False)
    with pytest.raises(RuntimeError, match="not configured"):
        emailer.send_report_email("a@b.com", "subject", "body", {})


def test_send_report_email_rejects_obviously_invalid_recipient():
    with pytest.raises(ValueError, match="valid email"):
        emailer.send_report_email("not-an-email", "subject", "body", {})


class FakeSMTP:
    instances = []

    def __init__(self, host, port, timeout=None):
        self.host = host
        self.port = port
        self.starttls_called = False
        self.login_args = None
        self.sent = None
        FakeSMTP.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def starttls(self):
        self.starttls_called = True

    def login(self, sender, password):
        self.login_args = (sender, password)

    def sendmail(self, sender, recipients, message):
        self.sent = (sender, recipients, message)


def test_send_report_email_sends_via_gmail_smtp_with_attachments(monkeypatch):
    FakeSMTP.instances.clear()
    monkeypatch.setattr(emailer.smtplib, "SMTP", FakeSMTP)

    emailer.send_report_email(
        "recipient@example.com",
        "Scan results",
        "See attached.",
        {"scan_results.csv": "a,b\n1,2\n", "tracked_picks.csv": "x,y\n3,4\n"},
    )

    smtp = FakeSMTP.instances[0]
    assert smtp.host == emailer.SMTP_HOST
    assert smtp.port == emailer.SMTP_PORT
    assert smtp.starttls_called is True
    assert smtp.login_args == ("sender@gmail.com", "fake-app-password")

    sender, recipients, message = smtp.sent
    assert sender == "sender@gmail.com"
    assert recipients == ["recipient@example.com"]
    assert "scan_results.csv" in message
    assert "tracked_picks.csv" in message
    assert "Scan results" in message
