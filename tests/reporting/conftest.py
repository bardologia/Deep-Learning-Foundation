"""Shared fixtures of the reporting tests: the SMTP stand-in the mailer tests send through."""

from __future__ import annotations

import pytest


class FakeSmtp:
    """Stand-in for smtplib.SMTP_SSL recording logins and sent messages.

    Attributes:
        instances : Every constructed fake, in creation order.
    """

    instances = []

    def __init__(self, host, port, timeout):
        """Records the endpoint and the timeout and registers the instance."""
        self.host     = host
        self.port     = port
        self.timeout  = timeout
        self.logins   = []
        self.messages = []
        FakeSmtp.instances.append(self)

    def __enter__(self):
        """Returns the fake as its own context."""
        return self

    def __exit__(self, *exc_info):
        """Leaves the context without suppressing exceptions."""
        return False

    def login(self, user, password):
        """Records one login attempt."""
        self.logins.append((user, password))

    def send_message(self, message):
        """Records one submitted message."""
        self.messages.append(message)


@pytest.fixture
def fake_smtp(monkeypatch):
    """Replaces SMTP_SSL in the mailer module with the recording fake."""
    FakeSmtp.instances = []
    monkeypatch.setattr("tools.reporting.report_mailer.smtplib.SMTP_SSL", FakeSmtp)
    return FakeSmtp
