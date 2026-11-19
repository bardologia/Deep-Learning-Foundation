
from __future__ import annotations

import pytest


class FakeSmtp:

    instances = []

    def __init__(self, host, port, timeout):
        self.host     = host
        self.port     = port
        self.timeout  = timeout
        self.logins   = []
        self.messages = []
        FakeSmtp.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def login(self, user, password):
        self.logins.append((user, password))

    def send_message(self, message):
        self.messages.append(message)


@pytest.fixture
def fake_smtp(monkeypatch):
    FakeSmtp.instances = []
    monkeypatch.setattr("tools.reporting.report_mailer.smtplib.SMTP_SSL", FakeSmtp)
    return FakeSmtp
