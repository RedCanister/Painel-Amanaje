import asyncio

import pytest

from app.database import db_session


def test_wait_for_database_retries_until_connection_succeeds(monkeypatch):
    class SuccessfulConnection:
        def __init__(self):
            self.statements = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        async def execute(self, statement):
            self.statements.append(str(statement))

    class FakeEngine:
        def __init__(self):
            self.calls = 0
            self.connection = SuccessfulConnection()

        def connect(self):
            self.calls += 1
            if self.calls == 1:
                raise ConnectionRefusedError("database is not ready")
            return self.connection

    fake_engine = FakeEngine()
    monkeypatch.setattr(db_session, "engine", fake_engine)

    asyncio.run(db_session.wait_for_database(retries=2, delay_seconds=0))

    assert fake_engine.calls == 2
    assert fake_engine.connection.statements == ["SELECT 1"]


def test_wait_for_database_raises_after_retry_budget(monkeypatch):
    class FakeEngine:
        def __init__(self):
            self.calls = 0

        def connect(self):
            self.calls += 1
            raise ConnectionRefusedError("database is not ready")

    fake_engine = FakeEngine()
    monkeypatch.setattr(db_session, "engine", fake_engine)

    with pytest.raises(ConnectionRefusedError):
        asyncio.run(db_session.wait_for_database(retries=2, delay_seconds=0))

    assert fake_engine.calls == 2
