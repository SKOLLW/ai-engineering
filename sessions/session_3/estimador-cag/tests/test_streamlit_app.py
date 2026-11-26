"""Tests for the Streamlit chat page. Network-free: stream_estimation is replaced by a fake."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app.config import get_settings
from app.services import llm_service

APP_FILE = str(Path(__file__).resolve().parent.parent / "streamlit_app.py")


class FakeStream:
    """Same contract as EstimationStream: iterate to get str deltas."""

    def __init__(self, deltas, error=None):
        self.deltas = deltas
        self.error = error

    def __iter__(self):
        yield from self.deltas
        if self.error:
            raise self.error


class FakeService:
    """Replaces stream_estimation: records the history of each call, replays scripted streams."""

    def __init__(self, streams):
        self.streams = list(streams)
        self.histories = []

    def __call__(self, history):
        self.histories.append([dict(message) for message in history])  # a copy, as the real one
        return self.streams.pop(0)


@pytest.fixture(autouse=True)
def settings_env(monkeypatch):
    # A dummy key beats the real .env (environment variables win), and no request can leave
    # the process anyway because every test replaces stream_estimation.
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-key-for-tests")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def use_fake(monkeypatch, *streams):
    service = FakeService(streams)
    monkeypatch.setattr(llm_service, "stream_estimation", service)
    return service


def start():
    # A generous timeout: the first run imports streamlit and openai, which can take a few seconds.
    return AppTest.from_file(APP_FILE, default_timeout=30).run()


def send(app, text):
    return app.chat_input[0].set_value(text).run()


def shown(app):
    """(role, text) of every chat message on screen, in order."""
    return [(m.name, m.markdown[0].value) for m in app.chat_message]


def test_starts_with_a_chat_input_and_no_messages():
    app = start()
    assert not app.exception
    assert len(app.chat_input) == 1
    assert len(app.chat_message) == 0


def test_shows_the_user_message_and_the_streamed_answer(monkeypatch):
    use_fake(monkeypatch, FakeStream(["Total: ", "40 hours"]))
    app = send(start(), "Landing page with a blog.")
    assert not app.exception
    assert shown(app) == [("user", "Landing page with a blog."), ("assistant", "Total: 40 hours")]


def test_second_message_keeps_the_history_and_sends_it_all(monkeypatch):
    service = use_fake(monkeypatch, FakeStream(["First ", "answer"]), FakeStream(["Cheaper."]))
    app = send(start(), "Landing page with a blog.")
    app = send(app, "Make it cheaper.")
    assert shown(app) == [
        ("user", "Landing page with a blog."),
        ("assistant", "First answer"),
        ("user", "Make it cheaper."),
        ("assistant", "Cheaper."),
    ]
    assert service.histories[1] == [
        {"role": "user", "content": "Landing page with a blog."},
        {"role": "assistant", "content": "First answer"},
        {"role": "user", "content": "Make it cheaper."},
    ]


def test_stream_error_shows_an_error_and_adds_no_assistant_message(monkeypatch):
    error = llm_service.LLMStreamError("secret detail")
    service = use_fake(monkeypatch, FakeStream(["Partial"], error=error), FakeStream(["Done."]))
    app = send(start(), "Landing page with a blog.")
    assert not app.exception
    assert len(app.error) == 1
    assert "secret detail" not in app.error[0].value
    assert app.session_state["messages"] == []
    # Sending again: the history holds the user message once and no broken assistant message.
    app = send(app, "Landing page with a blog.")
    assert service.histories[1] == [{"role": "user", "content": "Landing page with a blog."}]
    assert shown(app) == [("user", "Landing page with a blog."), ("assistant", "Done.")]


def test_missing_key_shows_an_error_without_details(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")  # the settings reject an empty key
    app = send(start(), "Landing page with a blog.")
    assert not app.exception
    assert len(app.error) == 1
    assert "OPENAI_API_KEY" not in app.error[0].value
    assert app.session_state["messages"] == []
