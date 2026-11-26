"""Tests for the "New chat" button. Network-free: stream_estimation is replaced by a fake."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app.config import get_settings
from app.services import llm_service

APP_FILE = str(Path(__file__).resolve().parent.parent / "streamlit_app.py")


class FakeStream:
    """Same contract as EstimationStream: iterate for str deltas, then read the metrics."""

    def __init__(self, deltas, model="fake-model", input_tokens=120, output_tokens=45):
        self.deltas = deltas
        self.model = model
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens

    def __iter__(self):
        yield from self.deltas


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
    # A dummy key beats the real .env, and no request can leave the process: the stream is fake.
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-key-for-tests")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def use_fake(monkeypatch, *streams):
    service = FakeService(streams)
    monkeypatch.setattr(llm_service, "stream_estimation", service)
    return service


def start():
    return AppTest.from_file(APP_FILE, default_timeout=30).run()


def send(app, text):
    return app.chat_input[0].set_value(text).run()


def new_chat_button(app):
    return [b for b in app.sidebar.button if b.label == "New chat"]


def sidebar_captions(app):
    return [c.value for c in app.sidebar.caption]


def test_the_sidebar_has_a_new_chat_button():
    app = start()
    assert not app.exception
    assert len(new_chat_button(app)) == 1


def test_new_chat_clears_the_messages_and_the_last_call(monkeypatch):
    use_fake(monkeypatch, FakeStream(["Total: 40 hours"]))
    app = send(start(), "Landing page with a blog.")
    assert len(app.chat_message) == 2
    assert not any("No call yet" in c for c in sidebar_captions(app))

    app = new_chat_button(app)[0].click().run()

    assert not app.exception
    assert len(app.chat_message) == 0
    assert any("No call yet" in c for c in sidebar_captions(app))
    assert not app.sidebar.metric


def test_a_message_after_new_chat_is_sent_without_the_old_history(monkeypatch):
    service = use_fake(monkeypatch, FakeStream(["First estimate"]), FakeStream(["Second estimate"]))
    app = send(start(), "First transcript.")
    app = new_chat_button(app)[0].click().run()
    app = send(app, "Second transcript.")

    assert not app.exception
    assert service.histories[-1] == [{"role": "user", "content": "Second transcript."}]
    assert [m.markdown[0].value for m in app.chat_message] == [
        "Second transcript.",
        "Second estimate",
    ]
