"""Chat page edge cases: cut-off answers and empty answers. Network-free: fake streams."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app.config import get_settings
from app.services import llm_service

APP_FILE = str(Path(__file__).resolve().parent.parent / "streamlit_app.py")


class FakeStream:
    """Same contract as EstimationStream: iterate for str deltas, then read the metrics."""

    def __init__(self, deltas, incomplete_reason=None, model="gpt-test-2026"):
        self.deltas = deltas
        self.model = model
        self.input_tokens = 900
        self.output_tokens = 16000
        self.incomplete_reason = incomplete_reason

    def __iter__(self):
        yield from self.deltas


class FakeService:
    def __init__(self, streams):
        self.streams = list(streams)

    def __call__(self, history):
        return self.streams.pop(0)


@pytest.fixture(autouse=True)
def settings_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-key-for-tests")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def use_fake(monkeypatch, *streams):
    monkeypatch.setattr(llm_service, "stream_estimation", FakeService(streams))


def start():
    return AppTest.from_file(APP_FILE, default_timeout=30).run()


def send(app, text):
    return app.chat_input[0].set_value(text).run()


def metrics(app):
    return {m.label: m.value for m in app.sidebar.metric}


def test_cut_off_answer_is_saved_with_a_warning_and_real_metrics(monkeypatch):
    use_fake(monkeypatch, FakeStream(["## Estimate: half"], incomplete_reason="max_output_tokens"))
    app = send(start(), "Landing page with a blog.")
    assert not app.exception
    assert app.session_state["messages"] == [
        {"role": "user", "content": "Landing page with a blog."},
        {"role": "assistant", "content": "## Estimate: half"},
    ]
    assert len(app.warning) == 1
    assert "cut off" in app.warning[0].value
    assert "incomplete" in app.warning[0].value
    assert "max_output_tokens" in app.warning[0].value
    assert metrics(app)["Input tokens"] == "900"
    assert metrics(app)["Output tokens"] == "16000"
    assert not app.error


def test_the_cut_off_warning_shows_on_that_run_only(monkeypatch):
    use_fake(
        monkeypatch,
        FakeStream(["Half"], incomplete_reason="max_output_tokens"),
        FakeStream(["Full answer."]),
    )
    app = send(start(), "Landing page with a blog.")
    assert len(app.warning) == 1
    app = send(app, "Make it cheaper.")
    assert len(app.warning) == 0
    # The warning is not part of the saved history.
    assert all("cut off" not in m["content"] for m in app.session_state["messages"])


def test_empty_answer_shows_an_error_and_saves_nothing(monkeypatch):
    use_fake(monkeypatch, FakeStream([]))
    app = send(start(), "Landing page with a blog.")
    assert not app.exception
    assert len(app.error) == 1
    assert app.session_state["messages"] == []
    assert "last_call" not in app.session_state
    assert not app.warning


def test_empty_answer_keeps_the_last_call_unchanged(monkeypatch):
    use_fake(monkeypatch, FakeStream(["First."], model="first-model"), FakeStream([]))
    app = send(start(), "Landing page with a blog.")
    before = dict(app.session_state["last_call"])
    app = send(app, "Make it cheaper.")
    assert len(app.error) == 1
    assert app.session_state["last_call"] == before
    assert app.session_state["messages"] == [
        {"role": "user", "content": "Landing page with a blog."},
        {"role": "assistant", "content": "First."},
    ]
