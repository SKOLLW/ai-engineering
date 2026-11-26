"""Tests for the context sidebar. Network-free: stream_estimation is replaced by a fake."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app.config import get_settings
from app.context.examples import ESTIMATION_EXAMPLES, HOURLY_RATE_USD
from app.services import llm_service

APP_FILE = str(Path(__file__).resolve().parent.parent / "streamlit_app.py")


class FakeStream:
    """Same contract as EstimationStream: iterate for str deltas, then read the metrics."""

    def __init__(self, deltas, model=None, input_tokens=None, output_tokens=None, error=None):
        self.deltas = deltas
        self.model = model
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.error = error

    def __iter__(self):
        yield from self.deltas
        if self.error:
            raise self.error


@pytest.fixture(autouse=True)
def settings_env(monkeypatch):
    # A dummy key beats the real .env, and no request can leave the process: the stream is fake.
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-key-for-tests")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def use_fake(monkeypatch, *streams):
    queue = list(streams)
    monkeypatch.setattr(llm_service, "stream_estimation", lambda history: queue.pop(0))


def start():
    return AppTest.from_file(APP_FILE, default_timeout=30).run()


def send(app, text="Landing page with a blog."):
    return app.chat_input[0].set_value(text).run()


def sidebar_text(app):
    """All the text the sidebar shows: code, markdown, captions and metric label + value."""
    parts = [e.value for e in app.sidebar.code]
    parts += [e.value for e in app.sidebar.markdown]
    parts += [e.value for e in app.sidebar.caption]
    parts += [f"{m.label}: {m.value}" for m in app.sidebar.metric]
    return "\n".join(parts)


def metrics(app):
    return {m.label: m.value for m in app.sidebar.metric}


def test_sidebar_shows_the_prompt_the_examples_and_no_call_yet():
    app = start()
    assert not app.exception
    text = sidebar_text(app)
    assert llm_service.build_system_prompt() in [e.value for e in app.sidebar.code]
    for example in ESTIMATION_EXAMPLES:
        assert example["meeting_summary"] in text
        assert example["estimation"].strip() in text
    assert f"{HOURLY_RATE_USD} USD/h" in text
    assert "No call yet" in text
    assert not app.sidebar.metric


def test_sidebar_shows_the_metrics_of_the_call_just_made(monkeypatch):
    use_fake(monkeypatch, FakeStream(["Total"], "fake-model", 120, 45))
    app = send(start())
    assert not app.exception
    shown = metrics(app)
    assert shown["Model"] == "fake-model"
    assert shown["Input tokens"] == "120"
    assert shown["Output tokens"] == "45"
    assert shown["Response time"].endswith(" s")
    assert "No call yet" not in sidebar_text(app)


def test_missing_token_counts_show_n_a_not_zero(monkeypatch):
    use_fake(monkeypatch, FakeStream(["Total"], None, None, None))
    app = send(start())
    assert not app.exception
    shown = metrics(app)
    assert shown["Model"] == "n/a"
    assert shown["Input tokens"] == "n/a"
    assert shown["Output tokens"] == "n/a"
    assert shown["Response time"].endswith(" s")  # the app measures this one itself


def test_a_failed_call_keeps_the_last_good_metrics(monkeypatch):
    broken = FakeStream(["Partial"], "bad-model", 1, 1, error=llm_service.LLMStreamError("boom"))
    use_fake(monkeypatch, FakeStream(["Total"], "fake-model", 120, 45), broken)
    app = send(start())
    app = send(app, "Make it cheaper.")
    assert len(app.error) == 1
    shown = metrics(app)
    assert shown["Model"] == "fake-model"
    assert shown["Input tokens"] == "120"
    assert shown["Output tokens"] == "45"


def test_a_failed_first_call_still_says_no_call_yet(monkeypatch):
    broken = FakeStream([], error=llm_service.LLMStreamError("boom"))
    use_fake(monkeypatch, broken)
    app = send(start())
    assert len(app.error) == 1
    assert "No call yet" in sidebar_text(app)
    assert not app.sidebar.metric
