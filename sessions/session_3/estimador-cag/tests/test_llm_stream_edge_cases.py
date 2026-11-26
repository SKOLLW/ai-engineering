"""Edge cases of the stream: refusals and cut-off answers. Network-free: scripted events."""

from types import SimpleNamespace

import pytest

from app.config import get_settings
from app.services import llm_service

HISTORY = [{"role": "user", "content": "Landing page with a blog."}]


def delta(text):
    return SimpleNamespace(type="response.output_text.delta", delta=text)


def refusal(text):
    return SimpleNamespace(type="response.refusal.delta", delta=text)


def incomplete(reason="max_output_tokens", details=True):
    response = SimpleNamespace(
        model="gpt-test-2026",
        usage=SimpleNamespace(input_tokens=900, output_tokens=16000),
        incomplete_details=SimpleNamespace(reason=reason) if details else None,
    )
    return SimpleNamespace(type="response.incomplete", response=response)


def completed():
    response = SimpleNamespace(
        model="gpt-test-2026", usage=SimpleNamespace(input_tokens=11, output_tokens=7)
    )
    return SimpleNamespace(type="response.completed", response=response)


@pytest.fixture(autouse=True)
def settings_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-key-for-tests")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def use_events(monkeypatch, events):
    client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs: iter(events)))
    monkeypatch.setattr(llm_service, "get_sync_client", lambda: client)


def test_refusal_deltas_are_yielded_like_text(monkeypatch):
    use_events(monkeypatch, [refusal("I can't "), refusal("help with that."), completed()])
    assert list(llm_service.stream_estimation(HISTORY)) == ["I can't ", "help with that."]


def test_incomplete_event_reads_the_metrics_and_the_reason(monkeypatch):
    use_events(monkeypatch, [delta("## Estimate: half"), incomplete("max_output_tokens")])
    stream = llm_service.stream_estimation(HISTORY)
    assert list(stream) == ["## Estimate: half"]
    assert stream.model == "gpt-test-2026"
    assert stream.input_tokens == 900
    assert stream.output_tokens == 16000
    assert stream.incomplete_reason == "max_output_tokens"


def test_incomplete_details_none_gives_no_reason_and_no_exception(monkeypatch):
    use_events(monkeypatch, [delta("Hi"), incomplete(details=False)])
    stream = llm_service.stream_estimation(HISTORY)
    assert list(stream) == ["Hi"]
    assert stream.incomplete_reason is None
    assert stream.model == "gpt-test-2026"


def test_incomplete_reason_is_none_by_default_and_on_normal_completion(monkeypatch):
    use_events(monkeypatch, [delta("Hi"), completed()])
    stream = llm_service.stream_estimation(HISTORY)
    assert stream.incomplete_reason is None
    list(stream)
    assert stream.incomplete_reason is None
