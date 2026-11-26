"""Tests for the streaming entry point. Network-free: a fake client replays scripted events."""

from types import SimpleNamespace

import pytest

from app.config import get_settings
from app.services import llm_service


def delta(text):
    return SimpleNamespace(type="response.output_text.delta", delta=text)


def completed(model="gpt-test-2026", input_tokens=11, output_tokens=7, usage=True):
    usage_obj = (
        SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens) if usage else None
    )
    response = SimpleNamespace(model=model, usage=usage_obj)
    return SimpleNamespace(type="response.completed", response=response)


class FakeClient:
    """Mimics OpenAI().responses.create(..., stream=True): records the call, replays events."""

    def __init__(self, events):
        self.events = events
        self.calls = []
        self.responses = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return iter(self.events)


@pytest.fixture(autouse=True)
def settings_env(monkeypatch):
    # The settings require a key; a dummy value is enough because no request leaves the process.
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-key-for-tests")
    monkeypatch.setenv("LLM_MODEL", "configured-model")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def use_fake(monkeypatch, events):
    fake = FakeClient(events)
    monkeypatch.setattr(llm_service, "get_sync_client", lambda: fake)
    return fake


HISTORY = [{"role": "user", "content": "Landing page with a blog."}]


def test_yields_the_text_deltas_in_order(monkeypatch):
    use_fake(monkeypatch, [delta("Hel"), delta("lo"), completed()])
    stream = llm_service.stream_estimation(HISTORY)
    assert list(stream) == ["Hel", "lo"]


def test_ignores_events_that_are_not_text_deltas(monkeypatch):
    other = SimpleNamespace(type="response.created")
    use_fake(monkeypatch, [other, delta("Hi"), SimpleNamespace(type="response.output_text.done"), completed()])
    assert list(llm_service.stream_estimation(HISTORY)) == ["Hi"]


def test_reads_metrics_from_the_completed_response(monkeypatch):
    use_fake(monkeypatch, [delta("Hi"), completed(model="gpt-4o-mini-2024", input_tokens=123, output_tokens=45)])
    stream = llm_service.stream_estimation(HISTORY)
    list(stream)
    assert stream.model == "gpt-4o-mini-2024"
    assert stream.provider == "openai"
    assert stream.input_tokens == 123
    assert stream.output_tokens == 45


def test_sends_system_prompt_then_the_whole_history_with_stream_on(monkeypatch):
    fake = use_fake(monkeypatch, [delta("ok"), completed()])
    history = [
        {"role": "user", "content": "Landing page with a blog."},
        {"role": "assistant", "content": "## Estimate: ..."},
        {"role": "user", "content": "Make it cheaper."},
    ]
    list(llm_service.stream_estimation(history))

    (call,) = fake.calls
    assert call["stream"] is True
    assert call["model"] == "configured-model"
    assert call["input"] == [{"role": "system", "content": llm_service.build_system_prompt()}, *history]


def test_does_not_change_the_history_it_receives(monkeypatch):
    use_fake(monkeypatch, [delta("ok"), completed()])
    history = list(HISTORY)
    list(llm_service.stream_estimation(history))
    assert history == HISTORY


def test_no_request_is_sent_until_the_stream_is_consumed(monkeypatch):
    fake = use_fake(monkeypatch, [delta("ok"), completed()])
    llm_service.stream_estimation(HISTORY)
    assert fake.calls == []


def test_metrics_are_none_when_the_completed_event_has_no_usage(monkeypatch):
    use_fake(monkeypatch, [delta("Hi"), completed(usage=False)])
    stream = llm_service.stream_estimation(HISTORY)
    assert list(stream) == ["Hi"]
    assert stream.model == "gpt-test-2026"
    assert stream.input_tokens is None
    assert stream.output_tokens is None


def test_metrics_are_none_when_the_stream_ends_without_a_completed_event(monkeypatch):
    use_fake(monkeypatch, [delta("Hi")])
    stream = llm_service.stream_estimation(HISTORY)
    assert list(stream) == ["Hi"]
    assert stream.model is None
    assert stream.input_tokens is None


def test_an_error_event_raises_after_the_deltas_already_sent(monkeypatch):
    error = SimpleNamespace(type="error", message="rate limit")
    use_fake(monkeypatch, [delta("Hi"), error])
    items = iter(llm_service.stream_estimation(HISTORY))
    assert next(items) == "Hi"
    with pytest.raises(llm_service.LLMStreamError, match="rate limit"):
        next(items)


def test_a_failed_response_raises(monkeypatch):
    failed = SimpleNamespace(
        type="response.failed",
        response=SimpleNamespace(error=SimpleNamespace(message="server down")),
    )
    use_fake(monkeypatch, [failed])
    with pytest.raises(llm_service.LLMStreamError, match="server down"):
        list(llm_service.stream_estimation(HISTORY))


def test_get_sync_client_is_cached_and_uses_the_settings_key():
    llm_service.get_sync_client.cache_clear()
    first = llm_service.get_sync_client()
    assert first is llm_service.get_sync_client()
    assert first.api_key == "dummy-key-for-tests"
    llm_service.get_sync_client.cache_clear()
