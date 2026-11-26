"""Streamlit chat page: paste a meeting transcript, read the streamed estimate."""

import time

import openai
import streamlit as st
from pydantic import ValidationError

from app.context.examples import ESTIMATION_EXAMPLES, HOURLY_RATE_USD
from app.services import llm_service

st.set_page_config(page_title="CAG Software Estimator")
st.title("CAG Software Estimator")
st.caption("Paste a meeting transcript to get an estimate. Then ask follow-up questions.")

# Only user and assistant turns live here. The system prompt is added by stream_estimation.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Streamlit reruns this script on every interaction, so draw the whole history each time.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Paste the meeting transcript here"):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # The history is only extended after the call works, so a failure leaves no trace in it.
            history = [*st.session_state.messages, {"role": "user", "content": prompt}]
            # write_stream accepts any iterable: it calls iter() on it, so the stream goes in as is.
            stream = llm_service.stream_estimation(history)
            started = time.perf_counter()
            answer = st.write_stream(stream)
            elapsed = time.perf_counter() - started
        except (ValidationError, openai.OpenAIError, llm_service.LLMStreamError):
            # Settings errors can name the variable and API errors can echo details: show neither.
            st.error(
                "The estimate could not be generated. Check the OpenAI key in .env "
                "and try again. Your last message was not saved: send it again."
            )
        else:
            if not answer.strip():
                # No text at all: nothing to save, so the user sends the message again.
                st.error(
                    "The model returned an empty answer. "
                    "Your last message was not saved: send it again."
                )
            else:
                st.session_state.messages = [*history, {"role": "assistant", "content": answer}]
                # Saved only on success, so a failed call never replaces the last good numbers.
                # getattr: a stream with no metrics (or none yet) counts as unknown, not as zero.
                st.session_state.last_call = {
                    "model": getattr(stream, "model", None),
                    "input_tokens": getattr(stream, "input_tokens", None),
                    "output_tokens": getattr(stream, "output_tokens", None),
                    "seconds": elapsed,
                }
                reason = getattr(stream, "incomplete_reason", None)
                if reason:
                    # Shown on this run only: it is not stored in the history.
                    st.warning(
                        f"The answer was cut off ({reason}) and may be incomplete. "
                        "The partial answer is saved."
                    )


def or_na(value) -> str:
    """Unknown values show as n/a: never an invented number."""
    return "n/a" if value is None else str(value)


def start_new_chat() -> None:
    # A button callback runs before the script reruns, so the old messages are never drawn again.
    st.session_state.messages = []
    st.session_state.pop("last_call", None)


# The sidebar is drawn last, after the call above, so it already sees this run's metrics.
with st.sidebar:
    st.button("New chat", on_click=start_new_chat)
    st.header("CAG context")

    with st.expander("System prompt (read-only)"):
        st.code(llm_service.build_system_prompt(), language=None, wrap_lines=True)

    st.subheader("Injected examples")
    st.caption(f"Rate in every example: {HOURLY_RATE_USD} USD/h")
    for number, example in enumerate(ESTIMATION_EXAMPLES, start=1):
        with st.expander(f"Example {number}"):
            st.markdown(example["meeting_summary"])
            st.markdown(example["estimation"].strip())

    st.subheader("Last call")
    last_call = st.session_state.get("last_call")
    if last_call is None:
        st.caption("No call yet.")
    else:
        st.metric("Model", or_na(last_call["model"]))
        st.metric("Input tokens", or_na(last_call["input_tokens"]))
        st.metric("Output tokens", or_na(last_call["output_tokens"]))
        st.metric("Response time", f"{last_call['seconds']:.1f} s")
