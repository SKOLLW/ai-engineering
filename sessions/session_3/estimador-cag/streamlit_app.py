"""Streamlit chat page: paste a meeting transcript, read the streamed estimate."""

import openai
import streamlit as st
from pydantic import ValidationError

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
            answer = st.write_stream(llm_service.stream_estimation(history))
        except (ValidationError, openai.OpenAIError, llm_service.LLMStreamError):
            # Settings errors can name the variable and API errors can echo details: show neither.
            st.error(
                "The estimate could not be generated. Check the OpenAI key in .env "
                "and try again. Your last message was not saved: send it again."
            )
        else:
            st.session_state.messages = [*history, {"role": "assistant", "content": answer}]
