import sys
from pathlib import Path
from unittest.mock import MagicMock
import streamlit as st
from loguru import logger
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()

# Ensure the src directory is accessible
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from src.main import HallucinationProofRAG

st.set_page_config(page_title="CCS Companion", page_icon="⚖️", layout="centered")

from state import get_rag_engine

rag = get_rag_engine()

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("⚖️ CCS Companion")
st.caption("AI-Powered Central Civil Services Rules & Procedures Assistant")
st.write("---")


if not st.session_state.messages:
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    left_pad, center_card, right_pad = st.columns([1, 5, 1])
    
    with center_card:
        st.markdown(
            """
            <div style='text-align: center; margin-bottom: 25px;'>
                <h3 style='margin-bottom: 5px;'>How can I help you with Service Rules today?</h3>
                <p style='color: gray;'>Ask complex questions regarding central government job guidelines, allowances, and entitlements.</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        col1, col2 = st.columns(2)
        with col1:
            st.info("**📅 Leave Rules**\n\n*“What are the eligibility criteria and maximum duration for Child Care Leave (CCL)?”*")
            st.info("**📈 Pay Fixation**\n\n*“How is pay fixed upon promotion under the 7th CPC guidelines?”*")
        with col2:
            st.info("**✈️ LTC Rules**\n\n*“Can you explain the current criteria for LTC encashment during service?”*")
            st.info("**🛡️ Pension & Gratuity**\n\n*“What is the maximum qualifying ceiling limit for retirement gratuity?”*")


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        if message.get("citations"):
            with st.expander("📚 Verified References"):
                for cite in message["citations"]:
                    section_info = f" (Section: {cite['section']})" if cite.get('section') else ""
                    st.caption(f"**[{cite['ref']}]** {cite['document']}{section_info}")
        
        if message.get("is_fallback"):
            st.warning(f"⚠️ Fallback Triggered: {message.get('fallback_reason')}")



if prompt := st.chat_input("Ask a question about central government rules..."):
    
    
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # RENDER and SAVE the assistant response in the same frame
    with st.chat_message("assistant"):
        with st.spinner("Consulting rulebooks and verifying citations..."):
            
            try:
                
                response = rag.query(prompt)
                
                if response.is_fallback:
                    st.warning(f"⚠️ Fallback Triggered: {response.fallback_reason}")
                    
                st.markdown(response.answer)
                
                if response.citations:
                    with st.expander("📚 Verified References", expanded=False):
                        for cite in response.citations:
                            section_info = f" (Section: {cite['section']})" if cite.get('section') else ""
                            st.caption(f"**[{cite['ref']}]** {cite['document']}{section_info}")
                
                st.caption(f"Confidence Score: {response.raw_confidence:.2f} | Reliability: {response.confidence_level}")
                
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response.answer,
                    "citations": response.citations,
                    "is_fallback": response.is_fallback,
                    "fallback_reason": response.fallback_reason
                })
                st.rerun()
            except Exception as e:
                error_msg = str(e)

                # Always log the full raw error to terminal
                logger.exception("LLM call failed | type={} | detail={}", type(e).__name__, error_msg)

                # Show only clean, non-technical messages in the UI
                if "401" in error_msg or "authentication" in error_msg.lower() or "api key" in error_msg.lower():
                    st.error("🔑 **Invalid API Key** — Please check your OpenRouter API key in the `.env` file.")

                elif "402" in error_msg or "credit" in error_msg.lower() or "balance" in error_msg.lower():
                    st.error("💳 **Account Quota Exhausted** — Your OpenRouter free limits are used up. Please wait or switch models.")

                elif "429" in error_msg or "rate limit" in error_msg.lower() or "rate-limited" in error_msg.lower():
                    st.error("⏳ **Model Busy** — The AI model is rate limited right now. Please wait a few seconds and try again.")

                elif "503" in error_msg or "service unavailable" in error_msg.lower():
                    st.error("🔧 **Service Unavailable** — The AI provider is temporarily down. Please try again shortly.")

                elif "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                    st.error("⌛ **Request Timed Out** — The model took too long to respond. Please try again.")

                elif "connection" in error_msg.lower():
                    st.error("🌐 **Connection Error** — Could not reach the AI provider. Please check your internet connection.")

                else:
                    st.error("❌ **Something went wrong** — The assistant encountered an unexpected error. Please try again.")

    
    
   
