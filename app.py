# app.py
import sys
from pathlib import Path
from unittest.mock import MagicMock


fake_module = MagicMock()
sys.modules['torchvision'] = fake_module
sys.modules['torchvision.transforms'] = fake_module
sys.modules['torchvision.transforms.v2'] = fake_module
fake_vision.__spec__ = types.SimpleNamespace(name="torchvision")

import streamlit as st


from transformers import logging as hf_logging
hf_logging.set_verbosity_error()


current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from src.main import HallucinationProofRAG


st.set_page_config(page_title="CCS Companion", page_icon="⚖️", layout="centered")


@st.cache_resource(show_spinner="Initializing Rules Engine & Indexes...")
def load_rag_engine():
    return HallucinationProofRAG()

rag = load_rag_engine()


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
    
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    


if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    user_prompt = st.session_state.messages[-1]["content"]
    
    with st.chat_message("assistant"):
        with st.spinner("Consulting rulebooks and verifying citations..."):
            
            response = rag.query(user_prompt)
            
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
