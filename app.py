import streamlit as st
import requests

st.set_page_config(page_title="Hardware Datasheet AI Assistant", page_icon="🔌", layout="centered")

st.title("🔌 Interactive Hardware Datasheet Assistant")
st.caption("Ask technical questions across 12 pre-indexed ECE datasheets (STM32, BeagleBone, ATmega2560, ESP32, MPU6050, L298N, LM2596 & more).")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            st.caption("📄 **Sources:** " + " | ".join(msg["sources"]))

if question := st.chat_input("e.g., What is the operating voltage range of LM2596 or STM32F103C8T6?"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching datasheets..."):
            try:
                response = requests.post(
                    "http://localhost:8000/ask",
                    json={"question": question},
                    timeout=30
                )
                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    sources = data["sources"]
                    
                    st.markdown(answer)
                    if sources:
                        st.caption("📄 **Sources:** " + " | ".join(sources))
                        
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                else:
                    st.error(f"API Error {response.status_code}: {response.text}")
            except Exception as e:
                st.error(f"Could not connect to FastAPI server. Make sure uvicorn is running! ({e})")
