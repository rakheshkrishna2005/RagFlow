import streamlit as st
import requests
import json
import os
from typing import Dict, List
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "https://ragflow-zhoa.onrender.com")
API_KEY = os.getenv("API_KEY")

st.set_page_config(
    page_title="RAG Assistant",
    page_icon="📚",
    layout="wide"
)

if "documents" not in st.session_state:
    st.session_state.documents = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def make_api_request(documents: List[str], questions: List[str]) -> Dict:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {"documents": documents, "questions": questions}

    try:
        response = requests.post(f"{API_URL}/run", headers=headers, json=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return None

with st.sidebar:
    st.header("📑 Settings")
    doc_name = st.text_input("Document Name")
    doc_url = st.text_input("Document URL")

    if st.button("Add Document", use_container_width=True):
        if doc_name and doc_url and doc_url.endswith(".pdf"):
            st.session_state.documents[doc_name] = doc_url
        else:
            st.error("Please provide a valid PDF URL")

    if st.session_state.documents:
        st.header("📜 Select Document(s)")
        selected_docs = st.multiselect(
            "Choose document(s)",
            options=list(st.session_state.documents.keys()),
            default=list(st.session_state.documents.keys())
        )

        if st.button("Clear Documents", use_container_width=True):
            st.session_state.documents.clear()
            st.session_state.chat_history.clear()
            st.rerun()

        if st.button("Clear Chat", use_container_width=True):
            st.session_state.chat_history.clear()
            st.experimental_rerun()
    else:
        st.info("No documents added yet.")

st.markdown("<h1 style='text-align: center;'>📚 RagFlow Agent</h1>", unsafe_allow_html=True)

for entry in st.session_state.chat_history:
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])

if st.session_state.documents and selected_docs:
    user_input = st.chat_input("Ask a question about your document(s)...")
    if user_input:
        st.chat_message("user").markdown(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        selected_urls = [st.session_state.documents[name] for name in selected_docs]

        with st.spinner("Analyzing with RAG..."):
            response = make_api_request(selected_urls, [user_input])
            if response and "answers" in response:
                answer = response["answers"][0]
                with st.chat_message("assistant"):
                    st.markdown(answer)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
else:
    st.info("👈 Add and select documents in the sidebar to start chatting!")

st.markdown("""
<hr>
<div style='text-align: center; color: gray;'>
Built with ❤️ using Streamlit + Custom RAG 
<a href='https://github.com/rakheshkrishna2005/RagFlow/tree/main/API' 
   target='_blank' style='text-decoration: none; color: green;'>
   API
</a>
</div>
""", unsafe_allow_html=True)
