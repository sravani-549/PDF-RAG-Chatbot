import streamlit as st
import backend as demo

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📄",
    layout="centered"
)

st.title("📄 PDF RAG Chatbot (Ask Your Document)")
st.caption("Upload any PDF and chat with it using AI + LangChain + FAISS + OpenAI")

# ==================================================
# SESSION STATE
# ==================================================
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "memory" not in st.session_state:
    st.session_state.memory = demo.demo_memory()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ==================================================
# SIDEBAR
# ==================================================
with st.sidebar:
    st.header("⚙️ Controls")

    # Reset chat
    if st.button("🧹 Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.memory = demo.demo_memory()
        st.rerun()

    # Remove PDF
    if st.button("📄 Remove PDF", use_container_width=True):
        st.session_state.vectorstore = None
        st.session_state.chat_history = []
        st.success("PDF removed. Upload a new one.")
        st.rerun()

    st.divider()

    st.markdown(
        """
### 🚀 Features

- Upload any PDF
- Ask questions from document
- Conversation memory
- FAISS vector search
- GPT-4o-mini reasoning

---
"""
    )


# ==================================================
# PDF UPLOAD SECTION
# ==================================================
st.subheader("📤 Upload PDF")

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"]
)

if uploaded_file is not None:

    if st.button("🚀 Process PDF", use_container_width=True):

        with st.spinner("Reading and indexing PDF..."):

            st.session_state.vectorstore = demo.process_pdf(uploaded_file)

        st.success("✅ PDF processed successfully! You can now chat.")


# ==================================================
# CHAT SECTION
# ==================================================
if st.session_state.vectorstore:

    st.divider()
    st.subheader("💬 Chat with your PDF")

    # Show chat history
    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.markdown(chat["text"])

    # Input box
    user_input = st.chat_input("Ask something from your PDF...")

    if user_input:

        # USER MESSAGE
        st.chat_message("user").markdown(user_input)

        st.session_state.chat_history.append(
            {"role": "user", "text": user_input}
        )

        # BOT RESPONSE
        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                response = demo.ask_question(
                    st.session_state.vectorstore,
                    user_input,
                    st.session_state.memory
                )

            st.markdown(response)

        # Save history
        st.session_state.chat_history.append(
            {"role": "assistant", "text": response}
        )

else:
    st.info("👆 Upload a PDF and click 'Process PDF' to start chatting.")


# ==================================================
# FOOTER
# ==================================================
st.divider()
st.caption("Built with LangChain + FAISS + OpenAI + Streamlit 🚀")