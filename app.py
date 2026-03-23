import streamlit as st
from rag_pipeline import build_vector_store , ask_question
import tempfile , os

st.set_page_config(
    page_title="DocuMind - RAG Q&A",
    page_icon="📃",
    layout="centered"
)

st.title("DocuMind")
st.caption("Upload a PDF and ask questions - powered by Hugging Face + RAG")

with st.sidebar:
    st.header("⚙️ Settings")
    hf_token = st.text_input(
        "Hugging Face Token",
        type="password",
        help="Get your free token at huggingface.co/settings/tokens"
    )
    st.markdown("---")
    st.markdown("**How it works:**")
    st.markdown("1. Upload a PDF")
    st.markdown("2. Split into chunks")
    st.markdown("3. Embed with Sentence Transformers")
    st.markdown("4. Retrieve relevant chunks")
    st.markdown("5. LLM generates the answer")

uploaded_file = st.file_uploader("📂 Upload a PDF document", type=["pdf"])

if uploaded_file and hf_token:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    with st.spinner("🔍 Reading and indexing document..."):
        try:
            vector_store = build_vector_store(tmp_path, hf_token)
            st.success(f"✅ '{uploaded_file.name}' indexed successfully!")
            os.unlink(tmp_path)
            st.session_state["vector_store"] = vector_store
            st.session_state["hf_token"] = hf_token
        except Exception as e:
            st.error(f"Error: {e}")

elif uploaded_file and not hf_token:
    st.warning("⚠️ Please enter your Hugging Face token in the sidebar.")

if "vector_store" in st.session_state:
    st.markdown("---")
    st.subheader("💬 Ask a question")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("e.g. What are the main findings?")

    if question:
        st.session_state["messages"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = ask_question(
                        question,
                        st.session_state["vector_store"],
                        st.session_state["hf_token"]
                    )
                    st.markdown(result["answer"])
                    with st.expander("📎 Source passages used"):
                        for i, src in enumerate(result["sources"], 1):
                            st.markdown(f"**Chunk {i}:** {src[:300]}...")
                    st.session_state["messages"].append({"role": "assistant", "content": result["answer"]})
                except Exception as e:
                    import traceback
                    st.error(f"Error: {traceback.format_exc()}")
else:
    st.info("👆 Upload a PDF and enter your Hugging Face token to get started.")