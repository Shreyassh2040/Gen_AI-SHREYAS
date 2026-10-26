"""NoteBot - a minimal RAG chatbot that answers questions about your PDF notes.

Run with:  streamlit run app.py
"""

import io
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

# ---------------------------------------------------------------- configuration
CHAT_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "models/gemini-embedding-001"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 4

load_dotenv(Path(__file__).with_name(".env"))

st.set_page_config(page_title="NoteBot", page_icon="📚", layout="centered")

PROMPT = ChatPromptTemplate.from_template(
    """You are my assistant tutor. Answer the question based on the following context and
if you did not get the context simply say "I don't know Jenny" :

{context}

Question: {input}
"""
)


def get_api_key() -> str | None:
    """Prefer Streamlit secrets, fall back to the .env file."""
    try:
        if "GOOGLE_API_KEY" in st.secrets:
            return st.secrets["GOOGLE_API_KEY"]
    except Exception:
        pass  # no secrets.toml configured - that is fine
    return os.getenv("GOOGLE_API_KEY")


@st.cache_resource(show_spinner="Reading your notes and building the knowledge base...")
def build_knowledge_base(
    file_bytes: bytes,
    file_name: str,
    api_key: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
):
    """PDF -> text -> chunks -> embeddings -> FAISS.

    Cached on the file bytes, so Streamlit reruns (every keystroke) never
    re-embed the document.
    """
    reader = PdfReader(io.BytesIO(file_bytes))

    text = ""
    for page in reader.pages:
        text += (page.extract_text() or "") + "\n"

    if not text.strip():
        raise ValueError(
            "This PDF has no selectable text. It is probably a scanned image, "
            "so run it through OCR first."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = splitter.split_text(text)

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL, google_api_key=api_key, transport="rest"
    )
    vector_store = FAISS.from_texts(chunks, embeddings)

    return vector_store, len(reader.pages), len(chunks), len(text)


# ------------------------------------------------------------------------- UI
st.header("NoteBot")

with st.sidebar:
    st.title("My Notes")
    file = st.file_uploader("Upload notes PDF and start asking questions", type="pdf")

api_key = get_api_key()

if not api_key:
    st.error("No API key found.")
    st.markdown(
        "1. Get a free key at <https://aistudio.google.com/apikey>  \n"
        "2. Open `server/GEN_AI/.env` and paste it after `GOOGLE_API_KEY=`  \n"
        "3. Restart the app."
    )
    st.stop()

if file is None:
    st.info("Upload a PDF from the sidebar, then ask a question below.")
    st.stop()

try:
    vector_store, page_count, chunk_count, char_count = build_knowledge_base(
        file.getvalue(), file.name, api_key
    )
except Exception as exc:
    st.error(f"Could not read **{file.name}**: {exc}")
    st.stop()

with st.sidebar:
    st.success(f"**{file.name}** is ready")
    st.caption(
        f"{page_count} pages · {chunk_count} chunks · "
        f"{char_count:,} characters · ~{char_count // 4:,} tokens"
    )

with st.form("ask"):
    user_query = st.text_input("Type your query here")
    submitted = st.form_submit_button("Ask")

if submitted and user_query.strip():
    matching_chunks = vector_store.similarity_search(user_query, k=TOP_K)

    llm = ChatGoogleGenerativeAI(
        model=CHAT_MODEL,
        temperature=0,
        max_output_tokens=300,
        thinking_budget=0,  # 2.5-flash reasons by default; keep the budget for the answer
        google_api_key=api_key,
        transport="rest",  # HTTP instead of gRPC - survives DNS/proxy quirks
    )
    chain = create_stuff_documents_chain(llm, PROMPT)

    with st.spinner("Thinking..."):
        try:
            # `context` is the key create_stuff_documents_chain expects.
            output = chain.invoke({"input": user_query, "context": matching_chunks})
        except Exception as exc:
            st.error(f"The model call failed: {exc}")
        else:
            st.write(output)

            with st.expander("Sources used"):
                for index, chunk in enumerate(matching_chunks, start=1):
                    preview = chunk.page_content[:500].replace("\n", " ")
                    st.markdown(f"**Chunk {index}**")
                    st.caption(f"{preview}...")
