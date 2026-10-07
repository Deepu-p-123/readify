import time
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

import agent  # your existing bot (tools, OCR, memory settings)

st.set_page_config(page_title="Document Chat Bot", page_icon="💬", layout="centered")

NO_DOC = "(no document loaded)"
SUPPORTED = ["pdf", "docx", "txt", "md", "csv", "png", "jpg", "jpeg", "webp"]

# ---------- Session state ----------
if "messages" not in st.session_state:
    st.session_state.messages = []       # what is shown on screen
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []   # what the model remembers (last N)
if "document_text" not in st.session_state:
    st.session_state.document_text = NO_DOC
if "doc_key" not in st.session_state:
    st.session_state.doc_key = None      # identifies the file already loaded
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None
if "doc_warning" not in st.session_state:
    st.session_state.doc_warning = None


def reset_chat():
    st.session_state.messages = []
    st.session_state.chat_history = []


def reset_document():
    st.session_state.document_text = NO_DOC
    st.session_state.doc_key = None
    st.session_state.doc_name = None
    st.session_state.doc_warning = None


def ask_bot(question: str) -> str:
    """Run the agent with memory and the loaded document. Retries on 503."""
    for attempt in range(3):
        try:
            result = agent.agent_executor.invoke({
                "input": question,
                "chat_history": st.session_state.chat_history,
                "document": st.session_state.document_text,
            })
            answer = agent.get_text(result["output"])

            history = st.session_state.chat_history + [
                HumanMessage(content=question),
                AIMessage(content=answer),
            ]
            history = history[-agent.MEMORY_SIZE:]
            if history and isinstance(history[0], AIMessage):
                history = history[1:]  # history must start with a human message
            st.session_state.chat_history = history
            return answer
        except Exception as e:
            if "503" in str(e) and attempt < 2:
                time.sleep(2 ** attempt)
            else:
                return "Sorry, something went wrong. Please try again."
    return "Sorry, something went wrong. Please try again."


# ---------- Sidebar: document ----------
with st.sidebar:
    st.header("📄 Document")
    uploaded = st.file_uploader(
        "Upload a file to ask questions about",
        type=SUPPORTED,
        help="Scanned PDFs and images are read with OCR (slower).",
    )

    if uploaded is None:
        # file removed (or never uploaded)
        if st.session_state.doc_key is not None:
            reset_document()
            reset_chat()
    else:
        key = (uploaded.name, uploaded.size)
        if key != st.session_state.doc_key:
            with st.status(f"Reading {uploaded.name}...", expanded=True) as status:
                # Show the bot's progress messages (OCR page x/y) in the sidebar
                agent.status_hook = lambda msg: st.write(msg)
                read_error = False
                try:
                    text = agent.read_document_bytes(uploaded.name, uploaded.getvalue())
                except Exception as e:
                    text = ""
                    read_error = True
                    status.update(label="Could not read the file", state="error")
                    st.error(str(e))
                finally:
                    agent.status_hook = lambda msg: print(f"Bot: {msg}")

                if text.strip():
                    warning = None
                    if len(text) > agent.MAX_DOC_CHARS:
                        text = text[:agent.MAX_DOC_CHARS]
                        warning = (f"The document is long, so only the first "
                                   f"{agent.MAX_DOC_CHARS:,} characters were loaded.")
                    if "could not be read]" in text:
                        warning = ("Some pages could not be read (the AI service was "
                                   "busy). Remove the file and upload it again to "
                                   "retry only those pages.")
                    st.session_state.document_text = text
                    st.session_state.doc_key = key
                    st.session_state.doc_name = uploaded.name
                    st.session_state.doc_warning = warning
                    reset_chat()
                    status.update(label=f"Loaded {uploaded.name}", state="complete",
                                  expanded=False)
                elif not read_error:
                    status.update(label="No text found in that file", state="error")

    if st.session_state.doc_name:
        st.success(f"**{st.session_state.doc_name}**\n\n"
                   f"{len(st.session_state.document_text):,} characters loaded")
        if st.session_state.doc_warning:
            st.warning(st.session_state.doc_warning)
    else:
        st.info("No document loaded. You can still chat, use the calculator, "
                "or ask for the weather.")

    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        reset_chat()
        st.rerun()
    st.caption(f"The bot remembers the last {agent.MEMORY_SIZE} messages.")


# ---------- Main: chat ----------
st.title("💬 Document Chat Bot")
st.caption("Ask questions about your document, do math, or check the weather.")

if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown("Hi! Upload a document in the sidebar, or just start chatting. "
                    "Try: *summarize the document*, *what is 23 * 47?*, or "
                    "*weather in Kochi*.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Ask something..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = ask_bot(question)
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})