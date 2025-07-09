import os
import time
import json
from pathlib import Path

import streamlit as st
import openai

# Load API key
openai.api_key = st.secrets['openai']["OPENAI_API_KEY"]

# Local folder containing your documents
FOLDER_PATH = "G:/My Drive/work/mi/Input RAG/files"
CONFIG_PATH = "config.json"

SUPPORTED_EXTENSIONS = {
    ".c", ".cpp", ".css", ".csv", ".doc", ".docx", ".gif", ".go", ".html", ".java",
    ".jpeg", ".jpg", ".js", ".json", ".md", ".pdf", ".php", ".pkl", ".png", ".pptx",
    ".py", ".rb", ".tar", ".tex", ".ts", ".txt", ".webp", ".xlsx", ".xml", ".zip"
}

# Load/save config for persistent assistant and vector store
def load_config():
    return json.loads(Path(CONFIG_PATH).read_text()) if Path(CONFIG_PATH).exists() else {}

def save_config(config):
    Path(CONFIG_PATH).write_text(json.dumps(config))

# Upload supported files and create vector store
def upload_files_and_create_vector_store(folder_path):
    file_ids = []
    for file_path in Path(folder_path).glob("*"):
        if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            with open(file_path, "rb") as f:
                file = openai.files.create(file=f, purpose="assistants")
                file_ids.append(file.id)
        else:
            print(f"Skipped unsupported file: {file_path.name}")
    vs = openai.vector_stores.create(name="CompanyDocsVectorStore", file_ids=file_ids)
    return vs.id

@st.cache_resource
def initialize_assistant_and_vector_store():
    config = load_config()

    # Create vector store if missing
    if not config.get("vector_store_id"):
        vs_id = upload_files_and_create_vector_store(FOLDER_PATH)
        config["vector_store_id"] = vs_id
        save_config(config)
    else:
        vs_id = config["vector_store_id"]

    # Create assistant if missing
    if not config.get("assistant_id"):
        assistant = openai.beta.assistants.create(
            name="Company Docs Assistant",
            instructions="Answer questions using the company documents only.",
            model="gpt-4o",
            tools=[{"type": "file_search"}],
            tool_resources={"file_search": {"vector_store_ids": [vs_id]}}
        )
        config["assistant_id"] = assistant.id
        save_config(config)
    else:
        assistant = openai.beta.assistants.retrieve(config["assistant_id"])

    return assistant.id, vs_id

# Start the assistant and vector store (only once)
assistant_id, vector_store_id = initialize_assistant_and_vector_store()

# Create a new thread per session
if "thread_id" not in st.session_state:
    thread = openai.beta.threads.create(
        tool_resources={"file_search": {"vector_store_ids": [vector_store_id]}}
    )
    st.session_state.thread_id = thread.id
    st.session_state.chat_history = []

# UI Layout
st.set_page_config(page_title="Company RAG Assistant", layout="wide")
st.title("📄 Ask Your Company Documents")

user_input = st.chat_input("Ask your question...")

# Handle input
if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    openai.beta.threads.messages.create(
        thread_id=st.session_state.thread_id,
        role="user",
        content=user_input
    )

    run = openai.beta.threads.runs.create(
        thread_id=st.session_state.thread_id,
        assistant_id=assistant_id
    )

    with st.spinner("Thinking..."):
        while True:
            status = openai.beta.threads.runs.retrieve(
                thread_id=st.session_state.thread_id,
                run_id=run.id
            )
            if status.status == "completed":
                break
            time.sleep(1)

    messages = openai.beta.threads.messages.list(thread_id=st.session_state.thread_id)
    response = messages.data[0].content[0].text.value

    st.session_state.chat_history.append({"role": "assistant", "content": response})

# Display conversation
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
