import os
import re
import json
from pathlib import Path
from io import BytesIO
from typing import Dict, Any, List, Tuple

import streamlit as st
from openai import OpenAI


# =========================
# ----- Streamlit UI  -----
# =========================

st.set_page_config(page_title="MI RAG Assistant", layout="wide")
st.title("📄 Ask About MI Documents")


# =========================
# ----- Configuration -----
# =========================

# Allowed upload types for the sidebar uploader (per your request)
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx", ".xlsx"}

# Local folder containing your documents for the initial ingest
FOLDER_PATH = "G:/My Drive/work/mi/Input RAG/files"
CONFIG_PATH = "config.json"

# Supported extensions for the *vector store* creation (keep broad for initial ingest)
SUPPORTED_EXTENSIONS = {
    ".c", ".cpp", ".css", ".csv", ".doc", ".docx", ".gif", ".go", ".html", ".java",
    ".jpeg", ".jpg", ".js", ".json", ".md", ".pdf", ".php", ".pkl", ".png", ".pptx",
    ".py", ".rb", ".tar", ".tex", ".ts", ".txt", ".webp", ".xlsx", ".xml", ".zip"
}

# Models (names only)
MODEL_CHOICES = [
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-4o",
    "gpt-4o-mini",
]
DEFAULT_MODEL = "gpt-5-nano"


# =========================
# ------ OpenAI Client ----
# =========================

def get_api_key() -> str:
    try:
        return st.secrets["openai"]["OPENAI_API_KEY"]
    except Exception:
        return os.environ.get("OPENAI_API_KEY", "")

api_key = get_api_key()
if not api_key:
    st.error("Missing OpenAI API key. Put it in st.secrets['openai']['OPENAI_API_KEY'] or the OPENAI_API_KEY environment variable.")
    st.stop()

client = OpenAI(api_key=api_key)


# =========================
# -------- Helpers --------
# =========================

def load_config() -> Dict[str, Any]:
    return json.loads(Path(CONFIG_PATH).read_text()) if Path(CONFIG_PATH).exists() else {}

def save_config(config: dict):
    Path(CONFIG_PATH).write_text(json.dumps(config))

def upload_files_and_create_vector_store(folder_path: str) -> str:
    file_ids = []
    for file_path in Path(folder_path).glob("*"):
        if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            with open(file_path, "rb") as f:
                uploaded = client.files.create(file=f, purpose="assistants")
                file_ids.append(uploaded.id)
        else:
            print(f"Skipped unsupported file: {file_path.name}")

    vs = client.vector_stores.create(name="CompanyDocsVectorStore", file_ids=file_ids if file_ids else None)
    return vs.id

@st.cache_resource
def ensure_vector_store() -> str:
    config = load_config()
    if not config.get("vector_store_id"):
        vs_id = upload_files_and_create_vector_store(FOLDER_PATH)
        config["vector_store_id"] = vs_id
        save_config(config)
    return load_config()["vector_store_id"]

# --- sanitize weird wrappers like , [turnXfileY], and zero-width chars ---
_CITE_WRAP_RE = re.compile(r"\ue200.*?\ue201", flags=re.DOTALL)  # matches 
_TURN_FILE_RE = re.compile(r"\[turn\d+file\d+\]")
_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060"), None)

def sanitize_text(s: str) -> str:
    if not s:
        return s
    s = _CITE_WRAP_RE.sub("", s)
    s = _TURN_FILE_RE.sub("", s)
    s = s.translate(_ZERO_WIDTH)
    return s.strip()

def system_instructions(mode: str) -> str:
    base_rule = (
        "Do not output any special citation wrappers like '' or internal IDs like [turnXfileY]. "
        "Use plain inline numeric citations like [1], [2]. "
        "Always end your answer with a section titled 'References:' that lists each cited source "
        "numbered to match your inline markers. For files, include the exact filename and a short quote; "
        "for web, include title and URL."
    )
    if mode == "docs_only":
        return base_rule + " Use ONLY the uploaded documents. If the documents do not contain the answer, say so."
    if mode == "docs_plus_model":
        return base_rule + " Prefer the uploaded documents; if insufficient, you may use general knowledge and say so."
    return base_rule + " Prefer uploaded documents; you may also use general knowledge and web search (cite when used)."

def build_tools(mode: str, vector_store_id: str) -> List[Dict[str, Any]]:
    """
    Responses API: include vector store on the file_search tool itself.
    """
    tools: List[Dict[str, Any]] = [{
        "type": "file_search",
        "vector_store_ids": [vector_store_id],
    }]
    if mode == "docs_plus_model_web":
        tools.append({"type": "web_search"})
    return tools

import hashlib

def sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()

def get_vs_name_to_ids(vector_store_id: str) -> Dict[str, List[str]]:
    """
    Build {filename: [file_id, ...]} for files currently attached to the vector store.
    """
    name_to_ids: Dict[str, List[str]] = {}
    try:
        vs_files = client.vector_stores.files.list(vector_store_id=vector_store_id)
        for ref in vs_files.data:
            info = client.files.retrieve(ref.id)
            name_to_ids.setdefault(info.filename, []).append(ref.id)
    except Exception:
        pass
    return name_to_ids


def extract_sources_and_quotes(final_response) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Best-effort extraction of:
      - web_sources: list of {title, url, snippet?}
      - file_quotes: list of {filename, quote}
    Works with both streaming.get_final_response() and create() responses.
    """
    web_sources: List[Dict[str, str]] = []
    file_quotes: List[Dict[str, str]] = []

    try:
        for block in getattr(final_response, "output", []) or []:
            # Tool results
            if getattr(block, "type", "") == "tool_result":
                tool_name = getattr(block, "tool_name", None) or getattr(block, "name", None)
                data = getattr(block, "output", None)

                if tool_name == "web_search" and data:
                    items = []
                    if isinstance(data, dict) and "results" in data:
                        items = data.get("results", [])
                    elif isinstance(data, list):
                        items = data
                    for r in items[:10]:
                        if isinstance(r, dict):
                            title = r.get("title") or "Source"
                            url = r.get("url") or r.get("link") or ""
                            snippet = r.get("snippet") or ""
                            if url:
                                web_sources.append({"title": title, "url": url, "snippet": snippet})

                if tool_name == "file_search" and data:
                    items = []
                    if isinstance(data, dict) and "results" in data:
                        items = data.get("results", [])
                    elif isinstance(data, list):
                        items = data
                    for r in items[:10]:
                        quote = ""
                        filename = "file"
                        if isinstance(r, dict):
                            quote = r.get("quote") or r.get("text") or ""
                            file_meta = r.get("file") or {}
                            if isinstance(file_meta, dict):
                                filename = file_meta.get("filename") or file_meta.get("name") or filename
                        if quote:
                            file_quotes.append({"filename": filename, "quote": quote})

            # Output text annotations (SDK-dependent)
            if getattr(block, "type", "") == "output_text":
                anns = getattr(block, "annotations", None) or []
                for a in anns:
                    if not isinstance(a, dict):
                        continue
                    if a.get("type") == "web_citation":
                        web_sources.append({
                            "title": a.get("title") or "Source",
                            "url": a.get("url") or "",
                            "snippet": a.get("snippet") or ""
                        })
                    if a.get("type") == "file_citation":
                        file = a.get("file") or {}
                        filename = file.get("filename") if isinstance(file, dict) else "file"
                        quote = a.get("quote") or ""
                        if quote:
                            file_quotes.append({"filename": filename or "file", "quote": quote})
    except Exception:
        pass

    # Dedup
    seen_urls = set()
    dedup_web = []
    for s in web_sources:
        u = s.get("url") or ""
        if u and u not in seen_urls:
            dedup_web.append(s)
            seen_urls.add(u)

    seen_quotes = set()
    dedup_quotes = []
    for q in file_quotes:
        key = (q.get("filename"), q.get("quote"))
        if q.get("quote") and key not in seen_quotes:
            dedup_quotes.append(q)
            seen_quotes.add(key)

    return dedup_web[:10], dedup_quotes[:10]

def inject_inline_citations(answer_text: str, files_list: List[Dict[str, str]], web_list: List[Dict[str, str]], max_refs: int = 20) -> Tuple[str, List[Tuple[int, Dict[str, str], str]]]:
    """
    Deterministically inject inline [n] markers into the answer text so they match
    the References list we show below.

    Strategy:
      - Build a references sequence: first files_list, then web_list (up to max_refs).
      - Walk line-by-line. For each non-empty content line (ignoring headings),
        append the next [n] until we exhaust refs.

    Returns:
      - new_answer_text with injected markers
      - refs_mapping: list of (n, source_dict, source_type) where source_type is "file" or "web"
    """
    if not answer_text:
        return answer_text, []

    lines = answer_text.splitlines()

    # Build combined refs (cap to max_refs for readability)
    combined: List[Tuple[str, Dict[str, str]]] = []
    for f in files_list:
        combined.append(("file", f))
    for w in web_list:
        combined.append(("web", w))
    combined = combined[:max_refs]

    refs_mapping: List[Tuple[int, Dict[str, str], str]] = []
    ref_idx = 1
    out_lines: List[str] = []

    def is_heading(line: str) -> bool:
        l = line.strip()
        return (l.startswith("#") or l.endswith(":") or l.endswith("："))

    for line in lines:
        if not line.strip():
            out_lines.append(line)
            continue

        if ref_idx <= len(combined) and not is_heading(line):
            out_lines.append(f"{line} [{ref_idx}]")
            src_type, src = combined[ref_idx - 1]
            refs_mapping.append((ref_idx, src, src_type))
            ref_idx += 1
        else:
            out_lines.append(line)

    return "\n".join(out_lines), refs_mapping


# =========================
# ---- Sidebar Controls ---
# =========================
# (Placed BEFORE chat handling so it NEVER disappears during streaming/reruns.)

with st.sidebar:
    st.header("⚙️ Model & Retrieval")

    cfg = load_config()
    saved_model = cfg.get("responses_model", DEFAULT_MODEL)
    saved_mode = cfg.get("retrieval_mode", "docs_only")

    model_choice = st.selectbox(
        "OpenAI model",
        options=MODEL_CHOICES,
        index=MODEL_CHOICES.index(saved_model) if saved_model in MODEL_CHOICES else MODEL_CHOICES.index(DEFAULT_MODEL),
        help="Choose which model to use for chat."
    )

    retrieval_mode = st.radio(
        "Context sources",
        options=[
            ("docs_only", "Only the uploaded docs"),
            ("docs_plus_model", "Docs + GPT’s knowledge"),
            ("docs_plus_model_web", "Docs + GPT’s knowledge + Web search"),
        ],
        index=["docs_only", "docs_plus_model", "docs_plus_model_web"].index(saved_mode),
        format_func=lambda t: t[1],
        horizontal=False
    )[0]

    show_chunks = True

    if (model_choice != saved_model) or (retrieval_mode != saved_mode):
        cfg["responses_model"] = model_choice
        cfg["retrieval_mode"] = retrieval_mode
        cfg["show_chunks"] = bool(show_chunks)
        save_config(cfg)

    # ---- File Manager (ALWAYS visible; lives in the same sidebar block) ----

    # Ensure persistent vector store before listing
    vector_store_id = ensure_vector_store()

    st.divider()
    
    st.subheader("📁 Files in Use")
    try:
        files = client.vector_stores.files.list(vector_store_id=vector_store_id)
        existing_filenames = {
            client.files.retrieve(f.id).filename
            for f in files.data
        }
        for f in files.data:
            file_info = client.files.retrieve(f.id)
            st.markdown(f"- `{file_info.filename}`")
    except Exception as e:
        st.error(f"Error listing files: {e}")
        files = None
        existing_filenames = set()


    st.subheader("➕ Upload / Replace Files")
    
    # Optional switch: control whether we delete old file objects from global storage too
    delete_old_file_objects = st.toggle("Delete old copies from OpenAI storage", value=True,
                                        help="If on, the old file objects are removed from your OpenAI Files as well.")
    
    upload_key = st.session_state.get("upload_key", 0)
    
    uploaded_files = st.file_uploader(
        "Choose one or more files",
        type=[ext[1:] for ext in ALLOWED_EXTENSIONS],
        accept_multiple_files=True,
        key=f"uploader_{upload_key}"
    )
    
    if uploaded_files:
        # Build current index from vector store
        name_to_ids = get_vs_name_to_ids(vector_store_id)
    
        # Optional: persistent content index to avoid re-uploading identical content
        cfg = load_config()
        file_index: Dict[str, Dict[str, str]] = cfg.setdefault("file_index", {})  # {filename: {"file_id": "...", "sha256": "..."}}
    
        added, replaced, skipped = 0, 0, 0
    
        for uf in uploaded_files:
            fname = uf.name
            raw = uf.getvalue()
            new_hash = sha256_bytes(raw)
    
            # Skip if identical content (based on our local hash index)
            prev_meta = file_index.get(fname)
            if prev_meta and prev_meta.get("sha256") == new_hash:
                skipped += 1
                continue
    
            # If a file with the same name exists in the vector store, remove it first
            existing_ids = name_to_ids.get(fname, [])
            for old_id in existing_ids:
                try:
                    client.vector_stores.files.delete(vector_store_id=vector_store_id, file_id=old_id)
                except Exception as e:
                    st.warning(f"Could not detach old {fname} from vector store: {e}")
                if delete_old_file_objects:
                    try:
                        client.files.delete(old_id)
                    except Exception as e:
                        st.warning(f"Could not delete old file object {old_id}: {e}")
    
            # Upload and index the new file (single-file batch)
            with st.spinner(f"Uploading {fname}…"):
                client.vector_stores.file_batches.upload_and_poll(
                    vector_store_id=vector_store_id,
                    files=[(fname, BytesIO(raw))]
                )
    
            # Update our local indices
            # (Re-list to find the brand new file_id for this filename)
            name_to_ids = get_vs_name_to_ids(vector_store_id)
            new_ids = name_to_ids.get(fname, [])
            new_id = new_ids[-1] if new_ids else None
            file_index[fname] = {"file_id": new_id or "", "sha256": new_hash}
    
            if existing_ids:
                replaced += 1
            else:
                added += 1
    
        save_config(cfg)
        st.success(f"Done. Added {added}, replaced {replaced}, skipped {skipped} identical.")
        st.session_state["upload_key"] = upload_key + 1
        st.rerun()



    st.subheader("❌ Delete a File")

    if files and files.data:
        file_to_delete = st.selectbox(
            "Select a file to remove",
            options=files.data,
            format_func=lambda f: client.files.retrieve(f.id).filename
        )

    if files and files.data and st.button("Delete selected file"):
        try:
            client.vector_stores.files.delete(
                vector_store_id=vector_store_id,
                file_id=file_to_delete.id
            )
            st.success("Deleted successfully.")
            st.rerun()
        except Exception as e:
            st.error(f"Delete failed: {e}")
    elif not (files and files.data):
        st.info("No files to delete.")

# Small status line under title
st.caption(
    f"Model: {load_config().get('responses_model', DEFAULT_MODEL)} • "
    f"Mode: {load_config().get('retrieval_mode', 'docs_only')} • "
    f"Show chunks: {bool(load_config().get('show_chunks', False))}"
)


# =========================
# ---- Conversation State --
# =========================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {"role","content"}


# =========================
# ---- Render history -----
# =========================

# Render existing history first; the new streamed message will appear below.
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# =========================
# ------ Chat Input -------
# =========================

user_input = st.chat_input("Ask your question...")

if user_input:
    # Append & show user bubble
    safe_user = sanitize_text(user_input)
    st.session_state.chat_history.append({"role": "user", "content": safe_user})
    with st.chat_message("user"):
        st.markdown(safe_user)

    # Stream assistant reply inside ONE chat bubble (no duplicates)
    streamed_text = ""
    sources_web: List[Dict[str, str]] = []
    quotes_files: List[Dict[str, str]] = []
    final_response_obj = None

    with st.chat_message("assistant"):
        bubble = st.empty()

        try:
            mode = load_config().get("retrieval_mode", "docs_only")
            model = load_config().get("responses_model", DEFAULT_MODEL)

            input_messages = [{"role": "system", "content": system_instructions(mode)}]
            for m in st.session_state.chat_history:
                input_messages.append({"role": m["role"], "content": m["content"]})

            with client.responses.stream(
                model=model,
                input=input_messages,
                tools=build_tools(mode, vector_store_id),
                include=["output[*].file_search_call.search_results"]
            ) as stream:
                for event in stream:
                    if event.type == "response.output_text.delta":
                        streamed_text += event.delta
                        bubble.markdown(sanitize_text(streamed_text))
                    elif event.type == "response.error":
                        bubble.error(f"Error: {event.error}")

                stream.until_done()
                final_response_obj = stream.get_final_response()

            sources_web, quotes_files = extract_sources_and_quotes(final_response_obj)

        except Exception as e:
            # Non-streaming fallback
            with st.spinner("Thinking..."):
                mode = load_config().get("retrieval_mode", "docs_only")
                model = load_config().get("responses_model", DEFAULT_MODEL)

                input_messages = [{"role": "system", "content": system_instructions(mode)}]
                for m in st.session_state.chat_history:
                    input_messages.append({"role": m["role"], "content": m["content"]})

                resp = client.responses.create(
                    model=model,
                    input=input_messages,
                    tools=build_tools(mode, vector_store_id),
                    include=["output[*].file_search_call.search_results"]
                )

                final_text = ""
                for item in resp.output or []:
                    if getattr(item, "type", "") == "output_text":
                        final_text += getattr(item, "text", "") or ""
                streamed_text = sanitize_text(final_text or "")

                bubble.markdown(streamed_text)
                final_response_obj = resp
                sources_web, quotes_files = extract_sources_and_quotes(resp)

        # 1) Sanitize final answer
        final_answer = sanitize_text(streamed_text)

        # 2) Build deterministic inline citations from extracted sources (files first, then web)
        final_answer_with_cites, refs_mapping = inject_inline_citations(
            final_answer,
            files_list=quotes_files,
            web_list=sources_web
        )

        # 3) Ensure a References section that matches the inline numbering
        if refs_mapping:
            refs_lines = []
            for n, src, src_type in refs_mapping:
                if src_type == "file":
                    fn = (src.get("filename") or "file").strip()
                    quote = (src.get("quote") or src.get("text") or "").strip()
                    refs_lines.append(f"[{n}] **{fn}** — {quote}")
                else:
                    title = src.get("title") or "Source"
                    url = src.get("url") or ""
                    snippet = src.get("snippet") or ""
                    if url:
                        refs_lines.append(f"[{n}] [{title}]({url})" + (f" — {snippet}" if snippet else ""))
                    else:
                        refs_lines.append(f"[{n}] {title}" + (f" — {snippet}" if snippet else ""))

            # If the model already wrote a References section, append ours below it;
            # otherwise add a fresh one.
            if re.search(r"(?im)^\s*references\s*:", final_answer_with_cites):
                final_answer_with_cites += "\n\n" + "\n".join(refs_lines)
            else:
                final_answer_with_cites += "\n\n**References:**\n" + "\n".join(refs_lines)

        # 4) Replace bubble with the final, cited answer
        bubble.markdown(final_answer_with_cites)

        # 5) Persist the answer to history (so it renders once on next rerun)
        st.session_state.chat_history.append({"role": "assistant", "content": final_answer_with_cites})

        # 6) Optional: show top retrieved chunks (from files)
        if bool(load_config().get("show_chunks", False)) and quotes_files:
            with st.expander("Top retrieved chunks (from your files)"):
                for q in quotes_files[:10]:
                    fn = (q.get("filename") or "file").strip()
                    quote = (q.get("quote") or "").strip()
                    if quote:
                        st.markdown(f"- **{fn}** — {quote}")

# (Nothing else below; sidebar stays visible, history renders once, streaming happens in-bubble)
