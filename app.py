import os
import streamlit as st
from dotenv import load_dotenv

from services.utils import extract_video_id, sec_to_mmss, youtube_timestamp_url
from services.youtube_service import fetch_transcript
from services.rag_service import build_pipeline, retrieve_context, answer_question, citations_from_docs

load_dotenv()

def main():
    st.set_page_config(page_title="YouTube RAG", layout="wide")
    st.title("YouTube RAG (Simple + Metadata + Timestamps)")

    # -----------------------------
    # Session state init
    # -----------------------------
    if "vs" not in st.session_state:
        st.session_state.vs = None
    if "video_id" not in st.session_state:
        st.session_state.video_id = None
    if "lang" not in st.session_state:
        st.session_state.lang = None
    if "last_input" not in st.session_state:
        st.session_state.last_input = ""

    # -----------------------------
    # Sidebar settings
    # -----------------------------
    with st.sidebar:
        st.header("Settings")
        preferred_lang = st.text_input("Preferred transcript language", value="en")
        translate_if_needed = st.checkbox("Translate to English if needed", value=True)
        k = st.slider("Top-K chunks", 2, 10, 5)

    # -----------------------------
    # Build section (use a form)
    # -----------------------------
    st.subheader("1) Choose YouTube video")
    with st.form("build_form", clear_on_submit=False):
        url_or_id = st.text_input("YouTube URL or Video ID", value="")
        build_btn = st.form_submit_button("Build / Load Index")

    # Reset index if user changed video input (even before clicking build)
    if url_or_id.strip() and url_or_id.strip() != st.session_state.last_input:
        st.session_state.vs = None
        st.session_state.video_id = None
        st.session_state.lang = None
        st.session_state.last_input = url_or_id.strip()

    if build_btn:
        if not url_or_id.strip():
            st.warning("Paste a YouTube URL or 11-char video ID.")
        else:
            try:
                video_id = extract_video_id(url_or_id)   # must support URL + ID
                st.session_state.video_id = video_id

                with st.spinner("Fetching transcript..."):
                    snippets = fetch_transcript(video_id, preferred_lang=preferred_lang)

                persist_dir = os.path.join(os.getcwd(), "chroma_store", video_id)
                os.makedirs(persist_dir, exist_ok=True)

                with st.spinner("Building / Loading vector index..."):
                    # IMPORTANT: build_pipeline must return vs + detected_lang at least
                    out = build_pipeline(
                        video_id=video_id,
                        snippets=snippets,
                        preferred_lang=preferred_lang,
                        persist_dir=persist_dir,
                        translate_if_needed=translate_if_needed
                    )

                    # Support either (vs, detected_lang, ...) OR dict return
                    if isinstance(out, dict):
                        vs = out["vs"]
                        detected_lang = out.get("detected_lang", "unknown")
                    else:
                        # your code expects: vs, detected_lang, _, _
                        vs = out[0]
                        detected_lang = out[1] if len(out) > 1 else "unknown"

                st.session_state.vs = vs
                st.session_state.lang = detected_lang

                st.success(f"Index ready ✅  video_id={video_id} | detected_lang={detected_lang}")

            except Exception as e:
                st.error(f"Build failed: {e}")

    st.divider()

    # -----------------------------
    # Ask section (use a form)
    # -----------------------------
    st.subheader("2) Ask a question")

    if st.session_state.vs is None:
        st.info("Paste a YouTube URL/ID and click **Build / Load Index**.")
        return

    with st.form("ask_form", clear_on_submit=False):
        question = st.text_input("Question", value="Summarize this video in 4 points.")
        ask_btn = st.form_submit_button("Ask")

    if ask_btn:
        if not question.strip():
            st.warning("Type a question.")
            return

        try:
            vs = st.session_state.vs
            video_id = st.session_state.video_id

            with st.spinner("Retrieving..."):
                docs = retrieve_context(vs, question, k=k)

            with st.spinner("Answering..."):
                answer = answer_question(question, docs)

            st.markdown("### Answer")
            st.write(answer)

            st.markdown("### Citations")
            cites = citations_from_docs(docs)

            for c in cites:
                start = c.get("start")
                end = c.get("end")
                preview = c.get("preview", "")

                if isinstance(start, (int, float)):
                    link = youtube_timestamp_url(video_id, float(start))
                    st.write(f"- [{sec_to_mmss(start)} → {sec_to_mmss(end or start)}] {preview}")
                    st.markdown(f"  - 🔗 {link}")
                else:
                    st.write(f"- [no timestamp] {preview}")

        except Exception as e:
            st.error(f"Ask failed: {e}")

if __name__ == "__main__":
    main()