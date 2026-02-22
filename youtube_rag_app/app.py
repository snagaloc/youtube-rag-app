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

    with st.sidebar:
        st.header("Settings")
        preferred_lang = st.text_input("Preferred transcript language", value="en")
        translate_if_needed = st.checkbox("Translate to English if needed", value=True)
        k = st.slider("Top-K chunks", 2, 10, 5)

    url_or_id = st.text_input("YouTube URL or Video ID", value="Gfr50f6ZBvo")
    build_btn = st.button("Build / Load Index")

    if "vs" not in st.session_state:
        st.session_state.vs = None
        st.session_state.video_id = None
        st.session_state.lang = None

    if build_btn:
        try:
            video_id = extract_video_id(url_or_id)
            st.session_state.video_id = video_id

            with st.spinner("Fetching transcript..."):
                snippets = fetch_transcript(video_id, preferred_lang=preferred_lang)

            persist_dir = os.path.join(os.getcwd(), "chroma_store", video_id)
            os.makedirs(persist_dir, exist_ok=True)

            with st.spinner("Building vector index..."):
                vs, detected_lang, _, _ = build_pipeline(
                    video_id=video_id,
                    snippets=snippets,
                    preferred_lang=preferred_lang,
                    persist_dir=persist_dir,
                    translate_if_needed=translate_if_needed
                )

            st.session_state.vs = vs
            st.session_state.lang = detected_lang
            st.success(f"Index ready  video_id={video_id} | detected_lang={detected_lang}")

        except Exception as e:
            st.error(str(e))

    st.divider()

    if st.session_state.vs is None:
        st.info("Paste a YouTube URL/ID and click **Build / Load Index**.")
        return

    st.subheader("Ask a question")
    question = st.text_input("Question", value="Summarize this video in 4 points.")
    ask_btn = st.button("Ask")

    if ask_btn and question.strip():
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

            if isinstance(start, (int, float)):
                link = youtube_timestamp_url(video_id, float(start))
                st.write(f"- [{sec_to_mmss(start)} → {sec_to_mmss(end or start)}] {c['preview']}")
                st.markdown(f"  - 🔗 {link}")
            else:
                st.write(f"- [no timestamp] {c['preview']}")

if __name__ == "__main__":
    main()