from typing import List, Tuple, Dict, Any
from dataclasses import dataclass

from langdetect import detect, LangDetectException

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser

from prompts.prompts import ANSWER_PROMPT, TRANSLATE_PROMPT
from services.youtube_service import TranscriptSnippet

def safe_detect_language(text: str) -> str:
    sample = (text or "").strip()
    if not sample:
        return "unknown"
    if len(sample) > 2000:
        sample = sample[:2000]
    try:
        return detect(sample)
    except LangDetectException:
        return "unknown"

def snippets_to_documents(
    snippets: List[TranscriptSnippet],
    video_id: str,
    lang: str,
    chunk_chars: int = 1800
) -> List[Document]:
    """
    Build timestamp-aware documents by grouping transcript snippets into blocks.
    Each Document gets metadata: video_id, start, end, lang
    """
    docs: List[Document] = []
    buf = []
    buf_chars = 0
    start_t = None
    end_t = None

    for s in snippets:
        t = (s.text or "").strip()
        if not t:
            continue

        s_start = float(s.start)
        s_end = float(s.start + (s.duration or 0.0))

        if start_t is None:
            start_t = s_start
        end_t = s_end

        projected = buf_chars + len(t) + 1
        if projected > chunk_chars and buf:
            docs.append(Document(
                page_content=" ".join(buf).strip(),
                metadata={"video_id": video_id, "start": start_t, "end": end_t, "lang": lang}
            ))
            buf = []
            buf_chars = 0
            start_t = s_start
            end_t = s_end

        buf.append(t)
        buf_chars += len(t) + 1

    if buf:
        docs.append(Document(
            page_content=" ".join(buf).strip(),
            metadata={"video_id": video_id, "start": start_t, "end": end_t, "lang": lang}
        ))

    return docs

def translate_to_english(text: str, source_lang: str) -> str:
    """
    Translate using OpenAI chat model in chunks (simple + safe).
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    parts = splitter.split_text(text)

    chain = TRANSLATE_PROMPT | llm | StrOutputParser()
    out = [chain.invoke({"lang": source_lang, "text": p}) for p in parts]
    return "\n".join(out).strip()

def build_vectorstore(docs: List[Document], persist_dir: str) -> Chroma:
    """
    Build/load Chroma persistent DB from documents.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_dir
    )

def retrieve_context(vs: Chroma, question: str, k: int = 5) -> List[Document]:
    retriever = vs.as_retriever(search_type="mmr", search_kwargs={"k": k, "lambda_mult": 0.5})
    return retriever.invoke(question)

def answer_question(question: str, docs: List[Document]) -> str:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    context = "\n\n".join(d.page_content for d in docs)
    chain = ANSWER_PROMPT | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": question})

def build_pipeline(
    video_id: str,
    snippets: List[TranscriptSnippet],
    preferred_lang: str,
    persist_dir: str,
    translate_if_needed: bool = True
) -> Tuple[Chroma, str, str, List[Document]]:
    """
    Returns:
      vectorstore, detected_lang, english_text, docs_indexed
    """
    raw_text = " ".join(s.text for s in snippets)
    detected = safe_detect_language(raw_text)

    english_text = raw_text
    if translate_if_needed and detected not in ("en", "unknown"):
        english_text = translate_to_english(raw_text, detected)

    # Build docs from ORIGINAL snippets but set lang to detected;
    # content used is English (better for embeddings).
    # We keep timestamps in metadata from original.
    base_docs = snippets_to_documents(snippets, video_id=video_id, lang=detected, chunk_chars=1800)

    # Replace doc content with English-chunked text (simple approach):
    # We'll re-split the english_text and attach "unknown" timestamps,
    # OR keep timestamps from base_docs but use English translation per block.
    #
    # Simple + understandable approach: keep base_docs (timestamps) and embed them,
    # but if translated, we just embed translated full text split with same metadata.
    if english_text != raw_text:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
        parts = splitter.split_text(english_text)
        docs_for_index = [
            Document(page_content=p, metadata={"video_id": video_id, "start": None, "end": None, "lang": detected})
            for p in parts
        ]
    else:
        # Further split timestamp docs a bit
        splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
        docs_for_index = []
        for d in base_docs:
            parts = splitter.split_text(d.page_content)
            for p in parts:
                docs_for_index.append(Document(page_content=p, metadata=d.metadata))

    vs = build_vectorstore(docs_for_index, persist_dir=persist_dir)
    return vs, detected, english_text, docs_for_index

def citations_from_docs(docs: List[Document]) -> List[Dict[str, Any]]:
    cites = []
    for d in docs:
        cites.append({
            "video_id": d.metadata.get("video_id"),
            "start": d.metadata.get("start"),
            "end": d.metadata.get("end"),
            "preview": (d.page_content[:220].replace("\n", " ").strip())
        })
    return cites