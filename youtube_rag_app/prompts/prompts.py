from langchain_core.prompts import ChatPromptTemplate

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant.\n"
     "Answer ONLY from the provided transcript context.\n"
     "If the context is insufficient, say exactly: I don't know.\n"
     "Return the answer in numbered points:\n"
     "1. ...\n2. ...\n3. ...\n4. ..."),
    ("human", "Context:\n{context}\n\nQuestion:\n{question}")
])

TRANSLATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a translation engine. Translate to English.\n"
     "- Preserve meaning and tone.\n"
     "- Do NOT add commentary.\n"
     "- Output ONLY translated English text."),
    ("human", "Source language guess: {lang}\n\nTEXT:\n{text}")
])