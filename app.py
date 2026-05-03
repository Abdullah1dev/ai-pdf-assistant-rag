import streamlit as st
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import ollama

st.title("AI PDF Assistant")


if "messages" not in st.session_state:
    st.session_state.messages = []


@st.cache_resource
def load_db():
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )

    db = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )
    return db

db = load_db()


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


query = st.chat_input("Ask something...")

if query:

    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    recent_messages = st.session_state.messages[-6:]
    chat_history = "\n".join(
        [f"{m['role']}: {m['content']}" for m in recent_messages]
    )

   
    with st.spinner("Searching documents..."):
        results = db.similarity_search(query, k=5)

    context = "\n\n".join([doc.page_content for doc in results])[:1200]

    prompt = f"""
You are a helpful AI Assistant.

Rules:
If the answer is not clearly present in the context:
→ respond ONLY with: I don't know
→ do NOT use prior knowledge

Conversation:
{chat_history}

Context:
{context}

Question:
{query}

Answer:
"""

 
 
 
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            stream = ollama.chat(
                model="qwen2.5:1.5b",
                messages=[{"role": "user", "content": prompt}],
                stream=True
            )

            answer = ""
            response_placeholder = st.empty()

            for chunk in stream:
                content = chunk['message']['content']
                answer += content
                response_placeholder.markdown(answer + "▌")

            response_placeholder.markdown(answer)
            
            

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })