# AI PDF Assistant — RAG Based

## What it does
Upload any PDF and ask questions about it. 
The app uses RAG (Retrieval Augmented Generation) to find 
relevant chunks from the PDF and answer intelligently.

## Tech Stack
- Python
- LangChain
- ChromaDB (Vector Database)
- Streamlit (UI)
- Ollama (Local LLM)

## How it Works
1. PDF is uploaded and split into chunks
2. Chunks are converted to embeddings
3. Stored in ChromaDB vector database
4. User asks a question
5. Relevant chunks are retrieved
6. LLM generates answer based on those chunks

## How to Run
git clone https://github.com/Abdullah1dev/ai-pdf-assistant-rag
pip install -r requirements.txt
streamlit run app.py


