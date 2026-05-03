from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("MLBOOK.pdf")
pages = loader.load()




from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

docs = text_splitter.split_documents(pages)


from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)




from langchain_chroma import Chroma
import os

if not os.path.exists("chroma_db"):
    db = Chroma.from_documents(docs,embeddings,persist_directory="chroma_db")
    db.persist()
else:
    db = Chroma(persist_directory="chroma_db",embedding_function = embeddings)

