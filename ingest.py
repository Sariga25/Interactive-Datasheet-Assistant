import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

print("Scanning datasheets/ folder and loading PDFs...")
loader = PyPDFDirectoryLoader("datasheets")
raw_documents = loader.load()
print(f"Loaded {len(raw_documents)} total pages from your datasheets!")

# Larger chunks & overlap keep electrical spec tables and section headers together
print("Splitting pages into larger, table-preserving chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=250,
    separators=["\n\n", "\n", " ", ""]
)
chunks = text_splitter.split_documents(raw_documents)
print(f"Created {len(chunks)} searchable text chunks.")

print("Loading embedding model (all-MiniLM-L6-v2)...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("Saving vectors into ChromaDB...")
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db"
)

print("SUCCESS! Upgraded ChromaDB knowledge base is ready!")

