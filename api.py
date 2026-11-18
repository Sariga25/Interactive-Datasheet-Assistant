import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

app = FastAPI(title="Interactive Hardware Datasheet RAG API")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

# Retrieve top 8 chunks using MMR (Maximal Marginal Relevance) to avoid duplicate snippets
retriever = vector_db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 8, "fetch_k": 20}
)

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.0)

prompt = ChatPromptTemplate.from_template("""
You are a senior Embedded Systems & Hardware Engineering Assistant.
Answer the user's question using ONLY the provided datasheet excerpts below.

STRICT ENGINEERING & SAFETY RULES:
1. Retrieval Completeness: Extract all relevant numerical specs (voltages, currents, frequencies, memory, interfaces) found across the excerpts. If a specific detail is missing from the excerpts, explicitly state: "Not specified in the retrieved datasheet sections." Never guess or fill gaps from outside memory.
2. Electrical & GPIO Safety: Pay strict attention to logic voltage limits vs. board power input. Never assume a 5V-powered board has 5V-tolerant GPIO pins unless explicitly stated in the excerpt (for example, BeagleBone Black GPIOs are strictly 3.3V and NOT 5V tolerant). Highlight critical voltage limits clearly.
3. Component Variant Precision: Many datasheets cover multiple device variants (such as LM358 vs. LM358B, or ATmega640/1280/2560). Always specify which exact part variant a specification applies to, and note differences if the datasheet distinguishes them.
4. Inline Source Traceability: Next to every major specification or claim, cite the exact document name, page number, and section title/number visible in the excerpt (e.g., [Beaglebone.pdf, Page 45, Section 7.1]).

Datasheet Excerpts:
{context}

Question: {question}
""")

class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_datasheet(req: QueryRequest):
    docs = retriever.invoke(req.question)
    
    formatted_chunks = []
    sources = []
    
    for doc in docs:
        file_name = os.path.basename(doc.metadata.get("source", "Unknown"))
        category = os.path.basename(os.path.dirname(doc.metadata.get("source", "")))
        page_num = doc.metadata.get("page", 0) + 1
        
        # Tag each chunk with its exact file and page so the LLM can cite it inline
        chunk_header = f"--- SOURCE: {file_name} | Category: {category} | Page: {page_num} ---"
        formatted_chunks.append(f"{chunk_header}\n{doc.page_content}")
        
        source_label = f"{file_name} (Page {page_num})"
        if source_label not in sources:
            sources.append(source_label)
            
    context_text = "\n\n".join(formatted_chunks)
    
    chain = prompt | llm
    response = chain.invoke({"context": context_text, "question": req.question})
            
    return {
        "answer": response.content,
        "sources": sources
    }

