# Interactive Hardware Datasheet Assistant (RAG Chatbot)

A Retrieval-Augmented Generation (RAG) engineering assistant built with **LangChain**, **ChromaDB**, **FastAPI**, **Groq LLM**, and **Streamlit**. It enables embedded engineers to query electrical specifications, pin limits, and communication interfaces across pre-indexed hardware datasheets with inline source and page citations.

## System Architecture & RAG Workflow

1. **Document Ingestion (`ingest.py`)**
   - Scans categorized hardware PDFs (`boards`, `sensors`, `communication`, `motor_drivers`, `analog_digital_ics`, `displays`, `power_management`) using `PyPDFDirectoryLoader`.
   - Applies `RecursiveCharacterTextSplitter` (`chunk_size=1200`, `chunk_overlap=250`) to preserve electrical specification tables and section headers.
   - Generates 384-dimensional vector embeddings locally via HuggingFace (`all-MiniLM-L6-v2`) and persists them in **ChromaDB** (`chroma_db/`).

2. **Retrieval & Grounded Generation (`api.py`)**
   - Exposes a **FastAPI** `POST /ask` REST endpoint validated with Pydantic.
   - Uses Maximal Marginal Relevance (`MMR`, `k=8`, `fetch_k=20`) retrieval from ChromaDB to gather diverse, non-redundant specification chunks.
   - Enforces strict prompt engineering guardrails (`temperature=0.0`) with **Groq (`openai/gpt-oss-120b`)** to prevent electrical safety hallucinations (e.g., 3.3V vs. 5V GPIO tolerance) and enforce inline `[Document, Page, Section]` citations.

3. **Interactive Chat Interface (`app.py`)**
   - Built with **Streamlit** to maintain multi-turn session history and render formatted technical responses alongside verified PDF page sources.

## Indexed Datasheet Collection (v1.0 Baseline)
- **Microcontrollers & Boards:** STM32F103C8T6, Raspberry Pi Pico (RP2040), BeagleBone Black, Arduino Uno R3, ESP32, ATmega2560
- **Peripherals & Power:** SSD1306 (OLED Display), MPU6050 (IMU), HC-05 (Bluetooth), L298N (Motor Driver), LM358 (Op-Amp), LM2596 (Buck Regulator)

*(Note: STM32F103C8T6, Raspberry Pi Pico RP2040, and SSD1306 demonstrated the highest retrieval accuracy in v1.0 baseline testing).*

## Installation & How to Run

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Configure environment variables:**
   Create a `.env` file in the root directory and add your Groq API key:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```
3. **Build the ChromaDB vector index (one-time setup):**
   ```bash
   python ingest.py
   ```
4. **Start the FastAPI backend (Terminal 1):**
   ```bash
   uvicorn api:app --reload
   ```
5. **Start the Streamlit chat UI (Terminal 2):**
   ```bash
   streamlit run app.py
   ```

## Sample Questions to Test
- *"What are the operating voltage range, flash memory, and communication interfaces of the STM32F103C8T6?"*
- *"List the core processor, SRAM size, and GPIO voltage specs for the Raspberry Pi Pico (RP2040)."*
- *"What communication protocols and supply voltage limits are supported by the SSD1306 OLED controller?"*

## Known Limitations & Roadmap (v1.1)
- **Complex Multi-Column Tables:** Dense PDF layout tables in large family datasheets (e.g., ATmega640/1280/2560) occasionally split across chunk boundaries.
- **Multi-Variant Disambiguation:** Datasheets covering multiple IC revisions (such as LM358 vs. LM358B) require stricter metadata filtering in future iterations.
- **Future Upgrade:** Replace lower-performing PDF documents iteratively with cleaner single-variant datasheets and benchmark retrieval accuracy using the same test suite.
