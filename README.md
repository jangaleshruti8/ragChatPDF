#  RAG-based PDF Query Bot 

This project allows you to upload a PDF (searchable or scanned) and ask natural language questions about its contents. It uses a **Retrieval-Augmented Generation (RAG)** approach powered by **FAISS for vector search** and **Groq-hosted LLaMA3 models** for answering queries.

---

##  Features

-  Upload PDFs (searchable or image-based)
-  Extract text using `pdfplumber` or fallback to OCR using `Tesseract`
-  Chunk, embed, and index the content using `SentenceTransformers + FAISS`
-  Ask questions about the content and receive answers powered by **Groq-hosted LLaMA3**
-  Simple HTML+JS chatbot UI for interaction

---

##  How It Works

### 1.  Upload PDF

- The frontend UI lets users upload a `.pdf` file.
- The file is stored under `data/uploads/` directory.

### 2.  Text Extraction

- If the PDF is **searchable**, text is extracted using `pdfplumber`.
- If the PDF has no extractable text (e.g., scanned image), it falls back to **OCR** using `Tesseract` (`pytesseract`).

### 3.  Chunking (splits text into chunks)

- The text is split into overlapping chunks using a sliding window to preserve context.
- This is handled by `pipeline/chunkingFiles.py`.

### 4.  Embedding & Indexing

- Each chunk is embedded using the `all-MiniLM-L6-v2` model from `sentence-transformers`.
- These embeddings are stored in a **FAISS index** (`faiss_index.idx`), and metadata in `doc_store.pkl`.

### 5.  Question Answering ChatBot

- When the user asks a question:
  - The question is embedded.
  - Top-k most relevant chunks are retrieved using vector similarity.
  - The question and chunks are passed to **Groq's LLaMA3-8B** model via the OpenAI-compatible API for answer generation.

---

##  Technologies Used

| Component      | Tool / Library                   |
|----------------|----------------------------------|
| Backend        | Flask                            |
| Frontend       | HTML, CSS, JS                    |
| OCR            | Tesseract                        |
| PDF Text       | pdfplumber                       |
| Embeddings     | sentence-transformers            |
| Indexing       | FAISS                            |
| LLM API        | Groq (OpenAI API compatible)     |
| Model          | LLaMA3-8B-8192                   |

---

##  Directory Structure

chatPDF/
│
├── data/
│ ├── uploads/ # Uploaded PDF files
│ ├── raw_texts/ # Extracted text files (.txt)
│ ├── faiss_index.idx # FAISS index file
│ └── doc_store.pkl # Metadata and chunks
│
├── pipeline/
│ ├── uploadFiles.py # Handles file saving
│ ├── textExtraction.py # Extracts text (pdfplumber + OCR)
│ ├── chunkingFiles.py # Chunks the text
│ ├── indexingFiles.py # Embeds and indexes chunks
│ └── ragChatBot.py # Handles RAG querying and answering
│
├── ragQuestionAnswer.py # Flask app
├── chatbotUserInterface.html # Frontend UI
├── rag_config.py # Configs and paths
└── .env # API keys and paths

---

##  Setup Instructions

### 1. Install Python Dependencies

pip install -r requirements.txt
2. Install Tesseract OCR
Download and install from: https://github.com/tesseract-ocr/tesseract

Update the OCR_APPLICATION_FILE_PATH in rag_config.py accordingly.

3. Setup .env
Create a .env file in the root directory:

GROQ_API_KEY=your_groq_api_key_here

4. Run the Server

python ragQuestionAnswer.py
Visit: http://localhost:5000


Usage Flow---
Upload a .pdf file from the frontend

Text is extracted (OCR fallback if needed)

Content is chunked and indexed

Ask questions through the text box

Receive an answer based on relevant context from the uploaded PDF

##  Clone This Repository
git clone https://github.com/jangaleshruti8/ragChatPDF.git
cd ragChatBotRepo
