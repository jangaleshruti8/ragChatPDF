#  RAG-based PDF Query Bot 

This project allows you to upload a PDF (searchable or scanned) and ask natural language questions about its contents. It uses a **Retrieval-Augmented Generation (RAG)** approach powered by **FAISS for vector search** and **Ollama-hosted LLaMA3 models** for answering queries.

---

##  Features

-  Upload PDFs (searchable or image-based)
-  Extract text using `pdfplumber` or fallback to OCR using `Tesseract`
-  Chunk, embed, and index the content using **locally downloaded SentenceTransformers + FAISS**
-  Ask questions about the content and receive answers powered by **Ollama's local LLaMA3 model**
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
  - The question and chunks are passed to **Ollama LLaMA3-8B** model via the OpenAI-compatible API for answer generation.

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
| LLM API        | Ollama                           |
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

---

##  Setup Instructions

### 1. Install Python Dependencies

```sh
pip install -r requirements.txt
```

### 2. Install Tesseract OCR

Download and install from: https://github.com/tesseract-ocr/tesseract

Update the `OCR_APPLICATION_FILE_PATH` in `rag_config.py` accordingly.

### 3. Download SentenceTransformer Model (for offline use)

While online, run:

```sh
python downloadModelSentenceTransformer.py
```

This will download and save the `all-MiniLM-L6-v2` model to the `all-MiniLM-L6-v2/` directory.  
**After this step, the embedding model is used completely offline.**

Or Use the provided model 'all-MiniLM-L6-v2' 

### 4. Install and Set Up Ollama (for local LLM inference)

- Download and install Ollama from: https://ollama.com/download
- Pull the LLaMA3 model (8B variant) for local use:

```sh
ollama pull llama3:8b
```

- Start the Ollama server (usually runs automatically in the background):

```sh
ollama serve
```

**Ollama and the LLaMA3-8B model run fully offline after this step.**

### 5. Run the Server

```sh
python ragQuestionAnswer.py
```

Visit: [http://localhost:5000](http://localhost:5000)

---

##  Usage Flow

1. Upload a .pdf file from the frontend
2. Text is extracted (OCR fallback if needed)
3. Content is chunked and indexed
4. Ask questions through the text box
5. Receive an answer based on relevant context from the uploaded PDF

---

##  Clone This Repository

```sh
git clone https://github.com/jangaleshruti8/ragChatPDF.git
cd ragChatPDF
```

**Note:**  
- All models (embedding and LLM) are used locally; no external API calls are required after setup.
- Make sure Ollama is running and the model is pulled before starting the Flask
