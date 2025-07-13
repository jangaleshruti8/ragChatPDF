import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

groqApiKey = os.getenv("GROQ_API_KEY")
embeddingModel = "all-MiniLM-L6-v2"
vectorDimension = 384
topKResults = 5
uploadDirectory = "data/uploads"
rawTextDirectory = "data/raw_texts"
indexPath = "data/faiss_index.idx"
documentStorePath = "data/doc_store.pkl"
OCR_APPLICATION_FILE_PATH = "tesseract_ocr\\tesseract.exe"

TESSERACT_MODEL = "eng_all"

def ensureDirs():
    Path(uploadDirectory).mkdir(parents=True, exist_ok=True)
    Path(rawTextDirectory).mkdir(parents=True, exist_ok=True)
