import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()



uploadDirectory = "data/uploads"
indexDirectory = "data/indices"
textDirectory = "data/raw_texts"
documentDirectory = "data/documents"

vectorDimension = 384  # or 768 depending on model
embeddingModel = "all-MiniLM-L6-v2"  # adjust as needed
groqApiKey = os.getenv("GROQ_API_KEY")
OCR_APPLICATION_FILE_PATH = r"tesseract_ocr\\tesseract.exe"
TESSERACT_MODEL = "eng_all"


def ensureDirs():
    os.makedirs(uploadDirectory, exist_ok=True)
    os.makedirs(indexDirectory, exist_ok=True)
    os.makedirs(textDirectory, exist_ok=True)
    os.makedirs(documentDirectory, exist_ok=True)