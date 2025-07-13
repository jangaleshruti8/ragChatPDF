import cv2
import pytesseract
import numpy as np
import pdfplumber
from pathlib import Path
from PIL import Image
import rag_config

pytesseract.pytesseract.tesseract_cmd = rag_config.OCR_APPLICATION_FILE_PATH

class OCRPipeline:
    def __init__(self):
        pass

    def extract_text(self, image_array=None, image_pil_object=None):
        print("Text Extraction pipeline starts")
        if image_pil_object is not None:
            image_array = np.array(image_pil_object, dtype=np.uint8)
        elif image_array is not None:
            image_array = np.array(image_array, dtype=np.uint8)
        else:
            raise ValueError("Either 'image_array' or 'image_pil_object' must be provided.")

        if image_array.dtype != np.uint8:
            image_array = image_array.astype(np.uint8)

        if len(image_array.shape) == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

        custom_config = r'--psm 3 -c preserve_interword_spaces=1'
        text = pytesseract.image_to_string(image_array, lang=rag_config.TESSERACT_MODEL, config=custom_config)

        print("Text returned successfully from OCR")
        return text

ocr_pipeline = OCRPipeline()

def extractTextFromPdf(pdfPath: str) -> str:
    try:
        text = ""
        with pdfplumber.open(pdfPath) as pdf:
            for i, page in enumerate(pdf.pages):
                # Try extracting layout-aware text
                page_text = page.extract_text(layout=True)

                if page_text and page_text.strip():
                    text += page_text + "\n"
                else:
                    # Fallback to OCR
                     print(f"Page {i+1} has no extractable text. Using OCR...")
                     pil_image = page.to_image(resolution=300).original  # already a PIL image
                     ocr_text = ocr_pipeline.extract_text(image_pil_object=pil_image)
                     text += ocr_text + "\n"

        return text

    except Exception as e:
        print(f"Error: {e}")
        return ""

def saveRawText(text: str, originalPdfPath: str) -> str:
    fileName = Path(originalPdfPath).stem + ".txt"
    savePath = Path("data/raw_texts") / fileName
    savePath.parent.mkdir(parents=True, exist_ok=True)
    with open(savePath, "w", encoding="utf-8") as f:
        f.write(text)
    return str(savePath)


