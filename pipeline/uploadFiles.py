import shutil
from pathlib import Path
from rag_config import ensureDirs, uploadDirectory

def saveUploadedPdf(file) -> str:
    ensureDirs()
    savePath = Path(uploadDirectory) / file.filename  
    with open(savePath, "wb") as f:
        shutil.copyfileobj(file, f)
    return str(savePath)
