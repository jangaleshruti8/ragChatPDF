
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from rag_config import embeddingModel, indexDirectory, vectorDimension, documentDirectory
from pathlib import Path

embeddingModel = SentenceTransformer(embeddingModel)

def embedChunks(chunks: list) -> np.ndarray:
    return embeddingModel.encode(chunks)

def indexChunks(chunks: list, docMeta: dict, fileName: str):
    vectors = embedChunks(chunks)
    index = faiss.IndexFlatL2(vectorDimension)
    index.add(np.array(vectors, dtype=np.float32))

    docStore = [{"text": chunk, "meta": docMeta} for chunk in chunks]

    # Save by file name
    indexPath = Path(indexDirectory) / f"{fileName}.index"
    docPath = Path(documentDirectory) / f"{fileName}.pkl"

    faiss.write_index(index, str(indexPath))
    with open(docPath, "wb") as f:
        pickle.dump(docStore, f)


def listIndexedFiles():
    return [f.stem for f in Path(indexDirectory).glob("*.index")]