import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from rag_config import embeddingModel,indexPath,vectorDimension,documentStorePath
from pathlib import Path


embeddingModel = SentenceTransformer(embeddingModel)

def embedChunks(chunks: list) -> np.ndarray:
    return embeddingModel.encode(chunks)

def createOrLoadIndex():
    if Path(indexPath).exists():
        return faiss.read_index(indexPath)
    return faiss.IndexFlatL2(vectorDimension)

def indexChunks(chunks: list, docMeta: dict):
    vectors = embedChunks(chunks)
    index = createOrLoadIndex()
    index.add(np.array(vectors, dtype=np.float32))

    if Path(documentStorePath).exists():
        with open(documentStorePath, "rb") as f:
            docStore = pickle.load(f)
    else:
        docStore = []

    for i, chunk in enumerate(chunks):
        docStore.append({"text": chunk, "meta": docMeta})

    with open(documentStorePath, "wb") as f:
        pickle.dump(docStore, f)
    faiss.write_index(index, indexPath)
