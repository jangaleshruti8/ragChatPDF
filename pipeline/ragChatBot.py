import openai
import pickle
import numpy as np
import faiss

from pipeline.indexingFiles import embedChunks,documentDirectory
import rag_config
import requests

def queryChunks(query: str, fileName: str, topK: int = 5):
    embeddedQuery = embedChunks([query])[0].reshape(1, -1)

    indexPath = rag_config.Path(rag_config.indexDirectory) / f"{fileName}.index"
    docPath = rag_config.Path(rag_config.documentDirectory) / f"{fileName}.pkl"

    if not indexPath.exists() or not docPath.exists():
        return ["No context found for selected file."]

    index = faiss.read_index(str(indexPath))
    with open(docPath, "rb") as f:
        docs = pickle.load(f)

    distances, indices = index.search(embeddedQuery, topK)
    print("Distances:", distances[0])

    # Dynamic threshold based on spread of distances
    avg_distance = np.mean(distances[0])
    std_dev = np.std(distances[0])
    adaptive_threshold = avg_distance + 0.25 * std_dev
    print(f"Adaptive threshold: {adaptive_threshold:.4f}")

    # Filter using adaptive threshold
    filtered = [
        (docs[i]["text"], distances[0][j])
        for j, i in enumerate(indices[0])
        if distances[0][j] <= adaptive_threshold
    ]

    if not filtered:
        return []  # insufficient context

    return [text for text, dist in filtered]


def generateAnswerWithOllama(query: str, context: str) -> str:
    prompt = f"""
    Use the context below to answer the question. If the context doesn't help, say so.

    CONTEXT:
    {context}

    QUESTION:
    {query}

    ANSWER:
    """
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3:8b",  
            "prompt": prompt,
            "stream": False 
        }
    )
    resp_json = response.json()
    print("Ollama response:", resp_json)  # For debugging
    if "response" in resp_json:
        return resp_json["response"]
    elif "error" in resp_json:
        return f"Ollama error: {resp_json['error']}"
    else:
        return "Unexpected Ollama response format."
