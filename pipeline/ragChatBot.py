import pickle
import numpy as np
import faiss
from pathlib import Path
from rag_config import indexDirectory, documentDirectory
from pipeline.indexingFiles import embedChunks
from openai import OpenAI
import rag_config

def queryChunks(query: str, fileName: str, topK: int = 5):
    embeddedQuery = embedChunks([query])[0].reshape(1, -1)

    indexPath = Path(indexDirectory) / f"{fileName}.index"
    docPath = Path(documentDirectory) / f"{fileName}.pkl"

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



def generateAnswerWithGroq(query: str, contextChunks: list) -> str:
    if not contextChunks:
        return "I'm sorry, I cannot answer this question because the available context is insufficient."

    context = "\n\n".join(contextChunks)

    client = OpenAI(api_key=rag_config.groqApiKey, base_url="https://api.groq.com/openai/v1")
    prompt = f"""
    Use the context below to answer the question. If the context doesn't help, say so.

    CONTEXT:
    {context}

    QUESTION:
    {query}

    ANSWER:
    """
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content
