import openai
import pickle
import numpy as np
import faiss
from rag_config import indexPath
from pipeline.indexingFiles import embedChunks,documentStorePath
from openai import OpenAI
import rag_config

def queryChunks(query: str, topK: int = 5):
    embeddedQuery = embedChunks([query])[0].reshape(1, -1)
    index = faiss.read_index(indexPath)

    distances, indices = index.search(embeddedQuery, topK)

    with open(documentStorePath, "rb") as f:
        docs = pickle.load(f)

    return [docs[i]["text"] for i in indices[0]]

def generateAnswerWithGroq(query: str, context: str) -> str:
    client = OpenAI(
        api_key=rag_config.groqApiKey,
        base_url="https://api.groq.com/openai/v1"
    )

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