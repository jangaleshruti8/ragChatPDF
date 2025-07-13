def chunkText(text: str, chunkSize: int = 1000, overlap: int = 200) -> list:
    text = text.replace("\n", " ").strip()
    chunks = []
    for i in range(0, len(text), chunkSize - overlap):
        chunk = text[i:i+chunkSize]
        if len(chunk) > 200:
            chunks.append(chunk)
    return chunks
