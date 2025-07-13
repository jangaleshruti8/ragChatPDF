from flask import Flask, request, jsonify, send_from_directory
from pathlib import Path
from pipeline import chunkingFiles, indexingFiles, textExtraction, uploadFiles, ragChatBot
import os

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_pdf():
    file = request.files['pdf']
    file_path = uploadFiles.saveUploadedPdf(file)

    if not Path(file_path).exists():
        return jsonify({"error": "Upload failed."}), 400

    raw_text = textExtraction.extractTextFromPdf(file_path)
    if raw_text:
        textExtraction.saveRawText(raw_text, file_path)
        chunks = chunkingFiles.chunkText(raw_text)
        indexingFiles.indexChunks(chunks, {"source": file.filename})
        return jsonify({"message": "File indexed successfully!"})

    return jsonify({"error": "Text extraction failed."}), 500

@app.route('/query', methods=['POST'])
def answer_question():
    data = request.get_json()
    query = data.get('query')
    if not query:
        return jsonify({"error": "No query provided."}), 400

    matched_chunks = ragChatBot.queryChunks(query)
    context = "\n\n".join(matched_chunks)
    answer = ragChatBot.generateAnswerWithGroq(query, context)
    return jsonify({"answer": answer})

@app.route('/')
def serve_ui():
    return send_from_directory('.', 'chatbotUserInterface.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
