# backend/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from model import make_db, generate_response
import os

app = Flask(__name__)
CORS(app) 

vectorstore = None
db_created = False

@app.route('/create-db', methods=['POST'])
def create_db():
    global vectorstore
    global db_created
    try:
        vectorstore = make_db()
        db_created = True
        return jsonify({"message": "Database created successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    question = data.get("query")
    if not question:
        return jsonify({"error": "No query provided"}), 400
    
    try:
        # Assume vectorstore is globally available for this example
        response = generate_response(vectorstore, question)
        return jsonify({"response": response}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
