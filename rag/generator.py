# rag/generator.py

import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma:2b-instruct-q4_0"

def generate_answer(prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        return response.json().get("response", "")
    except Exception as e:
        return f"[ERROR] Failed to generate answer: {e}"

def generate_answer_stream(prompt: str):
    """Generator function for streaming response"""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": True
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, stream=True)
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line.decode('utf-8'))
                    if 'response' in data:
                        chunk = data['response']
                        if chunk:  # Only yield non-empty chunks
                            yield chunk
                    
                    # Check if done
                    if data.get('done', False):
                        break
                        
                except json.JSONDecodeError:
                    continue
                    
    except Exception as e:
        yield f"[ERROR] Failed to generate answer: {e}"
