# app.py
import os
import json
from flask import (Flask, render_template, request,
                   redirect, url_for, flash, session, jsonify, Response)

# ---------------- import modul internal ----------------
from vectorstore import build_index, similarity_search
from rag.generator import generate_answer, generate_answer_stream
from db import save_chat_history

app = Flask(__name__)
app.secret_key = "change_me_in_production"

ALLOWED_EXT = {'.pdf', '.txt', '.csv'}

# ---------------- ROUTES ----------------

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    # Initialize chat history in session if not exists
    if 'chat_history' not in session:
        session['chat_history'] = []
    
    if request.method == 'POST':
        user_input = request.form.get('message', '').strip()
        if user_input:
            contexts = similarity_search(user_input, k=3)
            prompt = (
                "Anda adalah asisten berbahasa Indonesia.\n"
                "Jawab dalam bahasa Indonesia, gunakan tag HTML seperti <b>, <ol>, <li>, <br> untuk penomoran dan penekanan.\n"
                "Konteks:\n" + "\n".join(contexts) +
                "\n\nPertanyaan: " + user_input + "\nJawab:"
            )
            response = generate_answer(prompt)
            save_chat_history("anon", user_input, response)
            
            # Add to session history
            session['chat_history'].append({
                'user': user_input,
                'bot': response
            })
            session.modified = True
            
            return redirect(url_for('chat'))

    return render_template('chat.html', chat_history=session.get('chat_history', []))

@app.route('/chat-stream', methods=['POST'])
def chat_stream():
    user_input = request.json.get('message', '').strip()
    
    if not user_input:
        return jsonify({'error': 'No message provided'}), 400
    
    def generate():
        try:
            contexts = similarity_search(user_input, k=3)
            prompt = (
                "Anda adalah asisten berbahasa Indonesia.\n"
                "Jawab dalam bahasa Indonesia, gunakan tag HTML seperti <b>, <ol>, <li>, <br> untuk penomoran dan penekanan.\n"
                "Konteks:\n" + "\n".join(contexts) +
                "\n\nPertanyaan: " + user_input + "\nJawab:"
            )
            
            full_response = ""
            for chunk in generate_answer_stream(prompt):
                full_response += chunk
                yield f"data: {json.dumps({'chunk': chunk, 'done': False})}\n\n"
            
            # Save to database only (not session here due to context issue)
            save_chat_history("anon", user_input, full_response)
            
            # Send done signal with full response for session saving
            yield f"data: {json.dumps({'chunk': '', 'done': True, 'full_response': full_response, 'user_input': user_input})}\n\n"
            
        except Exception as e:
            error_msg = f"[ERROR] Terjadi kesalahan: {str(e)}"
            yield f"data: {json.dumps({'chunk': error_msg, 'done': True, 'error': True})}\n\n"
    
    return Response(generate(), mimetype='text/plain', headers={
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'Access-Control-Allow-Origin': '*'
    })

@app.route('/save-to-session', methods=['POST'])
def save_to_session():
    """Endpoint untuk menyimpan chat ke session setelah streaming selesai"""
    data = request.json
    user_input = data.get('user_input', '')
    bot_response = data.get('bot_response', '')
    
    # Initialize chat history in session if not exists
    if 'chat_history' not in session:
        session['chat_history'] = []
    
    # Add to session history
    session['chat_history'].append({
        'user': user_input,
        'bot': bot_response
    })
    session.modified = True
    
    return jsonify({'status': 'success'})

@app.route('/clear-chat', methods=['POST'])
def clear_chat():
    session['chat_history'] = []
    session.modified = True
    return jsonify({'status': 'success'})

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        files = request.files.getlist('file')
        saved = 0
        os.makedirs('pdf_files', exist_ok=True)

        for f in files:
            if f and f.filename:
                ext = os.path.splitext(f.filename)[1].lower()
                if ext in ALLOWED_EXT:
                    f.save(os.path.join('pdf_files', f.filename))
                    saved += 1
                else:
                    flash(f"{f.filename} bukan file yang diizinkan", "error")

        if saved:
            build_index('pdf_files')
            flash(f"{saved} file di-upload & FAISS diperbarui", "success")
        else:
            flash("Tidak ada file yang valid", "error")

        return redirect(url_for('upload'))

    return render_template('upload.html')

# ---------------- RUN ----------------
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)