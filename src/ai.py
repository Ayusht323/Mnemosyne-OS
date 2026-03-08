import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2"

def ask_ollama(question, context_chunks, pending_count):
    if not context_chunks and pending_count == 0:
        return "I searched your history and couldn't find any record of that."
        
    formatted_context = "\n---\n".join(context_chunks)

    prompt = f"""You are Mnemosyne, a factual personal assistant.
USER QUESTION: "{question}"
SYSTEM STATUS: NEW IMAGES PENDING: {pending_count}
EVIDENCE:
{formatted_context}
INSTRUCTIONS:
1. SEARCH the Evidence. Answer naturally and cite the [Time: ID].
2. IF NOT found AND PENDING > 0: Say: "I don't see '{question}' in your processed history yet, but {pending_count} new screenshots are processing."
3. IF NOT found AND PENDING == 0: Say: "I searched your history and couldn't find any record of that."
4. DO NOT HALLUCINATE."""

    try:
        res = requests.post(OLLAMA_URL, json={"model": MODEL_NAME, "prompt": prompt, "stream": False, "options": {"temperature": 0.0}})
        if res.status_code == 200:
            return res.json().get('response', "Empty response from AI.")
        return f"❌ Error: Ollama returned status {res.status_code}"
    except Exception as e:
        return f"❌ Error connecting to Ollama: {e}\n(Make sure 'ollama serve' is running!)"