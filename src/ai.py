import requests
import json

# Configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2"  # Ensure you have this pulled: `ollama pull llama3.2`

def query_brain(user_question, context_chunks):
    """
    Sends the user's question + relevant screen text to Ollama.
    """
    if not context_chunks:
        return "I couldn't find any relevant memories to answer that."

    # 1. Prepare the context (Clean up newlines)
    formatted_context = "\n---\n".join(context_chunks)

    # 2. Construct the Prompt
    prompt = f"""
    You are Mnemosyne, a personal AI assistant with access to the user's screen history.
    
    Here is the text extracted from the user's recent screens:
    ---------------------
    {formatted_context}
    ---------------------

    Based ONLY on the text above, answer this question:
    "{user_question}"

    If the answer is not in the text, say "I don't see that in your history."
    Keep the answer concise and helpful.
    """

    # 3. Send to Ollama
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload)
        if response.status_code == 200:
            return response.json().get('response', "Empty response from AI.")
        else:
            return f"❌ Error: Ollama returned status {response.status_code}"
    except Exception as e:
        return f"❌ Error connecting to Ollama: {e}\n(Make sure 'ollama serve' is running!)"