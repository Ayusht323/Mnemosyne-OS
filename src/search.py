import sys
import os

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.process import VisualCortex
from src.storage import MemoryBank

def main():
    # 1. Get Query from User
    if len(sys.argv) < 2:
        print("Usage: python src/search.py 'your search query'")
        return
    
    query = sys.argv[1]
    print(f"🔍 Searching for: '{query}'...")

    # 2. Convert Query to Vector (The "Meaning")
    cortex = VisualCortex()
    query_vector = cortex.embed_text(query)
    
    if not query_vector:
        print("❌ Failed to vectorize query.")
        return

    # 3. Search Database
    bank = MemoryBank()
    results = bank.search_memory(query_vector, limit=3)

    # 4. Show Results
    print(f"\n✅ Found {len(results)} matches:\n")
    
    for i, res in enumerate(results):
        # Calculate similarity score (LanceDB returns distance, lower is better usually, 
        # but for Cosine it depends on implementation. We just show raw data for now.)
        
        # 'res' is a dictionary-like object from LanceDB
        timestamp = res['id']
        text_snippet = res['text'][:100].replace('\n', ' ')
        
        print(f"match #{i+1} | ID: {timestamp}")
        print(f"   📄 Text Content: {text_snippet}...")
        print(f"   🖼️  File: {res['file_path']}")
        print("-" * 40)

if __name__ == "__main__":
    main()