import sys
import os
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.storage import MemoryBank

STOP_WORDS = {'where', 'did', 'i', 'see', 'saw', 'what', 'was', 'when', 'how', 'who', 'the', 'a', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'is', 'are'}

def smart_searcher(query, bank, cortex):
    raw_words = re.sub(r'[^\w\s]', '', query).lower().split()
    keywords = [w for w in raw_words if w not in STOP_WORDS] or [query.lower()]
    
    text_matches = []
    try:
        rows = bank.db.open_table("memories").search().limit(2000).to_list() 
        for row in rows:
            row_text = re.sub(r'[^\w\s]', '', row.get('text', '')).lower()
            for kw in keywords:
                if kw in row_text:
                    row['type'], row['matched_keyword'] = 'TEXT', kw
                    text_matches.append(row)
                    break 
    except: pass

    visual_matches = []
    vec = cortex.embed_text(query)
    if vec:
        for r in bank.search_memory(vec, limit=15):
            r['type'] = 'VISUAL'
            visual_matches.append(r)
            
    seen = set()
    final = []
    for r in text_matches + visual_matches:
        if r['id'] not in seen:
            final.append(r)
            seen.add(r['id'])
            
    return final, keywords

if __name__ == "__main__":
    from src.process import VisualCortex
    if len(sys.argv) < 2:
        print("Usage: python src/search.py 'your search query'")
        sys.exit()
    query = sys.argv[1]
    print(f"🔍 Searching for: '{query}'...")
    res, _ = smart_searcher(query, MemoryBank(), VisualCortex())
    print(f"\n✅ Found {len(res)} matches.\n")