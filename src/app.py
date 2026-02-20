import streamlit as st
import os
import sys
import re
import requests
import json
import pandas as pd
from datetime import datetime
from PIL import Image

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.process import VisualCortex
from src.storage import MemoryBank

# --- CONFIGURATION ---
st.set_page_config(page_title="Mnemosyne", layout="wide", page_icon="🧠")
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2"

STOP_WORDS = {
    'where', 'did', 'i', 'see', 'saw', 'what', 'was', 'when', 'how', 'who', 
    'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'is', 'are'
}

@st.cache_resource
def load_cortex():
    return VisualCortex()

# --- HELPER: FIND IMAGES ---
def get_image_path(mem_id):
    base_dirs = [
        os.path.join("data", "images"),
        os.path.join("data", "screenshots")
    ]
    extensions = [".jpg", ".png", ".jpeg"]
    for folder in base_dirs:
        for ext in extensions:
            path = os.path.join(folder, f"{mem_id}{ext}")
            if os.path.exists(path): return path
    return None

# --- HELPER: CHECK LAG ---
def get_pending_count(last_db_id):
    try:
        screenshot_dir = "data/screenshots"
        if not os.path.exists(screenshot_dir): return 0
        files = sorted([f.split('.')[0] for f in os.listdir(screenshot_dir) if f.endswith(".jpg")])
        if not files: return 0
        if not last_db_id: return len(files)
        pending = [f for f in files if f > last_db_id]
        return len(pending)
    except:
        return 0

# --- HELPER: LOAD METADATA ---
def load_all_metadata(bank):
    try:
        tbl = bank.db.open_table("memories")
        rows = tbl.search().select(['id', 'text']).limit(10000).to_list()
        data = []
        for r in rows:
            try:
                dt = datetime.strptime(r['id'], "%Y%m%d_%H%M%S")
                data.append({
                    'id': r['id'],
                    'datetime': dt,
                    'hour': dt.hour,
                    'date': dt.date(),
                    'text_preview': r.get('text', '')[:50]
                })
            except: continue
        return pd.DataFrame(data)
    except: return pd.DataFrame()

# --- AI BRAIN (STRICT MODE) ---
def ask_ollama(question, context, pending_count):
    prompt = f"""
    You are Mnemosyne, a factual personal assistant.
    
    USER QUESTION: "{question}"
    
    SYSTEM STATUS: 
    - NEW IMAGES PENDING (UNREAD): {pending_count}
    
    EVIDENCE FROM PROCESSED MEMORIES:
    ---------------------
    {context}
    ---------------------

    INSTRUCTIONS:
    1. SEARCH the Evidence for the answer.
    2. IF found: Answer naturally and cite the [Time: ID].
    3. IF NOT found AND 'NEW IMAGES PENDING' > 0: 
       You MUST say: "I don't see '{question}' in your processed history yet, but I see {pending_count} new screenshots currently processing. Please ask again in a moment."
    4. IF NOT found AND 'NEW IMAGES PENDING' == 0:
       Say: "I searched your history and couldn't find any record of that."
    5. DO NOT HALLUCINATE.
    """
    
    payload = {"model": MODEL_NAME, "prompt": prompt, "stream": False, "options": {"temperature": 0.0}}
    try:
        res = requests.post(OLLAMA_URL, json=payload)
        return res.json().get('response', "Empty.") if res.status_code==200 else "Error."
    except: return "Ollama offline."

# --- SEARCH ENGINE ---
def smart_searcher(query, bank, cortex):
    raw_words = re.sub(r'[^\w\s]', '', query).lower().split()
    keywords = [w for w in raw_words if w not in STOP_WORDS]
    if not keywords: keywords = [query.lower()]

    text_matches = []
    try:
        tbl = bank.db.open_table("memories")
        rows = tbl.search().limit(2000).to_list() 
        for row in rows:
            row_text = re.sub(r'[^\w\s]', '', row.get('text', '')).lower()
            for kw in keywords:
                if kw in row_text:
                    row['type'] = 'TEXT'
                    row['matched_keyword'] = kw
                    text_matches.append(row)
                    break 
    except: pass

    visual_matches = []
    vec = cortex.embed_text(query)
    if vec:
        v_rows = bank.search_memory(vec, limit=15)
        for r in v_rows:
            r['type'] = 'VISUAL'
            visual_matches.append(r)
            
    seen = set()
    final = []
    for r in text_matches + visual_matches:
        if r['id'] not in seen: final.append(r); seen.add(r['id'])
    return final, keywords

# --- COMPONENT: TIMELINE (FIXED: ALWAYS SLIDER) ---
@st.fragment(run_every=5) 
def render_timeline():
    st.header("Rewind Your Day")
    bank = MemoryBank()
    df = load_all_metadata(bank)
    
    if df.empty:
        st.warning("No memories yet.")
        return

    dates = sorted(df['date'].unique(), reverse=True)
    selected_date = st.selectbox("Select Date:", dates)
    day_data = df[df['date'] == selected_date].sort_values('datetime', ascending=True).reset_index(drop=True)
    
    if day_data.empty: return

    count = len(day_data)
    
    # --- SLIDER LOGIC ---
    if "timeline_idx" not in st.session_state: st.session_state.timeline_idx = count - 1
    if st.session_state.timeline_idx >= count: st.session_state.timeline_idx = count - 1

    selected_idx = st.slider("Scrub Time", 0, count - 1, st.session_state.timeline_idx, key="timeline_slider")
    st.session_state.timeline_idx = selected_idx

    memory_row = day_data.iloc[st.session_state.timeline_idx]
    mem_id = memory_row['id']
    time_str = memory_row['datetime'].strftime("%H:%M:%S")

    col_img, col_info = st.columns([3, 1])
    with col_img:
        img_path = get_image_path(mem_id)
        if img_path: 
            st.image(img_path, caption=f"Snapshot: {time_str}", width="stretch")
        else: st.warning("File Missing")
    
    with col_info:
        st.metric("Time", time_str)
        c1, c2 = st.columns(2)
        if c1.button("⏪"): 
            st.session_state.timeline_idx = max(0, st.session_state.timeline_idx - 1); st.rerun()
        if c2.button("⏩"): 
            st.session_state.timeline_idx = min(count - 1, st.session_state.timeline_idx + 1); st.rerun()
        
        # --- HERE IS THE CONTENT BOX YOU MISSED ---
        # It is now available regardless of how many images you have.
        st.subheader("Extracted Content:")
        st.text_area("OCR Output", memory_row.get('text_preview', ''), height=300, label_visibility="collapsed")

# --- COMPONENT: DASHBOARD ---
@st.fragment(run_every=10)
def render_dashboard():
    st.header("Productivity Analytics")
    bank = MemoryBank()
    df = load_all_metadata(bank)
    if not df.empty:
        hourly = df['hour'].value_counts().sort_index().reindex(range(24), fill_value=0)
        st.bar_chart(hourly)
        st.dataframe(df[['datetime', 'text_preview']].sort_values('datetime', ascending=False))
    else: st.info("No data.")

# --- MAIN APP LAYOUT ---
st.title("🧠 Mnemosyne OS")
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Search", "💬 Chat", "⏳ Time Travel", "📊 Dashboard"])

with tab1:
    q = st.text_input("Find Screenshot:", placeholder="jarvis")
    if q:
        bank = MemoryBank()
        cortex = load_cortex()
        results, kws = smart_searcher(q, bank, cortex)
        if results:
            st.success(f"Found {len(results)} items")
            cols = st.columns(3)
            for idx, item in enumerate(results):
                with cols[idx % 3]:
                    with st.container(border=True):
                        img_path = get_image_path(item['id'])
                        if img_path: 
                            st.image(img_path, width="stretch")
                        if item['type']=='TEXT': st.success(f"Text: {item.get('matched_keyword')}")
                        else: st.info("Visual Match")
        else: st.warning("No matches")

with tab2:
    if "chat_history" not in st.session_state: st.session_state.chat_history = []
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])
    
    if prompt := st.chat_input("Ask your brain..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            st.markdown("Thinking...")
            bank = MemoryBank()
            cortex = load_cortex()
            
            items, _ = smart_searcher(prompt, bank, cortex)
            context = "\n".join([f"[Time: {i['id']}] {i.get('text','')}" for i in items[:10]])
            
            last_db_id = None
            if items: last_db_id = sorted([i['id'] for i in items])[-1]
            if not last_db_id:
                df = load_all_metadata(bank)
                if not df.empty: last_db_id = df['id'].max()

            pending = get_pending_count(last_db_id)
            resp = ask_ollama(prompt, context, pending)
            
            st.markdown(resp)
            st.session_state.chat_history.append({"role": "assistant", "content": resp})

with tab3:
    render_timeline()

with tab4:
    render_dashboard()