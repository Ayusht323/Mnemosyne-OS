import lancedb
import os
import time
import pyarrow as pa

# --- CONFIGURATION ---
DB_PATH = "data/lancedb"
TABLE_NAME = "memories"
VECTOR_DIM = 1152 

class MemoryBank:
    def __init__(self):
        print(f"💾 Initializing LanceDB at {DB_PATH}...", flush=True)
        if not os.path.exists(DB_PATH):
            os.makedirs(DB_PATH)
            
        self.db = lancedb.connect(DB_PATH)
        self.schema = pa.schema([
            pa.field("id", pa.string()),
            pa.field("text", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), VECTOR_DIM)),
            pa.field("timestamp", pa.float64()),
            pa.field("file_path", pa.string())
        ])
        
        if TABLE_NAME not in self.db.table_names():
            print("   - Creating new 'memories' table...", flush=True)
            self.db.create_table(TABLE_NAME, schema=self.schema)

    def store_memory(self, memory_data):
        if not memory_data: return
        
        record = {
            "id": memory_data["timestamp"],
            "timestamp": time.time(),
            "file_path": f"data/screenshots/{memory_data['timestamp']}.jpg",
            "text": memory_data["text"],
            "vector": memory_data["vector"]
        }

        try:
            tbl = self.db.open_table(TABLE_NAME)
            tbl.add([record])
        except Exception as e:
            print(f"❌ Database Error: {e}")

    def search_memory(self, query_vector, limit=15):
        if TABLE_NAME not in self.db.table_names(): return []
        try:
            tbl = self.db.open_table(TABLE_NAME)
            return tbl.search(query_vector).limit(limit).to_list()
        except Exception: return []

    def get_stats(self):
        if TABLE_NAME not in self.db.table_names(): return "Empty"
        return f"Total Memories: {len(self.db.open_table(TABLE_NAME))}"