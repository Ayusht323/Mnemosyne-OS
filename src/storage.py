import lancedb
import os
import time
from datetime import datetime

# --- CONFIGURATION ---
DB_PATH = "data/lancedb"
TABLE_NAME = "memories"
# SigLIP vector size is usually 1152 for large or 768 for base. 
# The one we used (siglip-so400m-patch14-384) outputs 1152 dimensions.
VECTOR_DIM = 1152 

class MemoryBank:
    def __init__(self):
        print(f"💾 Initializing LanceDB at {DB_PATH}...", flush=True)
        
        # Ensure the directory exists
        if not os.path.exists(DB_PATH):
            os.makedirs(DB_PATH)
            
        self.db = lancedb.connect(DB_PATH)
        self.table = self._get_or_create_table()

    def _get_or_create_table(self):
        # Check if table exists
        if TABLE_NAME in self.db.table_names():
            return self.db.open_table(TABLE_NAME)
        
        print("   - Creating new 'memories' table...", flush=True)
        # We define a minimal schema by inserting a dummy record and deleting it, 
        # or letting LanceDB infer it from the first insertion.
        # We'll let it infer from the first insertion to keep it simple.
        return None

    def store_memory(self, memory_data):
        """
        Input: dict { "timestamp": "...", "text": "...", "vector": [...] }
        """
        if not memory_data:
            return

        # Prepare the record
        record = {
            "id": memory_data["timestamp"],
            "timestamp": time.time(), # Unix timestamp for sorting
            "file_path": f"data/screenshots/{memory_data['timestamp']}.jpg",
            "text": memory_data["text"],
            "vector": memory_data["vector"]
        }

        try:
            if TABLE_NAME not in self.db.table_names():
                # First time creation
                self.table = self.db.create_table(TABLE_NAME, data=[record])
                print(f"✨ Created table and stored first memory: {record['id']}")
            else:
                self.table = self.db.open_table(TABLE_NAME)
                self.table.add([record])
                print(f"💾 Stored memory: {record['id']}")
                
        except Exception as e:
            print(f"❌ Database Error: {e}")

    def search_memory(self, query_vector, limit=5):
        """
        Input: 1152-dim vector (from the user's query text)
        Output: List of matching records
        """
        if TABLE_NAME not in self.db.table_names():
            print("⚠️ Memory bank is empty.")
            return []
        
        self.table = self.db.open_table(TABLE_NAME)
        
        # LanceDB Vector Search
        results = self.table.search(query_vector) \
            .limit(limit) \
            .to_list()
            
        return results

    def get_stats(self):
        if TABLE_NAME not in self.db.table_names():
            return "Empty"
        self.table = self.db.open_table(TABLE_NAME)
        return f"Total Memories: {len(self.table)}"

if __name__ == "__main__":
    # TEST: Initialize the DB
    bank = MemoryBank()
    print(f"📊 Status: {bank.get_stats()}")