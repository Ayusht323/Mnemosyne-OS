import os
import sys
import shutil

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.storage import MemoryBank

def clean_ghosts():
    print("🧹 Starting Cleanup Protocol...")
    bank = MemoryBank()
    
    try:
        tbl = bank.db.open_table("memories")
        all_rows = tbl.search().limit(100000).to_list()
        
        valid_rows = []
        ghost_count = 0
        
        print(f"🔍 Scanning {len(all_rows)} memories...")
        
        for row in all_rows:
            # Check if file exists
            if os.path.exists(row['file_path']):
                valid_rows.append(row)
            else:
                print(f"   ❌ Found Ghost: {row['id']} (File missing)")
                ghost_count += 1
        
        if ghost_count > 0:
            print(f"⚠️ Found {ghost_count} broken memories. Removing them...")
            
            # LanceDB is append-only, so the cleanest way to "delete" 
            # is to overwrite the table with ONLY the valid data.
            bank.db.drop_table("memories")
            bank.create_table() # Create fresh
            
            if valid_rows:
                bank.add_memories(valid_rows) # Add back the good ones
                print(f"✅ Restored {len(valid_rows)} valid memories.")
            else:
                print("⚠️ Database is now empty (no valid images found).")
                
        else:
            print("✨ Database is already clean!")

    except Exception as e:
        print(f"❌ Error during cleanup: {e}")

if __name__ == "__main__":
    clean_ghosts()