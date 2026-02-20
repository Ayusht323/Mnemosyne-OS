import os
import time
import torch
import easyocr
import logging
import warnings
import cv2
import numpy as np
from PIL import Image
from transformers import SiglipProcessor, SiglipModel
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.storage import MemoryBank

# --- SILENCE WARNINGS ---
warnings.filterwarnings("ignore", message=".*pin_memory.*")
logging.getLogger("easyocr").setLevel(logging.ERROR)

# --- CONFIGURATION ---
DEVICE = "cpu"
SCREENSHOTS_DIR = "data/screenshots"
DEBUG_DIR = "data/debug_views"
OCR_LOG = "data/ocr_debug.log"
CHECK_INTERVAL = 0.1     
FILE_COOLDOWN = 0.5      

class VisualCortex:
    def __init__(self):
        print("🧠 Initializing Visual Cortex...", flush=True)
        
        if not os.path.exists(DEBUG_DIR):
            os.makedirs(DEBUG_DIR)
        
        # Clear log
        with open(OCR_LOG, "w", encoding="utf-8") as f:
            f.write("--- OCR DEBUG LOG ---\n")

        print("   - Loading OCR Engine (EasyOCR)...", flush=True)
        self.reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        
        print("   - Loading Vision Model...", flush=True)
        self.model_name = "google/siglip-so400m-patch14-384"
        self.processor = SiglipProcessor.from_pretrained(self.model_name)
        self.model = SiglipModel.from_pretrained(self.model_name).to(DEVICE)
        print("✅ Visual Cortex Online.", flush=True)

    def process_tile(self, img_tile):
        """
        Upscales and normalizes a single tile.
        """
        try:
            # 1. 3x Upscale (Lanczos)
            scale = 3.0
            width = int(img_tile.shape[1] * scale)
            height = int(img_tile.shape[0] * scale)
            upscaled = cv2.resize(img_tile, (width, height), interpolation=cv2.INTER_LANCZOS4)
            
            # 2. Grayscale
            gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
            
            # 3. Linear Contrast Stretch (Safe Normalization)
            # Makes darks darker and lights lighter without thresholding
            norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
            
            return norm
        except:
            return img_tile

    def scan_mosaic(self, image_path, file_id):
        """
        GUILLOTINE + 2x3 GRID MOSAIC
        """
        try:
            full_img = cv2.imread(image_path)
            if full_img is None: return ""
            
            h, w = full_img.shape[:2]
            
            # --- STEP 1: SEPARATE HEADER ---
            # Cut at 160px to clear the "chips" (Restaurants, Hotels, etc)
            header_h = 160
            img_header = full_img[0:header_h, 0:w]
            img_body = full_img[header_h:h, 0:w]
            
            # --- STEP 2: TILE THE BODY (2x3 Grid) ---
            # Splitting prevents EasyOCR from shrinking the image
            body_h, body_w = img_body.shape[:2]
            
            # Define grid (2 rows, 3 columns)
            rows = 2
            cols = 3
            step_h = body_h // rows
            step_w = body_w // cols
            overlap = 100 # Safety overlap
            
            body_text = []
            
            with open(OCR_LOG, "a", encoding="utf-8") as f:
                f.write(f"\n--- Processing {file_id} ---\n")

            tile_idx = 0
            for r in range(rows):
                for c in range(cols):
                    # Calc coords
                    y1 = max(0, r * step_h - overlap)
                    y2 = min(body_h, (r + 1) * step_h + overlap)
                    x1 = max(0, c * step_w - overlap)
                    x2 = min(body_w, (c + 1) * step_w + overlap)
                    
                    tile = img_body[y1:y2, x1:x2]
                    
                    # Process Tile
                    processed_tile = self.process_tile(tile)
                    
                    # Save CENTER Tile (Row 1, Col 1) as Debug Sample
                    if r == 0 and c == 1:
                        debug_path = os.path.join(DEBUG_DIR, f"{file_id}_tile_debug.jpg")
                        cv2.imwrite(debug_path, processed_tile)

                    # Read Tile
                    # canvas_size=2560 is plenty for a TILE (won't shrink)
                    res = self.reader.readtext(
                        processed_tile, 
                        detail=0,
                        paragraph=False,
                        batch_size=4,
                        canvas_size=2560,   
                        text_threshold=0.3, 
                        low_text=0.25,
                        mag_ratio=1.0 # Manual zoom active
                    )
                    
                    if res:
                        body_text.extend(res)
                        with open(OCR_LOG, "a", encoding="utf-8") as f:
                            f.write(f"Tile {tile_idx}: {res}\n")
                    
                    tile_idx += 1

            # --- STEP 3: READ HEADER ---
            res_header = self.reader.readtext(img_header, detail=0)
            
            # --- STEP 4: COMBINE ---
            # Deduplicate while preserving order (mostly)
            all_text = list(dict.fromkeys(body_text + res_header))
            
            return " ".join(all_text)

        except Exception as e:
            print(f"⚠️ Scan Error: {e}")
            with open(OCR_LOG, "a", encoding="utf-8") as f:
                f.write(f"ERROR: {e}\n")
            return ""

    def embed_text(self, text_query):
        try:
            inputs = self.processor(text=[text_query], padding="max_length", return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                outputs = self.model.get_text_features(**inputs)
            return outputs[0].tolist()
        except Exception as e:
            print(f"❌ Error embedding text: {e}")
            return None

    def process_image(self, image_path):
        if not os.path.exists(image_path): return None
        try:
            try:
                img_check = Image.open(image_path)
                img_check.verify()
            except: return None 

            file_id = os.path.basename(image_path).split(".")[0]
            
            # --- MOSAIC SCAN ---
            extracted_text = self.scan_mosaic(image_path, file_id)
            
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)
            
            return {
                "timestamp": file_id,
                "text": extracted_text,
                "vector": outputs[0].tolist()
            }
        except Exception as e:
            print(f"⚠️ Error processing {os.path.basename(image_path)}: {e}")
            return None

def get_all_processed_ids(bank):
    try:
        tables = bank.db.table_names() if hasattr(bank.db, 'table_names') else bank.db.list_tables()
        if "memories" not in tables: return set()
        tbl = bank.db.open_table("memories")
        results = tbl.search().select(["id"]).limit(1000000).to_list()
        processed = {r['id'] for r in results}
        print(f"📚 Loaded {len(processed)} existing memories.")
        return processed
    except Exception as e:
        print(f"⚠️ Could not load existing DB: {e}")
        return set()

if __name__ == "__main__":
    cortex = VisualCortex()
    bank = MemoryBank()
    processed_ids = get_all_processed_ids(bank)
    
    print(f"🚀 Visual Cortex Service Started. Watching {SCREENSHOTS_DIR}...", flush=True)
    print("⚡ Mode: GRID ZOOM (Mosaic Tiling)", flush=True)

    try:
        while True:
            if not os.path.exists(SCREENSHOTS_DIR):
                time.sleep(CHECK_INTERVAL)
                continue
            
            all_files = [f for f in os.listdir(SCREENSHOTS_DIR) if f.endswith(".jpg")]
            pending_files = []
            for f in all_files:
                fid = f.split(".")[0]
                if fid not in processed_ids:
                    pending_files.append(f)
            
            if not pending_files:
                time.sleep(CHECK_INTERVAL)
                continue
            
            pending_files.sort(reverse=True)
            target_filename = pending_files[0]
            target_id = target_filename.split(".")[0]
            image_path = os.path.join(SCREENSHOTS_DIR, target_filename)

            if time.time() - os.path.getmtime(image_path) < FILE_COOLDOWN:
                time.sleep(0.1)
                continue
            
            print(f"⚡ Processing Newest: {target_filename}...", end="", flush=True)
            data = cortex.process_image(image_path)
            
            if data:
                bank.store_memory(data)
                processed_ids.add(target_id)
                print(" Done ✅")
            else:
                print(" Skipped (Error) ⚠️")
                processed_ids.add(target_id)
            
    except KeyboardInterrupt:
        print("\n🛑 Service stopped.")