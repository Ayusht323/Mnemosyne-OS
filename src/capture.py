import time
import os
import mss
import cv2
import numpy as np
from datetime import datetime
import sys

# --- CONFIGURATION ---
# Where the images will be saved
SCREENSHOTS_DIR = "data/screenshots"
# How often to check the screen (in seconds)
INTERVAL = 2
# How similar images must be to be considered "duplicates" (0.98 = 98%)
SIMILARITY_THRESHOLD = 0.98

class ScreenSentinel:
    def __init__(self):
        print("⚙️  Initializing MSS (Screen Capture)...", flush=True)
        self.sct = mss.mss()
        
        # MONITOR SELECTION LOGIC
        # sct.monitors[0] is usually "All Monitors Combined"
        # sct.monitors[1] is usually "Primary Monitor"
        if len(self.sct.monitors) > 1:
            self.monitor = self.sct.monitors[1]
            print(f"✅ Selected Monitor 1: {self.monitor}", flush=True)
        else:
            self.monitor = self.sct.monitors[0]
            print(f"⚠️ Only 1 monitor entry found. Using Monitor 0.", flush=True)

        self.prev_frame_gray = None

    def get_timestamp(self):
        # Format: YYYYMMDD_HHMMSS
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def capture_loop(self):
        print(f"👁️  Mnemosyne Sentinel Active. Saving to: {SCREENSHOTS_DIR}", flush=True)
        print("Press Ctrl+C in the terminal to stop.", flush=True)
        
        try:
            while True:
                start_time = time.time()
                
                # 1. Grab the screen
                screenshot = self.sct.grab(self.monitor)
                
                # Convert raw pixels to an OpenCV image
                img_np = np.array(screenshot)
                frame = cv2.cvtColor(img_np, cv2.COLOR_BGRA2BGR)
                
                # 2. DUPLICATE DETECTION
                # Shrink image to 64x64 for a fast comparison check
                thumbnail = cv2.resize(frame, (64, 64))
                gray_thumbnail = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2GRAY)
                
                should_save = False
                
                # If it's the first frame ever, always save it
                if self.prev_frame_gray is None:
                    should_save = True
                else:
                    # Compare current frame vs previous frame
                    score = self._calculate_similarity(self.prev_frame_gray, gray_thumbnail)
                    
                    # If similarity is LOW (screen changed), we save.
                    if score < SIMILARITY_THRESHOLD:
                        should_save = True
                
                # 3. Save to Disk
                if should_save:
                    filename = f"{self.get_timestamp()}.jpg"
                    filepath = os.path.join(SCREENSHOTS_DIR, filename)
                    
                    # Save as JPG with quality 85 (good balance of size/quality)
                    cv2.imwrite(filepath, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                    print(f"📸 Captured: {filename}", flush=True)
                    
                    # Update "Previous Frame" memory
                    self.prev_frame_gray = gray_thumbnail
                
                # 4. Wait for the next interval
                elapsed = time.time() - start_time
                time_to_sleep = max(0, INTERVAL - elapsed)
                time.sleep(time_to_sleep)
                
        except KeyboardInterrupt:
            print("\n🛑 Sentinel stopped by user.", flush=True)
        except Exception as e:
            print(f"\n❌ Error detected: {e}", flush=True)

    def _calculate_similarity(self, img1, img2):
        """
        Compare two images. Returns 1.0 if identical, 0.0 if completely different.
        """
        res = cv2.matchTemplate(img1, img2, cv2.TM_CCOEFF_NORMED)
        return res[0][0]

if __name__ == "__main__":
    # Create the folder if it doesn't exist
    if not os.path.exists(SCREENSHOTS_DIR):
        os.makedirs(SCREENSHOTS_DIR)
    
    print("🚀 Starting Mnemosyne...", flush=True)
    sentinel = ScreenSentinel()
    sentinel.capture_loop()