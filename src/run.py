import subprocess
import time
import sys
import os
import requests

CAPTURE_SCRIPT = os.path.join("src", "capture.py")
PROCESS_SCRIPT = os.path.join("src", "process.py")
APP_SCRIPT = os.path.join("src", "app.py")

def is_ollama_running():
    try:
        requests.get("http://localhost:11434")
        return True
    except:
        return False

def main():
    print("🧠 Starting Mnemosyne System...")
    os.makedirs("data/screenshots", exist_ok=True)
    os.makedirs("data/debug_views", exist_ok=True)
    
    processes = []
    try:
        if is_ollama_running():
            print("   - ✅ Ollama is already running.")
        else:
            print("   - 🦙 Launching Ollama...")
            p_ollama = subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            processes.append(p_ollama)
            time.sleep(2)

        print("   - ⚡ Launching Visual Cortex...")
        p2 = subprocess.Popen([sys.executable, PROCESS_SCRIPT])
        processes.append(p2)
        time.sleep(1)

        print("   - 🚀 Launching UI...")
        p3 = subprocess.Popen([sys.executable, "-m", "streamlit", "run", APP_SCRIPT])
        processes.append(p3)

        print("\n✅ All systems online! Press Ctrl+C to stop.")
        print("💡 Note: The Screen Sentinel (Capture) is currently paused.")
        print("   -> Go to the Web UI (http://localhost:8501) and click 'Start Sentinel' in the sidebar to begin capturing.")
        
        p3.wait()

    except KeyboardInterrupt:
        print("\n🛑 Shutting down Mnemosyne...")
        for p in processes:
            try: p.terminate()
            except: pass
        print("   - Services stopped.")

if __name__ == "__main__":
    main()