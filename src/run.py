import subprocess
import time
import sys
import os
import requests

# Define paths
CAPTURE_SCRIPT = os.path.join("src", "capture.py")
PROCESS_SCRIPT = os.path.join("src", "process.py")
APP_SCRIPT = os.path.join("src", "app.py")

def is_ollama_running():
    """Checks if Ollama is already running to avoid errors."""
    try:
        requests.get("http://localhost:11434")
        return True
    except:
        return False

def main():
    print("🧠 Starting Mnemosyne System...")
    processes = []

    try:
        # 1. Start The Brain (Ollama)
        if is_ollama_running():
            print("   - ✅ Ollama is already running.")
        else:
            print("   - 🦙 Launching Ollama...")
            # We use creationflags for Windows to hide the new window if needed, 
            # but standard Popen is fine for now.
            p_ollama = subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            processes.append(p_ollama)
            time.sleep(2) # Wait for it to wake up

        # 2. Start The Eye (Capture)
        print("   - 👁️  Launching Capture Service...")
        p1 = subprocess.Popen([sys.executable, CAPTURE_SCRIPT])
        processes.append(p1)
        time.sleep(1)

        # 3. Start The Nervous System (Processing)
        print("   - ⚡ Launching Visual Cortex...")
        p2 = subprocess.Popen([sys.executable, PROCESS_SCRIPT])
        processes.append(p2)
        time.sleep(1)

        # 4. Start The Interface (Streamlit)
        print("   - 🚀 Launching UI...")
        # Streamlit needs to be run as a module or command
        p3 = subprocess.Popen(["streamlit", "run", APP_SCRIPT])
        processes.append(p3)

        print("\n✅ All systems online! Press Ctrl+C to stop.")
        
        # Keep main script alive by waiting for the UI to close
        p3.wait()

    except KeyboardInterrupt:
        print("\n🛑 Shutting down Mnemosyne...")
        for p in processes:
            try:
                p.terminate()
            except:
                pass
        print("   - Services stopped.")

if __name__ == "__main__":
    main()