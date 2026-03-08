# 🧠 Mnemosyne

**An AI-powered personal memory system that captures, understands, and lets you search through your screen history.**

Mnemosyne automatically captures your screen activity, extracts text and visual understanding through AI, and stores everything in a searchable vector database. Instantly recall what you saw on your computer—without frantically scrolling through screenshots or digging through file browsers.

---

## ✨ Features

- **Automatic Screenshot Capture** — Continuously monitors your screen with configurable intervals (2-second default)
- **AI-Powered Understanding** — Uses SigLIP vision model to understand image content and EasyOCR for text extraction
- **Vector Search** — Semantic search across your memory—find "where did I see that design?" instantly
- **Chat with Your Memory** — Ask questions about what you've seen, powered by Llama 3.2
- **Timeline Exploration** — Browse through your visual history chronologically
- **Dashboard Analytics** — See statistics about your activity (screenshots captured, text extracted, etc.)
- **Smart Deduplication** — Avoids storing near-duplicate screenshots to save space

---

## 🎯 Why Mnemosyne?

Forget digging through folders or using crude file search. Mnemosyne understands the *meaning* of what's on your screen:

- **Semantic Search**: "Show me where I saw the sidebar layout" finds relevant screenshots even if the text doesn't match exactly
- **Context-Aware Results**: AI understands visual hierarchies, objects, and concepts
- **Question Answering**: "What was I reading about machine learning?" gets answered by checking your recent screen history
- **Zero Manual Work**: Runs in the background—just search when you need to remember

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.9+**
- **Ollama** (for the language model) — Download from [ollama.com](https://ollama.com)
- Windows, macOS, or Linux

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Mnemosyne.git
   cd Mnemosyne
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Ollama**
   - Install [Ollama](https://ollama.com)
   - Pull the Llama 3.2 model:
     ```bash
     ollama pull llama3.2
     ```
   - Start Ollama (it will run on `http://localhost:11434` by default)

### Running Mnemosyne

Launch all services with one command:

```bash
python src/run.py
```

This starts:
- ✅ **Ollama** — The AI "brain" (if not already running)
- 👁️ **Capture Service** — Monitors and saves screenshots
- ⚡ **Visual Cortex** — Processes images with AI
- 🚀 **Web UI** — Streamlit dashboard (opens at `http://localhost:8501`)

---

## 📖 Usage

### Via Web UI (Main Interface)

Once running, access the Streamlit interface at `http://localhost:8501`

#### Search Tab
```
Find Screenshot: "jarvis"
→ Returns all screenshots matching "jarvis" (text or visual content)
```

#### Chat Tab
```
Ask your brain: "What was I working on yesterday?"
→ AI analyzes your screen history and answers based on what it saw
```

#### Time Travel Tab
- Browse screenshots chronologically
- View hourly/daily activity trends
- Inspect extracted text for any screenshot

#### Dashboard Tab
- Total screenshots captured
- Text extraction statistics
- Activity heatmap

### Via Command Line

Search from the terminal:
```bash
python src/search.py "your search query"
```

Example:
```bash
python src/search.py "where did I see that error message"
```

---

## 🏗️ Architecture

Mnemosyne is built around four core components:

| Component | File | Role |
|-----------|------|------|
| **Screen Sentinel** | `capture.py` | Captures screenshots, detects changes via frame similarity |
| **Visual Cortex** | `process.py` | Extracts text (OCR) and visual embeddings (SigLIP model) |
| **Memory Bank** | `storage.py` | Vector database using LanceDB for semantic search |
| **User Interface** | `app.py` | Streamlit dashboard for search, chat, and exploration |

**Data Flow:**
```
Screen → Capture → Process (OCR + Vision Model) → Vector Embeddings → LanceDB → Search/Chat UI
         (2s interval)  (SigLIP, EasyOCR)      (1152-dim vectors)
```

---

## ⚙️ Configuration

Edit these values in source files to customize behavior:

### Screenshot Capture ([capture.py](src/capture.py))
```python
INTERVAL = 2              # Capture every 2 seconds
SIMILARITY_THRESHOLD = 0.98  # Skip frames that are 98%+ similar to previous
```

### AI Models ([process.py](src/process.py))
```python
DEVICE = "cpu"            # Change to "cuda" for GPU acceleration
MODEL_NAME = "google/siglip-so400m-patch14-384"  # Vision model
```

### Language Model ([app.py](src/app.py))
```python
MODEL_NAME = "llama3.2"   # Change to other Ollama-supported models
OLLAMA_URL = "http://localhost:11434/api/generate"
```

---

## 📊 Data Storage

- **Screenshots**: `data/screenshots/` (YYYYMMDD_HHMMSS.jpg format)
- **Vector Database**: `data/lancedb/` (LanceDB storage)
- **Logs**: `data/ocr_debug.log` (OCR extraction logs)
- **Debug Views**: `data/debug_views/` (Annotated images for troubleshooting)

All data is stored locally on your machine. Nothing is uploaded to external servers.

---

## 🤝 Contributing

Contributions are welcome! Here's how to help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Test thoroughly
5. Commit with clear messages (`git commit -m 'Add amazing feature'`)
6. Push to your branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Development Setup
```bash
git clone your-fork-url
cd Mnemosyne
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 💬 Getting Help

- **Issues**: Report bugs on [GitHub Issues](https://github.com/yourusername/Mnemosyne/issues)
- **Discussions**: Ask questions on [GitHub Discussions](https://github.com/yourusername/Mnemosyne/discussions)
- **Documentation**: Check the [docs/](docs/) folder for deeper guides

### Common Issues

**Ollama not connecting?**
- Ensure `ollama serve` is running in another terminal
- Check: `curl http://localhost:11434`

**Out of memory during processing?**
- Reduce screenshot resolution in `capture.py`
- Switch to CPU-only mode or reduce model size

**Slow search results?**
- Check that LanceDB is indexing properly
- Consider increasing `INTERVAL` in `capture.py` to capture fewer screenshots

---

## 📜 License

This project is licensed under the [MIT License](LICENSE) — feel free to use it, modify it, and distribute it.

---

## 🙏 Acknowledgments

- [SigLIP](https://github.com/google-research/big_vision) for vision embeddings
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) for text extraction
- [LanceDB](https://lancedb.com) for vector database
- [Streamlit](https://streamlit.io) for the fantastic UI framework
- [Ollama](https://ollama.ai) for accessible local LLMs
- [Mnemosyne](https://en.wikipedia.org/wiki/Mnemosyne) (goddess of memory) for eternal inspiration

---

**Made with 🧠 to remember everything you see.**
