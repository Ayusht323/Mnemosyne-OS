import os
import time
import torch
import logging
import warnings
import cv2
import numpy as np
from PIL import Image
from transformers import SiglipProcessor, SiglipModel
import sys

# PaddleOCR Import
from paddleocr import PaddleOCR

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.storage import MemoryBank

# --- SILENCE WARNINGS ---
warnings.filterwarnings("ignore")
logging.getLogger("ppocr").setLevel(logging.ERROR)

# --- CONFIGURATION ---
DEVICE = "cpu"
SCREENSHOTS_DIR = "data/screenshots"
DEBUG_DIR = "data/debug_views"
OCR_LOG = "data/ocr_debug.log"
CHECK_INTERVAL = 0.1
FILE_COOLDOWN = 0.5


class OCRPostProcessor:
    """
    Context-aware OCR correction using local Ollama (llama3.2).

    WHY NOT SIMPLE REGEX / WORD SUBSTITUTION:
        A regex that replaces "Al" → "AI" will also destroy:
          - Names:     "Al Gore", "Al Smith"
          - Chemistry: "Al" (Aluminium element symbol)
          - Latin:     "et al."
          - Variables: code with a variable named `Al`
        You can't fix OCR errors without understanding context.
        Only an LLM can do that correctly.

    HOW IT WORKS:
        The full raw OCR text is sent to Ollama in ONE call.
        The prompt instructs it to:
          1. Fix character-level OCR errors (I↔l, 0↔O, Rs→₹, etc.)
             ONLY when the surrounding context makes the correction
             unambiguous and certain.
          2. Leave everything else exactly as-is — no paraphrasing,
             no rewording, no summarising.
          3. Return the corrected text and nothing else.

        Example:
            Input:  "embedded Al processing, contact Al Smith, Try Premium for 0"
            Output: "embedded AI processing, contact Al Smith, Try Premium for ₹0"
                              ↑ fixed                ↑ kept              ↑ fixed

    SAFE LAYER (always runs, zero ambiguity):
        Only unicode lookalike characters that have NO legitimate use
        as the wrong character — purely encoding artifacts:
          \u2018 → '   (left curly quote, OCR never needs this)
          \u2019 → '   (right curly quote)
          \u201c → "   (left double quote)
          \u201d → "   (right double quote)
    """

    # Only truly unambiguous unicode encoding artifacts — NOT word substitutions
    UNICODE_FIXES = {
        "\u2018": "'",    # left  curly '
        "\u2019": "'",    # right curly '
        "\u201c": '"',    # left  curly "
        "\u201d": '"',    # right curly "
        "\u2014": " -- ", # em dash
        "\u2013": "-",    # en dash
        "\u2026": "...",  # ellipsis character
        "\u00a0": " ",    # non-breaking space
    }

    # The core prompt sent to Ollama with the full OCR text
    CORRECTION_PROMPT = """\
You are an OCR post-correction engine. The text below was extracted from a \
screenshot using OCR software. OCR makes predictable character-level mistakes \
due to visual similarity between glyphs. Your job is to fix those mistakes.

UNDERSTAND THESE OCR ERROR PATTERNS:
1. I/l confusion — capital letter "I" and lowercase "l" look identical in many fonts.
   Decide which is correct purely from the surrounding words and context.
2. 0/O confusion — digit "0" and letter "O" look identical in many fonts.
   Decide which is correct from context.
3. Currency symbols — OCR often drops or garbles currency glyphs (₹, €, £, ¥, $).
   If a number appears where a price clearly belongs, restore the correct symbol.
4. Merged words — OCR sometimes runs two words together with no space.
   Split them if the surrounding context makes it obvious.
5. Double-reads — OCR sometimes reads the same visual element twice in a row
   (e.g. a logo captured as both an icon and text). Collapse to the correct single form.

HOW TO DECIDE:
- Read the FULL surrounding context before changing anything.
- Only fix something when you are certain it is an OCR error.
- If a word could legitimately be either option, leave it exactly as-is.
- Never rephrase, reorder, summarise, or paraphrase — only fix character errors.
- Preserve all special characters exactly: © ™ ® § numbers punctuation emoji.

Return ONLY the corrected text. No explanation. No preamble. No markdown.

TEXT:
{text}"""

    def __init__(self, use_llm: bool = True, ollama_url: str = "http://localhost:11434"):
        """
        Args:
            use_llm:     Enable Ollama correction. Default True.
                         Set False to skip LLM (raw OCR only, faster).
            ollama_url:  Ollama API endpoint — same as used in ai.py.
        """
        self.use_llm = use_llm
        self.ollama_url = ollama_url

    # Special symbols the LLM tends to silently drop.
    # These are restored after LLM correction if they existed in original.
    PRESERVE_SYMBOLS = ['©', '™', '®', '€', '£', '¥', '₹', '°', '±', '×', '÷', '≤', '≥']

    def _restore_dropped_symbols(self, original: str, corrected: str) -> str:
        """
        After LLM correction, check if any special symbols that existed in
        the original text were silently deleted. If so, reinsert them at the
        closest matching position using difflib.

        This is a safety net — the LLM prompt says to preserve these, but
        small models like llama3.2 sometimes drop them anyway.
        """
        import difflib

        for symbol in self.PRESERVE_SYMBOLS:
            original_count  = original.count(symbol)
            corrected_count = corrected.count(symbol)

            if original_count > corrected_count:
                # Symbol was dropped — find where it was and reinsert
                missing = original_count - corrected_count

                # Find all positions in original
                orig_words = original.split()
                corr_words = corrected.split()

                for _ in range(missing):
                    # Find the index of the symbol token in original
                    try:
                        sym_idx = next(
                            i for i, w in enumerate(orig_words) if symbol in w
                        )
                    except StopIteration:
                        break

                    # Find the nearest anchor word before the symbol in original
                    anchor = None
                    for w in reversed(orig_words[:sym_idx]):
                        if symbol not in w and w.strip():
                            anchor = w.strip('.,;:()[]')
                            break

                    # Find that anchor in corrected and insert symbol after it
                    if anchor:
                        for j, cw in enumerate(corr_words):
                            if anchor in cw and symbol not in cw:
                                corr_words.insert(j + 1, symbol)
                                break
                    else:
                        # Fallback: append at same relative position
                        pos = min(sym_idx, len(corr_words))
                        corr_words.insert(pos, symbol)

                    # Remove the symbol from orig_words so we don't match it again
                    orig_words[sym_idx] = orig_words[sym_idx].replace(symbol, '')

                corrected = " ".join(corr_words)

        return corrected

    def _unicode_fix(self, text: str) -> str:
        """Replace unicode encoding artifacts — these are never ambiguous."""
        for bad, good in self.UNICODE_FIXES.items():
            text = text.replace(bad, good)
        return text

    def _llm_fix(self, text: str) -> str:
        """
        Send the full OCR text to Ollama for context-aware correction.
        Returns original text on any failure — never crashes the pipeline.
        """
        import requests

        prompt = self.CORRECTION_PROMPT.format(text=text)
        try:
            resp = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": "llama3.2",
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.0}
                },
                timeout=30
            )
            corrected = resp.json().get("response", "").strip()
            if corrected and len(corrected) > len(text) * 0.5:
                corrected = self._restore_dropped_symbols(text, corrected)
                return corrected
        except Exception:
            pass

        return text

    def process(self, text: str) -> str:
        """
        Full correction pipeline:
          1. Unicode artifact fix  (always, instant)
          2. LLM context fix       (if use_llm=True, ~300ms)
        """
        if not text:
            return text
        text = self._unicode_fix(text)
        if self.use_llm:
            text = self._llm_fix(text)
        return text


class VisualCortex:
    def __init__(self):
        print("🧠 Initializing Visual Cortex v2...", flush=True)

        if not os.path.exists(DEBUG_DIR):
            os.makedirs(DEBUG_DIR)

        with open(OCR_LOG, "w", encoding="utf-8") as f:
            f.write("--- PADDLE-OCR DEBUG LOG ---\n")

        print("   - Loading OCR Engine (PaddleOCR)...", flush=True)
        self.reader = PaddleOCR(use_angle_cls=True, lang='en', device=DEVICE)

        # Post-processor: unicode fixes always on, LLM correction on by default.
        # Set use_llm=False to skip Ollama and get raw OCR output (faster).
        self.post_processor = OCRPostProcessor(use_llm=True)

        print("   - Loading Vision Model...", flush=True)
        self.model_name = "google/siglip-so400m-patch14-384"
        self.processor = SiglipProcessor.from_pretrained(self.model_name)
        self.model = SiglipModel.from_pretrained(self.model_name).to(DEVICE)
        print("✅ Visual Cortex Online.", flush=True)

    def _extract_text_from_paddle_result(self, result, log_file=None):
        """
        Universal parser for PaddleOCR results.

        PPOCRv5 / PaddleX format (confirmed from diagnostic output):
            result = [
                {
                    'input_path': ...,
                    'dt_polys': [ array([[x,y],...]), ... ],   ← bounding boxes
                    'rec_texts': ['text1', 'text2', ...],      ← THE TEXT
                    'rec_scores': [0.99, 0.87, ...],           ← confidence
                    'rec_boxes': [...],
                    ...
                }
            ]

        Also handles legacy v2 format:
            result = [ [ [[box], ("text", conf)], ... ] ]
        """
        found_texts = []
        if result is None:
            return found_texts

        def log(msg):
            if log_file:
                log_file.write(msg + "\n")

        log(f"\n=== RAW RESULT TYPE: {type(result)} ===")
        log(f"=== KEYS (if dict): {result[0].keys() if result and isinstance(result[0], dict) else 'N/A'} ===\n")

        for page in result:

            # ── FORMAT 1: PPOCRv5 / PaddleX dict ────────────────────────────────
            # Top-level page dict contains 'rec_texts' and 'rec_scores' as lists
            if isinstance(page, dict):
                texts  = page.get('rec_texts',  page.get('rec_text',  []))
                scores = page.get('rec_scores', page.get('rec_score', []))

                # Normalise: scores might be absent → fill with 1.0
                if texts and not scores:
                    scores = [1.0] * len(texts)

                if texts:
                    for t, c in zip(texts, scores):
                        t = str(t).strip()
                        c = float(c) if c is not None else 1.0
                        if t and c > 0.5:
                            found_texts.append((t, c))
                            log(f"  ✅ [{t}] (conf={c:.2f})")
                    continue  # done with this page

                # Sub-key fallback: some versions nest under 'ocr_result'
                for subkey in ('ocr_result', 'result', 'data'):
                    sub = page.get(subkey)
                    if isinstance(sub, list):
                        for item in sub:
                            if isinstance(item, dict):
                                t = item.get('rec_text', item.get('transcription', item.get('text', '')))
                                c = item.get('rec_score', item.get('score', item.get('confidence', 1.0)))
                                t = str(t).strip()
                                c = float(c) if c else 1.0
                                if t and c > 0.5:
                                    found_texts.append((t, c))
                                    log(f"  ✅ (sub) [{t}] (conf={c:.2f})")
                        if found_texts:
                            break
                continue

            # ── FORMAT 2: PaddleOCR v2 classic list ─────────────────────────────
            # page = [ [[box_coords], ("text", conf)], ... ]
            if isinstance(page, list):
                for line in page:
                    try:
                        if isinstance(line, (list, tuple)) and len(line) == 2:
                            text_pair = line[1]
                            if isinstance(text_pair, (list, tuple)) and len(text_pair) == 2:
                                t, c = text_pair[0], text_pair[1]
                                if isinstance(t, str) and isinstance(c, (float, int)):
                                    if t.strip() and float(c) > 0.5:
                                        found_texts.append((t.strip(), float(c)))
                                        log(f"  ✅ (v2) [{t}] (conf={c:.2f})")
                    except Exception:
                        pass

        return found_texts

    def scan_full_image(self, image_path, file_id):
        """
        Single-pass scan with universal PaddleOCR result parser.
        Handles v2 and v3 output formats robustly.
        """
        try:
            full_img = cv2.imread(image_path)
            if full_img is None:
                return ""

            # 2x Upscale for better OCR accuracy on small text
            scale = 2.0
            width = int(full_img.shape[1] * scale)
            height = int(full_img.shape[0] * scale)
            upscaled = cv2.resize(full_img, (width, height), interpolation=cv2.INTER_LANCZOS4)

            # Save debug view
            debug_path = os.path.join(DEBUG_DIR, f"{file_id}_debug.jpg")
            cv2.imwrite(debug_path, upscaled)

            # Run PaddleOCR
            result = self.reader.ocr(upscaled)

            with open(OCR_LOG, "a", encoding="utf-8") as f:
                f.write(f"\n--- Processing: {file_id} ---\n")

                if not result:
                    f.write("⚠️ PaddleOCR returned None/empty result\n")
                    return ""

                found_pairs = self._extract_text_from_paddle_result(result, log_file=f)

            texts = [text for text, conf in found_pairs]

            if not texts:
                # Last resort: stringify the entire result and grep for readable text
                # This handles completely unknown future formats
                with open(OCR_LOG, "a", encoding="utf-8") as f:
                    f.write("⚠️ Standard extraction found nothing. Attempting string fallback.\n")
                raw_str = str(result)
                # This won't be clean but it's better than nothing
                # You can disable this fallback if you prefer strict extraction
                # return raw_str[:2000]
                return ""

            joined = " ".join(texts)

            # ── POST-PROCESSING: LLM context-aware OCR correction ─────────
            joined = self.post_processor.process(joined)

            with open(OCR_LOG, "a", encoding="utf-8") as f:
                f.write(f"\n📝 FINAL TEXT ({len(texts)} items): {joined[:300]}...\n")

            return joined

        except Exception as e:
            print(f"⚠️ Scan Error: {e}")
            import traceback
            with open(OCR_LOG, "a", encoding="utf-8") as f:
                f.write(f"ERROR: {e}\n")
                f.write(traceback.format_exc())
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
        if not os.path.exists(image_path):
            return None
        try:
            try:
                img_check = Image.open(image_path)
                img_check.verify()
            except:
                return None

            file_id = os.path.basename(image_path).split(".")[0]

            extracted_text = self.scan_full_image(image_path, file_id)

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
        if "memories" not in tables:
            return set()
        tbl = bank.db.open_table("memories")
        results = tbl.search().select(["id"]).limit(1000000).to_list()
        processed = {r['id'] for r in results}
        print(f"📚 Loaded {len(processed)} existing memories.")
        return processed
    except Exception as e:
        print(f"⚠️ Could not load existing DB: {e}")
        return set()


# ─────────────────────────────────────────────────────────────────────────────
# DIAGNOSTIC TOOL: Run this file directly to see exactly what PaddleOCR
# returns for any image, and confirm the fix is working.
#
#   python src/process.py path/to/your/screenshot.jpg
# ─────────────────────────────────────────────────────────────────────────────
def run_diagnostic(image_path):
    print(f"\n🔬 DIAGNOSTIC MODE — Testing: {image_path}")
    print("=" * 60)

    from paddleocr import PaddleOCR
    reader = PaddleOCR(use_angle_cls=True, lang='en', device='cpu')

    img = cv2.imread(image_path)
    if img is None:
        print("❌ Could not load image!")
        return

    scale = 2.0
    upscaled = cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)),
                          interpolation=cv2.INTER_LANCZOS4)

    print("🚀 Running PaddleOCR...")
    result = reader.ocr(upscaled)

    print(f"\n📦 Result type:   {type(result)}")
    print(f"📦 Result length: {len(result) if result else 0}")

    if result and isinstance(result[0], dict):
        print(f"\n🗝️  Top-level keys in result[0]:\n   {list(result[0].keys())}")
        for key in ('rec_texts', 'rec_text', 'rec_scores', 'rec_score'):
            val = result[0].get(key)
            if val is not None:
                preview = val[:5] if isinstance(val, list) else val
                print(f"   {key}: {preview}  ← ({len(val) if isinstance(val, list) else 1} items)")
    else:
        print(f"\n📦 Raw result (first 1000 chars):\n{str(result)[:1000]}")

    print("\n" + "=" * 60)

    # Extraction — use __new__ to skip loading heavy models (we only need the parser)
    cortex = VisualCortex.__new__(VisualCortex)
    found = cortex._extract_text_from_paddle_result(result)

    print(f"\n✅ Extracted {len(found)} text items:\n")
    for text, conf in found:
        print(f"  [{conf:.2f}] {text}")

    # ── Raw text (pre-correction) ─────────────────────────────────────────
    raw_text = " ".join(t for t, _ in found)

    print("\n" + "=" * 60)
    print(f"📝 RAW OCR text (before LLM correction):\n{raw_text}")

    # ── LLM Post-processing ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("🤖 Running LLM post-correction (Ollama llama3.2)...")
    print("   (Make sure Ollama is running — same requirement as the Chat tab)\n")

    post = OCRPostProcessor(use_llm=True)
    corrected_text = post.process(raw_text)

    print(f"✨ CORRECTED text (after LLM):\n{corrected_text}")

    # ── Diff: show only lines that changed ───────────────────────────────
    print("\n" + "=" * 60)
    print("🔍 DIFF — changes made by LLM:\n")

    import difflib
    raw_words       = raw_text.split()
    corrected_words = corrected_text.split()

    matcher = difflib.SequenceMatcher(None, raw_words, corrected_words, autojunk=False)
    changes_found = False

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            continue
        changes_found = True
        # Show a small context window around the change
        ctx_before = " ".join(raw_words[max(0, i1-3):i1])
        ctx_after  = " ".join(corrected_words[j2:j2+3])

        if tag == 'replace':
            old = " ".join(raw_words[i1:i2])
            new = " ".join(corrected_words[j1:j2])
            print(f"  ✏️  REPLACE: [{old}]  →  [{new}]")
        elif tag == 'delete':
            old = " ".join(raw_words[i1:i2])
            print(f"  🗑️  DELETE:  [{old}]")
        elif tag == 'insert':
            new = " ".join(corrected_words[j1:j2])
            print(f"  ➕  INSERT:  [{new}]")

        if ctx_before or ctx_after:
            print(f"     context: ...{ctx_before} ^^^ {ctx_after}...")
        print()

    if not changes_found:
        print("  No changes — LLM agreed the raw OCR text was already correct.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Diagnostic mode: python src/process.py <image_path>
        run_diagnostic(sys.argv[1])
    else:
        # Normal service mode
        cortex = VisualCortex()
        bank = MemoryBank()
        processed_ids = get_all_processed_ids(bank)

        print(f"🚀 Visual Cortex Service Started. Watching {SCREENSHOTS_DIR}...", flush=True)
        print("⚡ Mode: PADDLE-OCR (Full Context AI)", flush=True)

        try:
            while True:
                if not os.path.exists(SCREENSHOTS_DIR):
                    time.sleep(CHECK_INTERVAL)
                    continue

                all_files = [f for f in os.listdir(SCREENSHOTS_DIR) if f.endswith(".jpg")]
                pending_files = [f for f in all_files if f.split(".")[0] not in processed_ids]

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
                    word_count = len(data['text'].split()) if data['text'] else 0
                    print(f" Done ✅ ({word_count} words extracted)")
                else:
                    print(" Skipped (Error) ⚠️")
                    processed_ids.add(target_id)

        except KeyboardInterrupt:
            print("\n🛑 Service stopped.")