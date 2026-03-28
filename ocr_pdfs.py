"""
OCR image-only PDFs and save extracted text as .txt files alongside the originals.
Run this once before ingest.py — ingest.py will then pick up the .txt files automatically.
"""
from pathlib import Path
import pytesseract
from pdf2image import convert_from_path

TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH  = r"C:\Users\danie\poppler\poppler-24.08.0\Library\bin"

pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

KB_DIR = Path(__file__).parent / "KB"

OCR_TARGETS = [
    "delmonico_menu.pdf",
    "grover cleveland menu 1891.pdf",
    "What's on the Menu - About.pdf",
    "Grover Cleveland NYT state dinner.pdf",
]

for filename in OCR_TARGETS:
    pdf_path = KB_DIR / filename
    if not pdf_path.exists():
        print(f"[SKIP] {filename} — not found")
        continue

    out_path = pdf_path.with_suffix(".txt")
    print(f"OCR-ing {filename}…")

    images = convert_from_path(str(pdf_path), dpi=300, poppler_path=POPPLER_PATH)
    pages_text = []
    for i, img in enumerate(images, 1):
        text = pytesseract.image_to_string(img, lang="eng")
        pages_text.append(f"--- Page {i} ---\n{text}")
        print(f"  Page {i}/{len(images)}: {len(text)} chars extracted")

    full_text = "\n\n".join(pages_text)
    out_path.write_text(full_text, encoding="utf-8")
    print(f"  Saved to {out_path.name} ({len(full_text):,} chars)\n")

print("OCR complete. Run ingest.py to rebuild the index.")
