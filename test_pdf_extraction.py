"""Test PDF extraction quality using pdfplumber."""
import pdfplumber
from pathlib import Path

KB_DIR = Path(__file__).parent / "KB"
pdfs = sorted(KB_DIR.glob("*.pdf"))

print(f"Found {len(pdfs)} PDFs\n{'='*60}")

for pdf_path in pdfs:
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            text_pages = 0
            total_chars = 0
            sample = ""

            for page in pdf.pages:
                text = page.extract_text() or ""
                if text.strip():
                    text_pages += 1
                    total_chars += len(text)
                    if not sample:
                        sample = text[:120].replace("\n", " ")

            coverage = (text_pages / total_pages * 100) if total_pages else 0
            status = "OK" if coverage >= 80 and total_chars > 500 else "NEEDS OCR"
            print(f"[{status}] {pdf_path.name}")
            print(f"       Pages: {text_pages}/{total_pages} with text ({coverage:.0f}%)")
            print(f"       Chars: {total_chars:,}")
            if sample:
                print(f"       Sample: {sample!r}")
            else:
                print("       Sample: (no text extracted)")
            print()
    except Exception as e:
        print(f"[ERROR] {pdf_path.name}: {e}\n")
