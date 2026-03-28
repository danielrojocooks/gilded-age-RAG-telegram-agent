"""
Extract tables from The Book of Yields PDF using pdfplumber and save as
clean structured text for ingestion. Run once, then re-run ingest.py.
"""
import re
import pdfplumber
from pathlib import Path

PDF_PATH = Path(__file__).parent / "KB" / "The-Book-of-Yields-Accuracy-in-Food-Costing-and-Purchasing.pdf"
OUT_PATH = Path(__file__).parent / "KB" / "book_of_yields_extracted.txt"

def clean_cell(val):
    if val is None:
        return ""
    return " ".join(str(val).split())  # collapse newlines/whitespace

def extract_section_header(text):
    """Pull section heading from page text (e.g. 'Vegetables', 'Meats & Poultry')."""
    if not text:
        return None
    # First non-blank, non-numeric line that looks like a category heading
    for line in text.splitlines():
        line = line.strip()
        if (line
                and not re.match(r"^[\d\s]+$", line)
                and not line.startswith("JWCL")
                and len(line) < 60
                and not re.search(r"ounce|pound|percent|Trimmed|Cleaned|Number|Weight|Measure|Item Name", line, re.I)):
            return line
    return None

output_lines = []
current_section = "General"

print(f"Processing {PDF_PATH.name}…")

with pdfplumber.open(str(PDF_PATH)) as pdf:
    for page_num, page in enumerate(pdf.pages, 1):
        raw_text = page.extract_text() or ""
        tables = page.extract_tables()

        # Update section heading if we find one
        heading = extract_section_header(raw_text)
        if heading and heading != current_section:
            current_section = heading
            output_lines.append(f"\n{'='*60}")
            output_lines.append(f"SECTION: {current_section}")
            output_lines.append(f"{'='*60}")

        if not tables:
            # For text-only pages, include the raw text if it looks useful
            if raw_text and len(raw_text) > 100 and "intentionally left blank" not in raw_text:
                cleaned = re.sub(r"JWCL\S+.*\n?", "", raw_text).strip()
                if cleaned:
                    output_lines.append(cleaned)
            continue

        for table in tables:
            rows_written = 0
            for row in table:
                cells = [clean_cell(c) for c in row]
                # Skip header rows and empty rows
                if not any(cells):
                    continue
                if all(c == "" or re.match(r"^(Trimmed|Cleaned|Number|Weight|Measure|Item Name|AP Unit|Ounce|Yield|Count|Cups).*", c, re.I) for c in cells if c):
                    continue

                # Try to parse: first non-empty cell = item name, find yield %
                item = cells[0] if cells[0] else ""
                rest = " | ".join(c for c in cells[1:] if c)

                # Extract yield percent if present
                yield_match = re.search(r"(\d+\.?\d*)\s*%", rest)
                yield_pct = f" | Yield: {yield_match.group(0)}" if yield_match else ""

                if item:
                    line = f"{current_section} — {item}: {rest}{yield_pct}"
                    output_lines.append(line)
                    rows_written += 1

        if page_num % 50 == 0:
            print(f"  Processed page {page_num}/{len(pdf.pages)}…")

OUT_PATH.write_text("\n".join(output_lines), encoding="utf-8")
print(f"\nDone. Wrote {len(output_lines):,} lines to {OUT_PATH.name}")
print("Now run ingest.py to rebuild the index.")
