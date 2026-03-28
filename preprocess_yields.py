"""
Extract Book of Yields PDF into a clean markdown table using word-position analysis.
Column x-coordinates confirmed from inspecting actual word positions (page 49):
  Item Name  x ~143  → col 100–263
  AP Unit    x ~264  (pound, each, bunch, etc.)  → col 263–309
  Count      x ~310  (16, 9, 5.5 — number of measure units)  → col 309–363
  Measure    x ~364  (ounce, each)  → col 363–410
  Weight     x ~410  (trimmed weight/count)  → col 410–460
  Yield %    x ~461  → col 460–505
  Oz/Cup     x ~506  → col 505–558
  Cups/AP    x ~561  → col 558+

Run this script, then run ingest.py to rebuild the index.
"""
import re
from pathlib import Path
from collections import defaultdict
import pdfplumber

PDF_PATH = Path(__file__).parent / "KB" / "The-Book-of-Yields-Accuracy-in-Food-Costing-and-Purchasing.pdf"
OUT_PATH = Path(__file__).parent / "KB" / "Book_of_Yields_clean.md"
OLD_PATH = Path(__file__).parent / "KB" / "book_of_yields_extracted.txt"

# Column x-boundaries confirmed from inspecting actual word x0 positions:
#   AP unit "pound"/"bag" at x0~263.7, count "16" at x0~309.9,
#   measure "ounce" at x0~363.7, weight at x0~410.0,
#   yield "81.3%"/"100.00%" at x0~460.7, oz/cup at x0~505.7, cups at x0~560.8
COL_ITEM   = (100, 263)   # Item Name  x ~143
COL_APUNIT = (263, 309)   # AP Unit (pound/each/bunch…)  x ~263.7
COL_COUNT  = (309, 363)   # Number of measure units per AP unit  x ~309.9
COL_MEASURE= (363, 410)   # Measure unit (ounce, each)  x ~363.7
COL_WEIGHT = (410, 460)   # Trimmed weight / count  x ~410.0
COL_YIELD  = (460, 505)   # Yield %  x ~460.7
COL_OZCUP  = (505, 558)   # Oz per cup  x ~505.7
COL_CUPS   = (558, 9999)  # Cups per AP unit  x ~560.8

SECTION_RE = re.compile(
    r"^(Vegetables|Fruits?|Meats?|Poultry|Fish|Seafood|Dairy|Grains?|Nuts?|"
    r"Herbs?|Spices?|Beverages?|Produce|Dry|Starchy|Baking|Fats?|Oils?|"
    r"Condiments?|Liquids?|Legumes?|Rice|Pasta|Flour|Sweeteners?|"
    r"Coffee|Tea|Canned|Measurement|Standard|Appendix)",
    re.I
)
AP_UNITS = {"pound", "each", "bunch", "head", "bag", "jar", "case",
            "quart", "liter", "ounce", "oz", "can", "box", "pint", "gallon"}

def col(word):
    x = word["x0"]
    if COL_ITEM[0]    <= x < COL_ITEM[1]:    return "item"
    if COL_APUNIT[0]  <= x < COL_APUNIT[1]:  return "apunit"
    if COL_COUNT[0]   <= x < COL_COUNT[1]:   return "count"
    if COL_MEASURE[0] <= x < COL_MEASURE[1]: return "measure"
    if COL_WEIGHT[0]  <= x < COL_WEIGHT[1]:  return "weight"
    if COL_YIELD[0]   <= x < COL_YIELD[1]:   return "yield"
    if COL_OZCUP[0]   <= x < COL_OZCUP[1]:   return "ozcup"
    if COL_CUPS[0]    <= x:                   return "cups"
    return None

def is_header_or_noise(text):
    return bool(re.match(
        r"(JWCL|Item Name|AP Unit|Trimmed|Cleaned|Number|Measure|Ounce|Yield|Cups|Weight|Count|"
        r"This page|intentionally|Copyright|All rights|Published|Wiley|ISBN|Printed|Library)",
        text, re.I
    ))

print(f"Processing {PDF_PATH.name}…")

sections = {}        # section_name -> list of row dicts
current_section = "General"
sections[current_section] = []

with pdfplumber.open(str(PDF_PATH)) as pdf:
    for page_num, page in enumerate(pdf.pages, 1):
        words = page.extract_words(x_tolerance=2, y_tolerance=2)
        if not words:
            continue

        # Group words into rows by y-position (within 5pt)
        rows_by_y = defaultdict(list)
        for w in words:
            y_key = round(w["top"] / 5) * 5
            rows_by_y[y_key].append(w)

        for y_key in sorted(rows_by_y):
            row_words = sorted(rows_by_y[y_key], key=lambda w: w["x0"])
            full_text = " ".join(w["text"] for w in row_words)

            # Skip headers/noise
            if is_header_or_noise(full_text):
                continue

            # Detect section heading (short line, no yield %, matches known section names)
            # Section headings are centered (x0 ~100–400) — don't restrict to left margin
            if (len(full_text) < 60
                    and not re.search(r"\d+\.?\d*%", full_text)
                    and not re.search(r"\d{3,}", full_text)   # skip rows with 3+ digit numbers
                    and SECTION_RE.match(full_text)):
                sec = full_text.strip().split()[0].capitalize()
                if sec not in sections:
                    sections[sec] = []
                current_section = sec
                continue

            # Build column buckets for this row
            buckets = defaultdict(list)
            for w in row_words:
                c = col(w)
                if c:
                    buckets[c].append(w["text"])

            item    = " ".join(buckets["item"]).strip()
            apunit  = " ".join(buckets["apunit"]).strip().lower()
            count   = " ".join(buckets["count"]).strip()
            measure = " ".join(buckets["measure"]).strip()
            weight  = " ".join(buckets["weight"]).strip()
            yield_  = " ".join(buckets["yield"]).strip()
            ozcup   = " ".join(buckets["ozcup"]).strip()
            cups    = " ".join(buckets["cups"]).strip()

            # Must have item name to proceed
            if not item or is_header_or_noise(item):
                continue

            # Yield % rescue: different chapters put the % in different columns
            # (Produce: x~461→yield bucket; Meats/Seafood/Poultry: x~413→weight bucket)
            # Find the first column that contains a real percentage value
            if not re.search(r"\d+\.?\d*%", yield_):
                for cname, cval in [("weight", weight), ("ozcup", ozcup),
                                    ("count", count), ("measure", measure)]:
                    if re.search(r"\d+\.?\d*%", cval):
                        yield_ = cval
                        # Clear it from the original field to avoid duplication
                        if cname == "weight":  weight  = ""
                        elif cname == "ozcup": ozcup   = ""
                        elif cname == "count": count   = ""
                        elif cname == "measure": measure = ""
                        break

            yield_is_pct = bool(re.search(r"\d+\.?\d*%", yield_))
            if not yield_is_pct and apunit not in AP_UNITS:
                continue
            # Reject prose rows: item should be ≤8 words and not end with sentence punctuation
            if len(item.split()) > 8:
                continue
            if item.endswith((".", ":", ";")):
                continue

            sections[current_section].append({
                "item": item,
                "apunit": apunit,
                "count": count,
                "measure": measure,
                "weight": weight,
                "yield": yield_,
                "ozcup": ozcup,
                "cups": cups,
            })

        if page_num % 50 == 0:
            print(f"  Page {page_num}/{len(pdf.pages)}…")

# ── Write markdown ─────────────────────────────────────────────────────────────
lines = ["# The Book of Yields — Extracted Data\n",
         "_Source: The Book of Yields: Accuracy in Food Costing and Purchasing, F.T. Lynch_\n"]

total_rows = 0
for section, rows in sections.items():
    if not rows:
        continue
    lines.append(f"\n## {section}\n")
    lines.append("| Item | AP Unit | Count | Measure | Weight | Yield % | Oz/Cup | Cups/AP |")
    lines.append("|------|---------|-------|---------|--------|---------|--------|---------|")
    for r in rows:
        lines.append(
            f"| {r['item']} | {r['apunit']} | {r['count']} | {r['measure']} "
            f"| {r['weight']} | {r['yield']} | {r['ozcup']} | {r['cups']} |"
        )
        total_rows += 1

OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
print(f"\nWrote {total_rows:,} rows to {OUT_PATH.name}")

# Remove old extracted txt if it exists
if OLD_PATH.exists():
    OLD_PATH.unlink()
    print(f"Removed old {OLD_PATH.name}")

print("Run ingest.py to rebuild the index.")
