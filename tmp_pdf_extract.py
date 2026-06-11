import sys
sys.stdout.reconfigure(encoding='utf-8')
from pypdf import PdfReader
import re

path = r"C:\Users\StoerkJulius\.claude\projects\c--Users-StoerkJulius-OneDrive---VARTA-Microbattery-GmbH-Dokumente-Zehner-Schl-nder\f2a6329f-06c0-4aa5-9b09-2024e2419e28\tool-results\webfetch-1781179840809-r9i898.pdf"
reader = PdfReader(path)
print(f"Pages: {len(reader.pages)}")

# Find pages mentioning separator + thermal conductivity values
for i, page in enumerate(reader.pages):
    text = page.extract_text() or ""
    if re.search(r"separator", text, re.I) and re.search(r"W\s*[/ ]?\s*\(?m?\s*[· ]?K|W\s*m.1\s*K", text):
        # Print lines containing separator or conductivity values
        lines = text.split("\n")
        relevant = [l for l in lines if re.search(r"separator|W.?m.?1.?K|W/m|0\.\d{2}", l, re.I)]
        if relevant:
            print(f"\n=== page {i+1} ===")
            for l in relevant[:40]:
                print(" ", l.strip()[:160])
