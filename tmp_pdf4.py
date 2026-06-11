import sys
sys.stdout.reconfigure(encoding='utf-8')
from pypdf import PdfReader
import re

path = r"C:\Users\StoerkJulius\.claude\projects\c--Users-StoerkJulius-OneDrive---VARTA-Microbattery-GmbH-Dokumente-Zehner-Schl-nder\f2a6329f-06c0-4aa5-9b09-2024e2419e28\tool-results\webfetch-1781179933947-vhopne.pdf"
reader = PdfReader(path)
print(f"Pages: {len(reader.pages)}")
for i, page in enumerate(reader.pages):
    text = page.extract_text() or ""
    if re.search(r"separator|W/m|W m|conductivit", text, re.I):
        clean = re.sub(r"\s+", " ", text)
        print(f"\n===== page {i+1} =====")
        print(clean[:1200])
