import sys
sys.stdout.reconfigure(encoding='utf-8')
from pypdf import PdfReader
import re

path = r"C:\Users\StoerkJulius\.claude\projects\c--Users-StoerkJulius-OneDrive---VARTA-Microbattery-GmbH-Dokumente-Zehner-Schl-nder\f2a6329f-06c0-4aa5-9b09-2024e2419e28\tool-results\webfetch-1781179840809-r9i898.pdf"
reader = PdfReader(path)
for i in [10, 11, 16, 23, 24]:   # 0-based: pages 11,12,17,24,25
    text = reader.pages[i].extract_text() or ""
    if "eparator" in text:
        idxs = [m.start() for m in re.finditer("eparator", text)]
        print(f"\n========== page {i+1} ==========")
        for j in idxs[:6]:
            print("..." + text[max(0,j-350):j+350].replace("\n", " ") + "...")
            print("---")
