import sys
sys.stdout.reconfigure(encoding='utf-8')
from pypdf import PdfReader

path = r"C:\Users\StoerkJulius\.claude\projects\c--Users-StoerkJulius-OneDrive---VARTA-Microbattery-GmbH-Dokumente-Zehner-Schl-nder\f2a6329f-06c0-4aa5-9b09-2024e2419e28\tool-results\webfetch-1781179840809-r9i898.pdf"
reader = PdfReader(path)
t = reader.pages[2].extract_text() or "(EMPTY)"
print("PAGE 3 sample:", t[:500])
# count pages with any text
n_text = sum(1 for p in reader.pages[:20] if (p.extract_text() or "").strip())
print(f"\npages 1-20 with text: {n_text}/20")
# search loosely for 'separator' across all pages
hits = [i+1 for i, p in enumerate(reader.pages) if "eparator" in (p.extract_text() or "")]
print("pages mentioning separator:", hits[:30])
