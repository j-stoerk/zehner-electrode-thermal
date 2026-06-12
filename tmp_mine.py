import sys
sys.stdout.reconfigure(encoding='utf-8')
from pypdf import PdfReader
import re

for path, label in [("_scratch/litdata/oehler_diss.pdf", "OEHLER DISS"),
                    ("_scratch/litdata/tum_1651438.pdf", "TUM 1651438")]:
    r = PdfReader(path)
    print(f"\n{'='*70}\n{label}: {len(r.pages)} pages")
    # title page
    t = (r.pages[0].extract_text() or "")[:300].replace("\n", " ")
    print("TITLE:", t[:200])
    # find pages with measured lambda + porosity tables
    hits = []
    for i, p in enumerate(r.pages):
        txt = p.extract_text() or ""
        if re.search(r"W/?\(?m\s?\.?\s?K\)?|W\s?m-1|Wm−1", txt) and re.search(r"orosit|NMC|raphit|node|athod", txt):
            hits.append(i + 1)
    print("candidate pages:", hits[:40])
