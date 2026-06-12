import sys
sys.stdout.reconfigure(encoding='utf-8')
from pypdf import PdfReader
import re

r = PdfReader("_scratch/litdata/oehler_diss.pdf")
for pg in [135, 136, 137, 141, 143, 144, 145, 147, 148]:
    txt = (r.pages[pg-1].extract_text() or "").replace("\n", " ")
    txt = re.sub(r"\s+", " ", txt)
    print(f"\n===== OEHLER p.{pg} =====")
    print(txt[:900])
