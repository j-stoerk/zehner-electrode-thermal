import sys
sys.stdout.reconfigure(encoding='utf-8')
from pypdf import PdfReader
import re

r = PdfReader("_scratch/litdata/tum_1651438.pdf")
for pg in [31, 62, 64, 65, 66, 67, 68, 69, 71, 72, 76]:
    txt = (r.pages[pg-1].extract_text() or "").replace("\n", " ")
    txt = re.sub(r"\s+", " ", txt)
    print(f"\n===== STEINHARDT p.{pg} =====")
    print(txt[:1100])
