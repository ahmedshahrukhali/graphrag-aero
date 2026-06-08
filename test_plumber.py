import pdfplumber
import json
import os
from huggingface_hub import hf_hub_download

# Download the PDF
pdf_path = hf_hub_download(repo_id="ahmedsali/graphaero-corpus", repo_type="dataset", filename="en/tsb/a00q0094.pdf")

with pdfplumber.open(pdf_path) as pdf:
    p = pdf.pages[1]  # p.2 (0-indexed)
    print("Page 2 text:")
    print(p.extract_text()[:200])
    hits = p.search("runway", regex=True, case=False)
    print(f"Hits for runway: {len(hits)}")
    if hits:
        print(hits[0])
