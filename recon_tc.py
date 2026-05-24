#!/usr/bin/env python
"""TC AC index reconnaissance — fetch and analyze."""
from ingestion.acquisition.http_client import make_session, fetch_text
from ingestion.acquisition.tc import extract_pdf_urls
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import os

os.makedirs('data/recon/tc', exist_ok=True)
s = make_session()

# Step 1: Fetch EN and FR indexes
print("Step 1: Fetching EN and FR AC indexes...")
html_en = fetch_text(s, 'https://tc.canada.ca/en/aviation/reference-centre/advisory-circulars')
with open('data/recon/tc/index_en.html', 'w', encoding='utf-8') as f:
    f.write(html_en)
print("[OK] Saved index_en.html")

html_fr = fetch_text(s, 'https://tc.canada.ca/fr/aviation/centre-reference/circulaires-information')
with open('data/recon/tc/index_fr.html', 'w', encoding='utf-8') as f:
    f.write(html_fr)
print("[OK] Saved index_fr.html")

# Step 2: Run existing parser
print("\nStep 2: Running existing extract_pdf_urls on EN index...")
base_en = 'https://tc.canada.ca/en/aviation/reference-centre/advisory-circulars'
urls_direct = extract_pdf_urls(html_en, base_en)
print(f"Direct PDF links found: {len(urls_direct)}")
for u in urls_direct[:20]:
    print(f"  {u}")

# Step 3: Dump all same-host links
print("\nStep 3: Extracting all tc.canada.ca links from EN index...")
soup = BeautifulSoup(html_en, 'html.parser')
links = sorted({
    urljoin(base_en, a['href'])
    for a in soup.find_all('a', href=True)
    if urlparse(urljoin(base_en, a['href'])).netloc.endswith('tc.canada.ca')
})
with open('data/recon/tc/index_en_links.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(links))
print(f"Wrote {len(links)} unique links to index_en_links.txt")

# Show unique path prefixes (first 3 segments)
print("\nUnique path prefixes (first 3 segments):")
prefixes = sorted({'/'.join(urlparse(u).path.split('/')[:4]) for u in links})
for p in prefixes[:30]:
    print(f"  {p}")
if len(prefixes) > 30:
    print(f"  ... and {len(prefixes) - 30} more")

# Step 4: Pick a sample subpage (non-PDF, under /en/aviation/)
print("\nStep 4: Selecting sample subpage...")
subpages = [
    u for u in links
    if '/en/aviation/' in u and not u.lower().endswith('.pdf')
]
if subpages:
    sample_url = subpages[0]  # Take first non-PDF AC-related page
    print(f"Selected: {sample_url}")
    subpage_html = fetch_text(s, sample_url)
    with open('data/recon/tc/subpage_sample.html', 'w', encoding='utf-8') as f:
        f.write(subpage_html)

    # Extract PDFs from subpage
    sample_pdfs = extract_pdf_urls(subpage_html, sample_url)
    print(f"Direct PDF links on subpage: {len(sample_pdfs)}")
    for u in sample_pdfs[:10]:
        print(f"  {u}")
    print("[OK] Saved subpage_sample.html")
else:
    print("No non-PDF subpages found; skipping subpage fetch.")

# Confirm files
print("\nStep 5: Confirming artifacts...")
for fname in ['index_en.html', 'index_fr.html', 'index_en_links.txt', 'subpage_sample.html']:
    path = f'data/recon/tc/{fname}'
    if os.path.exists(path):
        size = os.path.getsize(path)
        print(f"[OK] {fname} ({size} bytes)")
    else:
        print(f"[FAIL] {fname} (missing)")
