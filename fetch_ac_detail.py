#!/usr/bin/env python
"""Fetch an AC detail page to see if PDFs are there."""
from ingestion.acquisition.http_client import make_session, fetch_text
from ingestion.acquisition.tc import extract_pdf_urls

s = make_session()
url = 'https://tc.canada.ca/en/aviation/reference-centre/advisory-circulars/advisory-circular-ac-no-100-001'
print(f'Fetching AC detail page...')
html = fetch_text(s, url)
with open('data/recon/tc/ac_detail_sample.html', 'w', encoding='utf-8') as f:
    f.write(html)

pdfs = extract_pdf_urls(html, url)
print(f'Direct PDF links on AC detail page: {len(pdfs)}')
for p in pdfs[:10]:
    print(f'  {p}')

# Also look for "download" or "PDF" text in the page
if 'download' in html.lower() or 'pdf' in html.lower():
    print('\nPage contains "download" or "pdf" mentions.')

print(f'[OK] Saved ac_detail_sample.html ({len(html)} bytes)')
