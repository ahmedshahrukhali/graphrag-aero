#!/usr/bin/env python
"""Analyze TC AC index link structure."""
from urllib.parse import urlparse

with open('data/recon/tc/index_en_links.txt') as f:
    links = f.read().strip().split('\n')

# Filter to AC-related paths
ac_links = [u for u in links if '/advisory-circulars' in u]
print(f'Links under /advisory-circulars: {len(ac_links)}')
for u in ac_links[:30]:
    print(f'  {u}')

# Analyze subpage
print('\n--- Sample subpage links ---')
with open('data/recon/tc/subpage_sample.html') as f:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(f.read(), 'html.parser')
    subpage_links = {
        u for a in soup.find_all('a', href=True)
        if (u := 'https://' + a['href'].lstrip('/') if a['href'].startswith('/') else a['href']).startswith('https://tc.canada.ca')
    }

ac_in_subpage = [u for u in subpage_links if '/advisory-circulars' in u or 'AC_' in u or '.pdf' in u.lower()]
print(f'AC-related + PDF links on subpage: {len(ac_in_subpage)}')
for u in sorted(ac_in_subpage)[:30]:
    print(f'  {u}')
