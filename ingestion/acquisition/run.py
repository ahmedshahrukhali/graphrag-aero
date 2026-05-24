"""CLI: scrape TSB + TC indexes and download PDFs into ``data/corpus/``.

Layout:
    data/corpus/en/tsb/<id>.pdf
    data/corpus/fr/tsb/<id>.pdf
    data/corpus/en/tc/<filename>.pdf
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import requests

from . import tc, tsb
from .http_client import DownloadResult, download, fetch_text, make_session

logger = logging.getLogger(__name__)


def run_tsb(
    out_root: Path,
    session: requests.Session,
    limit: int | None = None,
) -> list[DownloadResult]:
    html = fetch_text(session, tsb.INDEX_URL_EN)
    ids = tsb.extract_report_ids(html)
    logger.info("TSB: %d report IDs in index", len(ids))
    if limit is not None:
        ids = ids[:limit]
    results: list[DownloadResult] = []
    for rid in ids:
        for lang in ("en", "fr"):
            url = tsb.build_pdf_url(rid, lang)  # type: ignore[arg-type]
            dest = out_root / lang / "tsb" / f"{rid.lower()}.pdf"
            try:
                results.append(download(session, url, dest))
            except requests.HTTPError as e:
                logger.warning("TSB %s/%s: %s", rid, lang, e)
    return results


def _run_tc_lang(
    out_root: Path,
    session: requests.Session,
    index_url: str,
    lang: str,
    limit: int | None,
) -> list[DownloadResult]:
    index_html = fetch_text(session, index_url)
    detail_urls = tc.extract_ac_detail_urls(index_html, index_url)
    logger.info("TC %s: %d AC detail pages in index", lang, len(detail_urls))
    if limit is not None:
        detail_urls = detail_urls[:limit]

    results: list[DownloadResult] = []
    for detail_url in detail_urls:
        try:
            detail_html = fetch_text(session, detail_url)
        except requests.HTTPError as e:
            logger.warning("TC %s detail %s: %s", lang, detail_url, e)
            continue
        pdf_urls = tc.extract_pdf_urls(detail_html, detail_url)
        if not pdf_urls:
            logger.warning("TC %s detail %s: no PDF link found", lang, detail_url)
            continue
        for pdf_url in pdf_urls:
            dest = out_root / lang / "tc" / tc.filename_for(pdf_url)
            try:
                results.append(download(session, pdf_url, dest))
            except requests.HTTPError as e:
                logger.warning("TC %s %s: %s", lang, pdf_url, e)
    return results


def run_tc(
    out_root: Path,
    session: requests.Session,
    limit: int | None = None,
) -> list[DownloadResult]:
    results: list[DownloadResult] = []
    results.extend(_run_tc_lang(out_root, session, tc.INDEX_URL_EN, "en", limit))
    results.extend(_run_tc_lang(out_root, session, tc.INDEX_URL_FR, "fr", limit))
    return results


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Scrape & download TC + TSB PDFs.")
    p.add_argument("--out", type=Path, default=Path("data/corpus"),
                   help="Output root (default: data/corpus)")
    p.add_argument("--source", choices=["tsb", "tc", "all"], default="all")
    p.add_argument("--limit", type=int, default=None,
                   help="Max docs per source — handy for sample runs.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    session = make_session()
    all_results: list[DownloadResult] = []
    if args.source in ("tsb", "all"):
        all_results.extend(run_tsb(args.out, session, limit=args.limit))
    if args.source in ("tc", "all"):
        all_results.extend(run_tc(args.out, session, limit=args.limit))

    new = sum(1 for r in all_results if r.downloaded)
    skipped = sum(1 for r in all_results if not r.downloaded)
    logger.info("done: %d new, %d skipped, %d total", new, skipped, len(all_results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
