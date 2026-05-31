"""Map an on-disk corpus PDF to its canonical ``doc_id`` and ``source_url``.

A corpus path looks like::

    data/corpus/{lang}/{source}/{stem}.pdf

    e.g.  data/corpus/en/tsb/a00a0110.pdf
          data/corpus/fr/tc/AC_100-001_f08_20210622.pdf

``doc_id`` = ``"{source}/{stem}"`` — no language prefix, because EN and FR
versions of the same document are distinct doc_ids in our index (the stem
already differs for TC; for TSB the stem is the same but the language path
disambiguates).

``source_url`` is reconstructed from the URL patterns the acquisition layer
already encodes. For TSB we reuse :func:`ingestion.acquisition.tsb.build_pdf_url`.
For TC we can't reconstruct the full URL from the stem alone (the path
includes a YYYY-MM directory we don't have) so we return ``None`` and the
chunk record carries the empty source_url — the local PDF path is the
authoritative pointer until we wire a side-table.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ingestion.acquisition import tsb
from .lang import Lang, lang_for_path


@dataclass(frozen=True)
class DocRef:
    doc_id: str         # "{source}/{stem}"
    source: str         # "tsb" | "tc"
    stem: str           # filename without .pdf
    lang: Lang
    source_url: str | None
    path: Path
    corpus: str         # "tsb" | "tc" | "caac" — first-class corpus tag (WS-0)


_KNOWN_SOURCES = ("tsb", "tc")


def doc_ref_for_path(path: Path) -> DocRef:
    """Parse a corpus PDF path into a :class:`DocRef`.

    Raises ``ValueError`` if the path doesn't look like
    ``data/corpus/{lang}/{source}/<stem>.pdf``.
    """
    p = Path(path)
    if p.suffix.lower() != ".pdf":
        raise ValueError(f"not a PDF: {p}")
    parts = p.parts
    # find the {lang}/{source}/<stem>.pdf tail
    source = None
    for known in _KNOWN_SOURCES:
        if known in parts:
            source = known
            break
    if source is None:
        raise ValueError(f"no known source in path: {p}")
    stem = p.stem
    lang = lang_for_path(p)
    doc_id = f"{source}/{stem}"
    source_url = _source_url(source, stem, lang)
    return DocRef(
        doc_id=doc_id,
        source=source,
        stem=stem,
        lang=lang,
        source_url=source_url,
        path=p,
        # corpus == source for the EN/FR TSB+TC corpus; the ZH axis (WS-A) will
        # mint "caac" here. Kept distinct from ``source`` so the overlap demo can
        # group/filter by corpus without overloading the URL-reconstruction key.
        corpus=source,
    )


def _source_url(source: str, stem: str, lang: Lang) -> str | None:
    if source == "tsb":
        # TSB IDs in the index are uppercase; stems on disk are lowercase.
        return tsb.build_pdf_url(stem.upper(), lang)
    # TC URL includes a YYYY-MM dir we don't know from the stem alone.
    return None
