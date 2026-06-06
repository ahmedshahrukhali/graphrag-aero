"""Pre-compute the sample-query answers into ``hf_space/sample_cache.json``.

The Space serves these instantly (no LLM generation) when a user clicks an
example — the same idea as Gradio's ``cache_examples=True``. Run once against a
live backend whenever the corpus/index changes:

    python -m hf_space.build_sample_cache --backend http://localhost:8080

The query list mirrors ``SAMPLE_QUERIES`` in ``hf_space/app.py`` (kept here as a
literal so this script has no gradio import and can run on a bare host).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import httpx

# Mirror of hf_space.app.SAMPLE_QUERIES: (query, lang, source, max_hops).
SAMPLE_QUERIES: list[tuple[str, str, str, int]] = [
    ("fuel exhaustion forced landing", "en", "tsb", 2),
    ("engine failure after takeoff", "en", "tsb", 2),
    ("carburetor icing", "en", "all", 2),
    ("VFR flight into IMC", "en", "tsb", 2),
    ("alimentation en carburant", "fr", "tsb", 2),
    ("approach procedures helicopter", "en", "tc", 2),
    ("安捷飛航訓練中心 DA-40NG 發動機失效迫降高雄外海", "zh", "ttsb", 2),
    ("民用航空器维修计划和控制 CCAR-121", "zh", "caac", 2),
]

CACHE_PATH = Path(__file__).with_name("sample_cache.json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", default=os.environ.get("BACKEND_URL", "http://localhost:8080"))
    ap.add_argument("--timeout", type=float, default=900.0, help="per-query seconds (generation is slow)")
    args = ap.parse_args()

    cache: dict[str, dict] = {}
    with httpx.Client(timeout=args.timeout) as client:
        for i, (q, lang, source, hops) in enumerate(SAMPLE_QUERIES, 1):
            body = {
                "query": q, "thread_id": f"cache-{i}", "max_hops": hops,
                "lang": None if lang == "all" else lang,
                "source": None if source == "all" else source,
            }
            print(f"[{i}/{len(SAMPLE_QUERIES)}] {q!r} (lang={lang}, source={source}) …", flush=True)
            r = client.post(f"{args.backend}/query", json=body)
            r.raise_for_status()
            data = r.json()
            cache[q] = {
                "draft": data.get("draft", ""),
                "sources": data.get("sources", []),
                "trace": data.get("trace", []),
                "thread_id": data.get("thread_id", ""),
            }
            print(
                f"    → draft {len(cache[q]['draft'])} chars, "
                f"{len(cache[q]['sources'])} sources",
                flush=True,
            )

    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {CACHE_PATH} ({len(cache)} queries)", flush=True)


if __name__ == "__main__":
    main()
