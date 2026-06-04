"""One-off corpus maintenance tools (quarantine, cleanup, audit).

Distinct from :mod:`ingestion.processing` — these tools mutate the corpus on
disk rather than producing chunks. Each module ships as a runnable CLI.
"""
