import json
import logging
from pathlib import Path
from typing import Iterable, List, Dict

logger = logging.getLogger(__name__)


class GraphArtifacts:
    def __init__(self, graph_context_data: dict, cites_edges: dict):
        self._graph_context = graph_context_data
        self._cites_edges = cites_edges

    @classmethod
    def load(cls, dir_path: Path) -> "GraphArtifacts":
        graph_context_path = dir_path / "graph_context.json"
        cites_edges_path = dir_path / "cites_edges.json"

        try:
            with open(graph_context_path, "r", encoding="utf-8") as f:
                graph_context_data = json.load(f)
        except FileNotFoundError:
            logger.warning(f"graph_context.json not found in {dir_path}")
            graph_context_data = {}

        try:
            with open(cites_edges_path, "r", encoding="utf-8") as f:
                cites_edges = json.load(f)
        except FileNotFoundError:
            logger.warning(f"cites_edges.json not found in {dir_path}")
            cites_edges = {}

        return cls(graph_context_data, cites_edges)

    def graph_context(self, occurrence_ids: Iterable[str]) -> List[Dict]:
        """Return the knowledge-graph context for occurrences."""
        ids = list(dict.fromkeys(occurrence_ids))
        if not ids:
            return []

        out = []
        for doc_id in ids:
            bare_id = doc_id.split("/", 1)[-1] if "/" in doc_id else doc_id
            if bare_id in self._graph_context:
                # Provide defaults for older pre-v4 structures where possible
                row = dict(self._graph_context[bare_id])
                row.setdefault("rec_regs", [])
                row.setdefault("reg_guided_acs", [])
                out.append(row)

        logger.debug("graph_context: %d ids in, %d rows out", len(ids), len(out))
        return out

    def recurring_context(
        self,
        occurrence_ids: Iterable[str],
        *,
        max_regs: int = 5,
        max_siblings_per_reg: int = 3,
        max_reg_degree: int = 15,
    ) -> List[Dict]:
        """Pure Python port of recurring_context_for_occurrences."""
        ids = list(dict.fromkeys(occurrence_ids))
        if not ids:
            return []

        seeds = set()
        for doc_id in ids:
            bare_id = doc_id.split("/", 1)[-1] if "/" in doc_id else doc_id
            seeds.add(bare_id)

        reg_ids = set()
        occ_cites = self._cites_edges.get("occ_cites", {})
        for seed_id in seeds:
            reg_ids.update(occ_cites.get(seed_id, []))

        reg_occs = self._cites_edges.get("reg_occs", {})

        candidates = []
        for reg_id in reg_ids:
            occ_ids = reg_occs.get(reg_id, [])
            deg = len(occ_ids)
            if deg > len(seeds) and deg <= max_reg_degree:
                sibling_ids = [x for x in occ_ids if x not in seeds]
                if sibling_ids:
                    candidates.append({
                        "reg": reg_id,
                        "occ_count": deg,
                        "siblings": sibling_ids
                    })

        # Order by occ_count DESC (and reg to ensure deterministic ordering)
        candidates.sort(key=lambda x: (-x["occ_count"], x["reg"]))

        out = []
        for cand in candidates[:max_regs]:
            sib_ids = cand["siblings"][:max_siblings_per_reg]
            if not sib_ids:
                continue
            sibs = [{"occ_id": x, "source_doc_id": f"tsb/{x}"} for x in sib_ids]
            out.append({
                "reg": cand["reg"],
                "occ_count": cand["occ_count"],
                "siblings": sibs,
            })

        logger.debug("recurring_context: %d seeds in, %d recurring regs out", len(ids), len(out))
        return out

    def doc_lookup(self, doc_id: str) -> dict:
        """Returns the single row for GET /graph/{doc_id}."""
        bare_id = doc_id.split("/", 1)[-1] if "/" in doc_id else doc_id
        if bare_id not in self._graph_context:
            from hf_space.api_client import ApiError
            raise ApiError(404, f"No graph data for {doc_id!r}")
        row = dict(self._graph_context[bare_id])
        row.setdefault("rec_regs", [])
        row.setdefault("reg_guided_acs", [])
        return row
