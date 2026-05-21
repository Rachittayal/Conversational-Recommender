import logging
from typing import Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL, MAX_RECOMMENDATIONS, TOP_K_RESULTS
from schemas import CatalogItem

logger=logging.getLogger(__name__)

class SearchQuery:
    def __init__(
        self,
        query="",
        test_types=None,
        job_levels=None,
        require_remote=None,
        languages=None,
        max_results=MAX_RECOMMENDATIONS,
    ):
        self.query = query
        self.test_types = test_types or []
        self.job_levels = job_levels or []
        self.require_remote = require_remote
        self.languages = languages or []
        self.max_results = max_results


class Retriever:
    def __init__(self):
        self.model=None
        self.index=None
        self.items: list[CatalogItem] = []
        self.ready=False

    def build_index(self, items: list[CatalogItem]) -> None:
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
        self.model=SentenceTransformer(EMBEDDING_MODEL)
        self.items=items

        texts=[item.to_embedding_text() for item in items]
        logger.info("Embedding %d items...", len(texts))

        embeddings=self.model.encode(texts,convert_to_numpy=True).astype(np.float32)    
        faiss.normalize_L2(embeddings)

        dimension=embeddings.shape[1]
        self.index=faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

        self.ready=True
        logger.info(
            "FAISS index ready: %d items, %d dimensions",
            self.index.ntotal,
            dimension,
        )

    def search(self, query: SearchQuery) -> list[CatalogItem]:
        if not self.ready:
            raise RuntimeError("Call build_index() before search()")

        if not query.query.strip() and not query.test_types:
            return self.items[:query.max_results]

        candidates = self._semantic_search(query.query)
        candidates = self._boost_name_matches(candidates, query.query)

        results = self._filter(candidates, query)

        if not results:
            logger.info("No results after filtering — relaxing progressively")
            results = self._relax_and_retry(candidates, query)

        return results[:query.max_results]

    def search_by_name(self, name_query: str) -> Optional[CatalogItem]:
        if not self.ready:
            return None

        name_lower = name_query.lower().strip()

        for item in self.items:
            if item.name.lower() == name_lower:
                return item

        for item in self.items:
            if name_lower in item.name.lower():
                return item

        results = self.search(SearchQuery(query=name_query, max_results=1))
        return results[0] if results else None


    def _semantic_search(self, query_text: str) -> list[CatalogItem]:
        """Embed query and find top K similar items."""
        query_vec = self.model.encode([query_text],convert_to_numpy=True,).astype(np.float32)

        faiss.normalize_L2(query_vec)

        k = min(TOP_K_RESULTS, len(self.items))
        scores, indices = self.index.search(query_vec, k)

        return [
            self.items[idx]
            for idx in indices[0]
            if 0 <= idx < len(self.items)
        ]

    def _boost_name_matches(self,candidates: list[CatalogItem],query_text: str) -> list[CatalogItem]:
        query_words = [
            w.lower().strip(".,+" ) for w in query_text.split()
            if len(w.strip(".,+")) >= 2
        ]

        if not query_words:
            return candidates

        boosted=[]
        normal=[]
        reports=[]

        for item in candidates:
            name_lower = item.name.lower()
            
            # Deprioritize reports - they should come last
            is_report = any(word in name_lower for word in [
                "report", "planner", "profile", "narrative", "feedback"
            ])
            
            if is_report:
                reports.append(item)
            elif any(word in name_lower for word in query_words):
                boosted.append(item)
            else:
                normal.append(item)

        # Return: exact matches first, then normal items, then reports last
        return boosted + normal + reports

    def _filter(
        self,
        candidates: list[CatalogItem],
        query: SearchQuery,
    ) -> list[CatalogItem]:
        results = []

        for item in candidates:

            if query.test_types:
                if not any(t in item.all_types for t in query.test_types):
                    continue

            if query.require_remote is not None:
                if item.remote != query.require_remote:
                    continue

            if query.job_levels and item.job_levels:
                matched = any(
                    q.lower() in level.lower()
                    for q in query.job_levels
                    for level in item.job_levels
                )
                if not matched:
                    continue

            if query.languages and item.languages:
                matched = any(
                    q.lower() in lang.lower()
                    for q in query.languages
                    for lang in item.languages
                )
                if not matched:
                    continue

            results.append(item)

        return results

    def _relax_and_retry(self,candidates: list[CatalogItem],query: SearchQuery,) -> list[CatalogItem]:
        relaxed = SearchQuery(
            query=query.query,
            test_types=query.test_types,
            job_levels=query.job_levels,
            require_remote=query.require_remote,
            languages=[],
            max_results=query.max_results,
        )
        results = self._filter(candidates, relaxed)
        if results:
            logger.info("Results found after dropping language filter")
            return results

        relaxed.job_levels = []
        results = self._filter(candidates, relaxed)
        if results:
            logger.info("Results found after dropping job level filter")
            return results

        relaxed.require_remote = None
        results = self._filter(candidates, relaxed)
        if results:
            logger.info("Results found after dropping remote filter")
            return results

        logger.info("Returning pure semantic results")
        return candidates