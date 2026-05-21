import json
import logging
from pathlib import Path

import httpx

from config import (
    CATALOG_CACHE_PATH,
    CATALOG_URL,
    TEST_TYPE_MAPPING,
)
from schemas import CatalogItem

logger=logging.getLogger(__name__)


async def load_catalog() -> list[CatalogItem]:
    cache_path=Path(CATALOG_CACHE_PATH)

    if cache_path.exists():
        logger.info("Loading catalog from cache: %s", cache_path)
        with open(cache_path, "r", encoding="utf-8") as f:
            raw_data=json.load(f)
    else:
        logger.info("Fetching catalog from network: %s", CATALOG_URL)
        async with httpx.AsyncClient(timeout=60.0) as client:
            response=await client.get(CATALOG_URL)
            response.raise_for_status()
            raw_data=response.json()

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(raw_data, f, indent=2)
        logger.info("Catalog cached to: %s", cache_path)

    items=[]

    for raw in raw_data:
        name=raw.get("name", "").strip()
        url=raw.get("link", "").strip()

        if not name or not url:
            logger.warning("Skipping item missing name or url: %s", raw)
            continue

        keys=raw.get("keys", [])

        all_types = []
        for label in keys:
            if label not in TEST_TYPE_MAPPING:
                logger.warning("Unknown catalog label %r — defaulting to K", label)
            code=TEST_TYPE_MAPPING.get(label, "K")
            if code not in all_types:
                all_types.append(code)

        if not all_types:
            all_types=["K"]

        test_type=all_types[0]

        item = CatalogItem(
            name=name,
            url=url,
            test_type=test_type,
            all_types=all_types,
            description=raw.get("description", "").strip(),
            duration=raw.get("duration") or None,
            languages=raw.get("languages", []),
            job_levels=raw.get("job_levels", []),
            remote=str(raw.get("remote", "")).lower() == "yes",
            adaptive=str(raw.get("adaptive", "")).lower() == "yes",
        )

        items.append(item)

    if not items:
        raise RuntimeError("Catalog empty after parsing. Check field names.")

    logger.info("Catalog loaded: %d items", len(items))
    return items