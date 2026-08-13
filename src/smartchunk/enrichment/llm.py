"""LLM-based enrichment engine using LiteLLM.

Enriches SmartChunks with summaries, entities, keywords, and
confidence scores via any LLM provider supported by LiteLLM.

Includes a hash-based caching layer that skips redundant LLM calls
for duplicate or previously-enriched chunk texts.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from typing import Any

from litellm import acompletion  # type: ignore[import-untyped]

from smartchunk.enrichment.base import BaseEnricher
from smartchunk.enrichment.prompts import SYSTEM_PROMPT, build_enrichment_prompt
from smartchunk.models import EnrichmentConfig, EnrichmentField, SmartChunk

logger = logging.getLogger("smartchunk.enrichment")


class LLMEnricher(BaseEnricher):
    """Enriches chunks using LLM inference via LiteLLM.

    Features:
    - **Hash-based caching**: identical chunk texts skip the LLM entirely.
    - Batch processing with configurable concurrency
    - Structured JSON output with validation
    - Automatic retry with exponential backoff
    - Cost tracking (tokens used + cache savings)
    - Neighbor summary linking (prev_summary / next_summary)
    """

    def __init__(self, config: EnrichmentConfig | None = None) -> None:
        super().__init__(config)
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._total_cost = 0.0

        # ── Cost Optimization: Hash-based enrichment cache ──
        # Key: SHA-256 hash of chunk text  →  Value: enrichment dict
        from pathlib import Path
        self._cache_file = str(Path(__file__).parent.parent.parent.parent / ".smartchunk_cache.json")
        self._cache: dict[str, dict[str, Any]] = {}
        
        import os
        if os.path.exists(self._cache_file):
            try:
                with open(self._cache_file, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
                logger.info("Loaded %d entries from persistent cache: %s", len(self._cache), self._cache_file)
            except Exception as e:
                logger.warning("Failed to load persistent cache from %s: %s", self._cache_file, e)

        self._cache_hits = 0
        self._cache_misses = 0
        self._time_saved_seconds = 0.0  # Estimated wall-time savings

    # ── Public Properties ──────────────────────────────────────────────────

    @property
    def usage_stats(self) -> dict[str, Any]:
        """Return cumulative token usage, cost, and cache performance."""
        return {
            "prompt_tokens": self._total_prompt_tokens,
            "completion_tokens": self._total_completion_tokens,
            "total_tokens": self._total_prompt_tokens + self._total_completion_tokens,
            "estimated_cost_usd": round(self._total_cost, 6),
        }

    @property
    def cache_stats(self) -> dict[str, Any]:
        """Return cache performance metrics for cost-optimization reporting."""
        total_lookups = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total_lookups * 100) if total_lookups > 0 else 0.0

        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_entries": len(self._cache),
            "hit_rate_percent": round(hit_rate, 1),
            "llm_calls_saved": self._cache_hits,
            "estimated_time_saved_seconds": round(self._time_saved_seconds, 2),
        }

    def inject_cache(self, cache: dict[str, dict[str, Any]]) -> None:
        """Pre-load an external cache (e.g. from a previous run or persistent store)."""
        self._cache.update(cache)
        logger.info("Injected %d entries into enrichment cache", len(cache))

    def export_cache(self) -> dict[str, dict[str, Any]]:
        """Export the current cache for external persistence."""
        return dict(self._cache)

    # ── Core API ───────────────────────────────────────────────────────────

    def enrich(self, chunks: list[SmartChunk]) -> list[SmartChunk]:
        """Enrich chunks synchronously (runs async loop internally)."""
        if not chunks:
            return chunks

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # We're already in an async context — use thread
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(asyncio.run, self._enrich_all(chunks))
                future.result()
        else:
            asyncio.run(self._enrich_all(chunks))

        # Link neighbor summaries
        self._link_neighbors(chunks)

        return chunks

    async def _enrich_all(self, chunks: list[SmartChunk]) -> None:
        """Enrich all chunks with controlled concurrency and caching."""
        semaphore = asyncio.Semaphore(self.config.max_concurrency)

        async def _limited_enrich(chunk: SmartChunk) -> None:
            async with semaphore:
                await self._enrich_single_cached(chunk)

        # Process in batches for logging
        batch_size = self.config.batch_size
        total = len(chunks)

        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch = chunks[batch_start:batch_end]

            logger.info(
                "Enriching chunks %d–%d of %d",
                batch_start + 1,
                batch_end,
                total,
            )

            tasks = [_limited_enrich(chunk) for chunk in batch]
            await asyncio.gather(*tasks)

        # Persist cache to disk after all batches completed
        try:
            with open(self._cache_file, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, indent=2)
            logger.debug("Saved persistent cache with %d entries to %s", len(self._cache), self._cache_file)
        except Exception as e:
            logger.warning("Failed to save persistent cache to %s: %s", self._cache_file, e)

    # ── Caching Layer ──────────────────────────────────────────────────────

    @staticmethod
    def _hash_text(text: str) -> str:
        """Compute a stable SHA-256 hash of the chunk text."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    async def _enrich_single_cached(self, chunk: SmartChunk) -> None:
        """Check cache before calling the LLM — skip if we've seen this text."""
        text_hash = self._hash_text(chunk.text)

        cached = self._cache.get(text_hash)
        if cached is not None:
            self._cache_hits += 1
            self._time_saved_seconds += 1.5  # Estimate ~1.5s saved per cache hit
            self._apply_enrichment(chunk, cached)
            chunk.cache_status = "HIT"
            logger.debug("Cache HIT for chunk %s (hash=%s…)", chunk.id, text_hash[:8])
            return

        self._cache_misses += 1

        # Call LLM and store result in cache
        data = await self._enrich_single_llm(chunk)
        if data is not None:
            self._cache[text_hash] = data

    async def _enrich_single_llm(self, chunk: SmartChunk) -> dict[str, Any] | None:
        """Call the LLM to enrich a single chunk. Returns parsed data or None."""

        prompt = build_enrichment_prompt(
            chunk_text=chunk.text,
            parent_context=chunk.parent_context,
        )

        for attempt in range(self.config.max_retries):
            try:
                response = await acompletion(
                    model=self.config.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.config.temperature,
                    response_format={"type": "json_object"},
                )

                # Track usage
                usage = getattr(response, "usage", None)
                if usage:
                    self._total_prompt_tokens += getattr(usage, "prompt_tokens", 0)
                    self._total_completion_tokens += getattr(usage, "completion_tokens", 0)

                # Parse response
                content = response.choices[0].message.content  # type: ignore[union-attr]
                data = self._parse_response(content)
                self._apply_enrichment(chunk, data)

                logger.debug("Enriched chunk %s", chunk.id)
                return data

            except json.JSONDecodeError as exc:
                wait = 2**attempt
                logger.warning(
                    "JSON parse error on chunk %s (attempt %d/%d): %s — retrying in %ds",
                    chunk.id,
                    attempt + 1,
                    self.config.max_retries,
                    exc,
                    wait,
                )
                await asyncio.sleep(wait)
            except Exception as exc:
                wait = 2**attempt
                logger.warning(
                    "LLM error on chunk %s (attempt %d/%d): %s — retrying in %ds",
                    chunk.id,
                    attempt + 1,
                    self.config.max_retries,
                    exc,
                    wait,
                )
                await asyncio.sleep(wait)

        logger.error("Failed to enrich chunk %s after %d attempts", chunk.id, self.config.max_retries)
        return None

    # ── Parsing & Application ──────────────────────────────────────────────

    def _parse_response(self, content: str) -> dict[str, Any]:
        """Parse and validate the LLM JSON response."""
        content = content.strip()
        
        # 1. Try direct parsing
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # 2. Try to strip markdown code blocks (e.g. ```json ... ``` or ``` ... ```)
        import re
        pattern = r"```(?:json)?\s*(.*?)\s*```"
        match = re.search(pattern, content, re.DOTALL)
        if match:
            code_content = match.group(1).strip()
            try:
                return json.loads(code_content)
            except json.JSONDecodeError:
                content = code_content  # Fallback to further cleaning on code_content

        # 3. Find first '{' and last '}' to extract raw JSON object
        first_brace = content.find("{")
        last_brace = content.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            json_candidate = content[first_brace:last_brace + 1]
            try:
                return json.loads(json_candidate)
            except json.JSONDecodeError:
                pass

        # 4. Final attempt: raise original parse error or fallback
        return json.loads(content)

    def _apply_enrichment(self, chunk: SmartChunk, data: dict[str, Any]) -> None:
        """Apply parsed enrichment data to a SmartChunk."""
        fields = self.config.enrichments

        # Normalize keys to lowercase for case-insensitive lookup
        norm_data = {k.lower(): v for k, v in data.items()}

        if EnrichmentField.SUMMARY in fields:
            summary_val = norm_data.get("summary") or norm_data.get("description")
            if summary_val is not None:
                chunk.summary = str(summary_val)

        if EnrichmentField.ENTITIES in fields:
            entities_val = (
                norm_data.get("entities") or 
                norm_data.get("named_entities") or 
                norm_data.get("named entities")
            )
            if entities_val is not None:
                if isinstance(entities_val, list):
                    chunk.entities = [str(e) for e in entities_val]
                else:
                    chunk.entities = [str(entities_val)]

        if EnrichmentField.KEYWORDS in fields:
            keywords_val = norm_data.get("keywords") or norm_data.get("tags")
            if keywords_val is not None:
                if isinstance(keywords_val, list):
                    chunk.keywords = [str(k) for k in keywords_val]
                else:
                    chunk.keywords = [str(keywords_val)]

        if EnrichmentField.CONFIDENCE in fields:
            confidence_val = norm_data.get("confidence") or norm_data.get("score")
            if confidence_val is not None:
                try:
                    confidence = float(confidence_val)
                    chunk.confidence = max(0.0, min(1.0, confidence))
                except (ValueError, TypeError):
                    chunk.confidence = 0.0

    def _link_neighbors(self, chunks: list[SmartChunk]) -> None:
        """Populate prev_summary and next_summary for each chunk."""
        for i, chunk in enumerate(chunks):
            if i > 0 and chunks[i - 1].summary:
                chunk.prev_summary = chunks[i - 1].summary
            if i < len(chunks) - 1 and chunks[i + 1].summary:
                chunk.next_summary = chunks[i + 1].summary
