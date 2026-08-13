"""Configuration management for SmartChunk.

Loads settings from environment variables and .env files,
producing a validated PipelineConfig with sensible defaults.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from smartchunk.models import (
    ChunkerConfig,
    ChunkStrategy,
    EnrichmentConfig,
    EnrichmentField,
    PipelineConfig,
)


def load_config(
    *,
    env_file: str | Path | None = None,
    **overrides: object,
) -> PipelineConfig:
    """Build a PipelineConfig from environment + explicit overrides.

    Priority (highest → lowest):
        1. Explicit keyword overrides
        2. Environment variables (SMARTCHUNK_*)
        3. .env file
        4. Built-in defaults

    Parameters
    ----------
    env_file:
        Path to a .env file. If ``None``, looks for ``.env`` in cwd.
    **overrides:
        Flat key-value overrides.  Recognised keys:
        ``model``, ``chunk_size``, ``chunk_overlap``, ``strategy``,
        ``enrich``, ``enrichments``, ``batch_size``, ``max_concurrency``,
        ``temperature``.
    """
    # Load .env (never overwrites already-set env vars)
    load_dotenv(dotenv_path=env_file or ".env", override=False)

    def _env(key: str, default: str | None = None) -> str | None:
        return os.environ.get(key, default)

    # ── Chunker config ─────────────────────────────────────────
    strategy_raw = overrides.get("strategy") or _env("SMARTCHUNK_STRATEGY", "recursive")
    if isinstance(strategy_raw, ChunkStrategy):
        strategy = strategy_raw
    else:
        val = str(strategy_raw).lower()
        if "." in val:
            val = val.split(".")[-1]
        strategy = ChunkStrategy(val)

    chunk_size = int(overrides.get("chunk_size") or _env("SMARTCHUNK_CHUNK_SIZE", "512"))
    chunk_overlap = int(overrides.get("chunk_overlap") or _env("SMARTCHUNK_CHUNK_OVERLAP", "50"))

    similarity_threshold = float(
        overrides.get("similarity_threshold")
        or _env("SMARTCHUNK_SIMILARITY_THRESHOLD", "0.75")
    )

    chunker = ChunkerConfig(
        strategy=strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        similarity_threshold=similarity_threshold,
    )

    # ── Enrichment config ──────────────────────────────────────
    model = str(overrides.get("model") or _env("SMARTCHUNK_MODEL", "gpt-4o-mini"))
    temperature = float(overrides.get("temperature") or _env("SMARTCHUNK_TEMPERATURE", "0.0"))
    batch_size = int(overrides.get("batch_size") or _env("SMARTCHUNK_BATCH_SIZE", "10"))
    max_concurrency = int(
        overrides.get("max_concurrency") or _env("SMARTCHUNK_MAX_CONCURRENCY", "5")
    )

    enrichments_raw = overrides.get("enrichments")
    if enrichments_raw is None:
        env_enrichments = _env("SMARTCHUNK_ENRICHMENTS")
        if env_enrichments:
            enrichments = [EnrichmentField(e.strip()) for e in env_enrichments.split(",")]
        else:
            enrichments = list(EnrichmentField)
    else:
        enrichments = []
        for e in enrichments_raw:  # type: ignore[union-attr]
            if isinstance(e, EnrichmentField):
                enrichments.append(e)
            else:
                val = str(e).lower()
                if "." in val:
                    val = val.split(".")[-1]
                enrichments.append(EnrichmentField(val))

    enrichment = EnrichmentConfig(
        model=model,
        temperature=temperature,
        enrichments=enrichments,
        batch_size=batch_size,
        max_concurrency=max_concurrency,
    )

    # ── Pipeline config ────────────────────────────────────────
    enrich_raw = overrides.get("enrich")
    if enrich_raw is None:
        enrich = _env("SMARTCHUNK_ENRICH", "true").lower() in ("true", "1", "yes")
    else:
        enrich = bool(enrich_raw)

    return PipelineConfig(
        chunker=chunker,
        enrichment=enrichment,
        enrich=enrich,
    )
