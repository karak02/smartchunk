"""Enrichment — LLM-powered metadata generation for chunks."""

from smartchunk.enrichment.base import BaseEnricher
from smartchunk.enrichment.llm import LLMEnricher

__all__ = ["BaseEnricher", "LLMEnricher"]
