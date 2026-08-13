"""Prompt templates for LLM-based chunk enrichment.

All prompts are designed to produce structured JSON output
that can be validated against Pydantic models.
"""

SYSTEM_PROMPT = """\
You are a document analysis assistant. You enrich text chunks with structured metadata.
Always respond with valid JSON. Never include markdown fences or explanatory text outside the JSON.\
"""

ENRICHMENT_PROMPT = """\
Analyse the following text chunk and return a JSON object with these fields:

1. "summary": A single concise sentence summarising the main point of this chunk.
2. "entities": A list of named entities
   (people, organisations, monetary amounts, dates, locations, products).
   Extract only entities explicitly mentioned.
3. "keywords": A list of 3–8 semantic keywords that would help a search engine retrieve this chunk.
   Include synonyms and related terms not necessarily in the text.
4. "confidence": A float between 0.0 and 1.0 indicating how self-contained and atomic this chunk is.
   1.0 means it stands perfectly alone;
   0.0 means it's a fragment that makes no sense without context.

{context_instruction}

Text chunk:
\"\"\"
{chunk_text}
\"\"\"

Respond with ONLY a JSON object, no other text.\
"""

CONTEXT_INSTRUCTION_WITH_CONTEXT = """\
This chunk appears in the following document context:
Section: {parent_context}
This context should inform your summary but should not be repeated verbatim.\
"""

CONTEXT_INSTRUCTION_WITHOUT_CONTEXT = """\
No additional context is available for this chunk.\
"""


def build_enrichment_prompt(
    chunk_text: str,
    parent_context: str = "",
) -> str:
    """Build the enrichment prompt for a single chunk.

    Parameters
    ----------
    chunk_text:
        The raw text of the chunk.
    parent_context:
        The section/heading hierarchy, if available.

    Returns
    -------
    str
        The formatted prompt ready for LLM inference.
    """
    if parent_context:
        context_instruction = CONTEXT_INSTRUCTION_WITH_CONTEXT.format(parent_context=parent_context)
    else:
        context_instruction = CONTEXT_INSTRUCTION_WITHOUT_CONTEXT

    return ENRICHMENT_PROMPT.format(
        chunk_text=chunk_text,
        context_instruction=context_instruction,
    )
