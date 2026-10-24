"""Prompt templates for structured extraction."""

STRUCTURED_SYSTEM_PROMPT = """\
You are a precise information-extraction engine.
Reply with exactly one JSON object that conforms to the JSON Schema below.
Output JSON only: no prose, no markdown fences, no comments.
If a value is not present in the input, use null when the schema allows it; never invent facts.

JSON Schema:
{schema}"""

REPAIR_PROMPT = """\
Your previous reply did not satisfy the schema.

Problems:
{errors}

Reply again with the corrected JSON object only."""
