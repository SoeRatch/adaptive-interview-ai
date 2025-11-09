import re
import json
from typing import Dict, Any, Optional, List

def strip_code_fences(text: str) -> str:
    """Remove markdown-style code fences like ```json ... ``` (both fenced blocks and single-line)."""
    if not isinstance(text, str):
        return text
    # Remove leading ```json or ``` and trailing ```
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.DOTALL)
    text = re.sub(r"\s*```$", "", text, flags=re.DOTALL)
    return text.strip()

def safe_json_parse(text: str) -> Dict[str, Any]:
    """Try to parse text as JSON; fallback to extracting the first {...} block."""
    if not isinstance(text, str) or not text.strip():
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return {}
    return {}

def coerce_types(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    General-purpose coercion to normalize LLM outputs to match EVALUATION_SCHEMA.
    Fixes:
      - Converts numeric-like strings to floats (e.g., "8/10" -> 8.0)
      - Converts stringified lists to real lists (e.g., "['a', 'b']" or "a. b." -> ["a", "b"])
      - Converts empty strings ("") to None
      - Ensures nullables stay None instead of invalid strings
    """
    if not isinstance(result, dict):
        return result

    for key, value in list(result.items()):
        # 1. Handle empty strings → None
        if isinstance(value, str) and not value.strip():
            result[key] = None
            continue

        # 2. Normalize numeric-like fields
        if key == "score" and isinstance(value, str):
            cleaned = re.sub(r"[^\d.]", "", value)
            try:
                result[key] = float(cleaned) if cleaned else None
            except ValueError:
                result[key] = None
            continue

        # 3. Normalize array-like fields (e.g., strengths/weaknesses)
        if key in ["strengths", "weaknesses"] and value is not None:
            if isinstance(value, str):
                # Try to parse if it's a JSON-like list string
                import ast
                try:
                    parsed = ast.literal_eval(value)
                    if isinstance(parsed, list):
                        result[key] = [str(v).strip() for v in parsed if str(v).strip()]
                    else:
                        result[key] = [v.strip() for v in value.split(".") if v.strip()]
                except Exception:
                    # fallback: split on punctuation
                    result[key] = [v.strip() for v in value.split(".") if v.strip()]

            elif isinstance(value, list):
                # ensure all items are strings
                result[key] = [str(v).strip() for v in value if str(v).strip()]

        # 4. Normalize enum-like fields
        if key == "next_action" and isinstance(value, str):
            val = value.strip().lower()
            if val not in ["retry_same_topic", "next_topic_question"]:
                result[key] = "retry_same_topic"  # fallback default

    return result

