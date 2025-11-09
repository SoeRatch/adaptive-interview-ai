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

def coerce_numeric_fields(result: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize numeric-like 'score' fields to floats when possible."""
    if not isinstance(result, dict):
        return result
    if "score" in result and isinstance(result["score"], str):
        # allow "8", "8/10", "8.0", "score: 8"
        cleaned = re.sub(r"[^\d.]", "", result["score"])
        try:
            result["score"] = float(cleaned) if cleaned else None
        except ValueError:
            result["score"] = None
    return result
