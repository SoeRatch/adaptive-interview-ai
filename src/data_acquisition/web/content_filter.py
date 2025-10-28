# src/data_acquisition/web/content_filter.py

import re
from src.data_acquisition.web.web_constants import RELEVANT_KEYWORDS, UNWANTED_KEYWORDS, MIN_PARAGRAPH_LENGTH

def filter_system_design_content(text: str) -> str:
    """
    Keep only relevant paragraphs:
    - Paragraph contains at least one keyword
    OR
    - Paragraph is long enough (to avoid dropping actual content)
    - Drop paragraphs with blacklisted/commercial words
    """
    filtered_lines = []
    # skip empty or trivially short raw content
    if not isinstance(text, str) or len(text) < 100:
        return filtered_lines

    for line in text.splitlines():
        line_lower = line.lower().strip()
        if not line_lower:
            continue

        # Skip obvious commercial / navigational content
        if any(bad in line_lower for bad in UNWANTED_KEYWORDS):
            continue

        # Keep if keyword exists
        if any(kw in line_lower for kw in RELEVANT_KEYWORDS):
            filtered_lines.append(line)
        
        # Otherwise, keep long paragraphs (likely real content)
        elif len(line_lower) >= MIN_PARAGRAPH_LENGTH:
            filtered_lines.append(line)

    return "\n".join(filtered_lines)
