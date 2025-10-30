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
    # skip empty or trivially short raw content
    if not isinstance(text, str) or len(text) < 100:
        return ''

    filtered_lines = []

    # Replace multiple spaces/tabs with single space
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Replace 3+ line breaks with 2 (paragraph separator)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Strip trailing spaces/newlines
    text = text.strip()

    
    for line in text.splitlines():
        line_lower = line.lower().strip()
        if not line_lower:
            continue

        # Skip promotional or irrelevant text
        if any(bad in line_lower for bad in UNWANTED_KEYWORDS):
            continue

        # Keep if contains relevant keywords
        if any(kw in line_lower for kw in RELEVANT_KEYWORDS):
            filtered_lines.append(line)
        
        # Otherwise, keep long paragraphs (likely real content)
        elif len(line_lower) >= MIN_PARAGRAPH_LENGTH:
            filtered_lines.append(line)

    # Remove duplicates, join cleanly
    filtered_lines = list(dict.fromkeys(filtered_lines))

    # Join back preserving paragraph structure
    cleaned_text = "\n".join(filtered_lines)
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text).strip()  # ensure consistent spacing
    return cleaned_text
