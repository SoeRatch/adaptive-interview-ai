# src/data_acquisition/web/content_filter.py

import re
from src.data_acquisition.web.web_constants import (
    RELEVANT_KEYWORDS,
    HARD_BLOCK_KEYWORDS, 
    SOFT_BLOCK_KEYWORDS,
    UI_NOISE_WORDS,
    MIN_PARAGRAPH_LENGTH,
    MIN_PARAGRAPH_WORDS
)

def filter_system_design_content(text: str) -> str:
    """
    Keep only relevant paragraphs:
    - Paragraph contains at least one keyword
    OR
    - Paragraph is long enough (to avoid dropping actual content)
    - Drop paragraphs with blacklisted/commercial words
    - Preserve paragraph and inline line-break structure
    """
    # skip empty or trivially short raw content
    if not isinstance(text, str) or len(text) < 100:
        return ''

    # Normalize spacing
    text = re.sub(r'[ \t]+', ' ', text) # Replace multiple spaces/tabs with single space
    text = re.sub(r'\n{3,}', '\n\n', text) # Replace 3+ line breaks with 2 (paragraph separator)
    
    # Preserve paragraph breaks while cleaning UI noise
    cleaned_lines = []
    for line in text.splitlines():
        line_stripped = line.strip()

        if not line_stripped:
            cleaned_lines.append("") # preserve blank lines for paragraph structure
            continue

        line_lower = line_stripped.lower()
        line_words = line_lower.split()
        
        # Drop lines with less words and has a year but not actual sentences
        if (
            re.search(r'\b\d{4}\b', line_lower) and
            len(line_words) < 5 and 
            not re.search(r'[.!?]', line_lower)
            ):
            continue
        
        # Drop lines with less words and no punctuation
        if len(line_words) < 5 and not re.search(r'[.!?]', line_lower):
            continue

        cleaned_lines.append(line_stripped)
    text = "\n".join(cleaned_lines)


    # Split by paragraph (2+ newlines)
    paragraphs = re.split(r"\n{2,}", text)
    filtered_paragraphs = []
    
    for para in paragraphs:
        para_clean = para.strip()
        # Drop empty or trivial paragraphs
        if not para_clean or len(para_clean.split()) < 3:
            continue

        para_lower = para_clean.lower()

        # Skip short UI noise like “Home”, “Next”, etc.
        if any(bad in para_lower for bad in UI_NOISE_WORDS):
            continue

        # Skip promotional or irrelevant text
        if any(bad in para_lower for bad in HARD_BLOCK_KEYWORDS):
            continue
        # Skip only if short or non-technical
        if any(bad in para_lower for bad in SOFT_BLOCK_KEYWORDS):
            if len(para_lower) < 200 or not any(kw in para_lower for kw in RELEVANT_KEYWORDS):
                continue

        # ---  Footer pattern cleanup ---
        # Skip if it is just a year or copyright notice
        if re.fullmatch(r"\d{4}", para_lower):
            continue
        if re.match(r"^(©|\(c\)|copyright)", para_lower): # Starts with © or copyright
            continue
        
        # --- Keep useful content ---
        if any(kw in para_lower for kw in RELEVANT_KEYWORDS):
            filtered_paragraphs.append(para_clean)

        # Otherwise, keep long or dense reasoning paragraphs
        elif (
            len(para_clean) >= MIN_PARAGRAPH_LENGTH or 
            len(para_lower.split()) >= MIN_PARAGRAPH_WORDS
            ):
            filtered_paragraphs.append(para_clean)

    # Deduplicate while preserving order
    seen = set()
    unique_paragraphs = []
    for para in filtered_paragraphs:
        if para not in seen:
            seen.add(para)
            unique_paragraphs.append(para)

    # Join with paragraph separation preserved
    cleaned_text = "\n\n".join(unique_paragraphs)
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text).strip()  # ensure consistent spacing
    return cleaned_text