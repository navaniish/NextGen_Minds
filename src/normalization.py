"""
normalization.py — Pure normalization functions for business name, address,
and country fields.

Design principles
─────────────────
1. Each function is PURE: same input → same output, no side effects.
2. Returns None for any form of missing input (never empty string).
3. Does NOT invent information. Only formatting is changed.
4. Does NOT delete records or filter by country.
5. Unicode-aware: NFKC normalization is applied before any regex work.
6. Compatible with PyArrow-backed pandas (uses Python re, not str.match).
"""

import re
import unicodedata
from typing import Optional

# Import mapping tables from config
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    LEGAL_SUFFIX_MAP,
    ADDRESS_ABBREV_MAP,
    MISSING_SENTINELS,
)

# ─── Pre-compile all regex patterns once ─────────────────────────────
# This gives a large speedup when called millions of times.
_LEGAL_SUFFIX_RE  = [(re.compile(pat, re.IGNORECASE), repl)
                     for pat, repl in LEGAL_SUFFIX_MAP]
_ADDRESS_ABBREV_RE = [(re.compile(pat, re.IGNORECASE), repl)
                      for pat, repl in ADDRESS_ABBREV_MAP]

_MULTI_SPACE_RE   = re.compile(r"\s+")
_MULTI_PUNCT_RE   = re.compile(r"([!?,;:]{2,})")   # repeated punctuation
_LEADING_JUNK_RE  = re.compile(r"^[\s\W]+")         # leading non-word chars
_TRAILING_JUNK_RE = re.compile(r"[\s\W]+$")         # trailing non-word chars


# ─────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────

def _is_missing(value: Optional[str]) -> bool:
    """Return True for any form of missing: None, empty, whitespace, sentinel."""
    if value is None:
        return True
    stripped = str(value).strip().lower()
    return stripped in MISSING_SENTINELS


def _nfkc(text: str) -> str:
    """Apply Unicode NFKC normalization (canonical decomposition + composition).

    This converts:
      - Fullwidth/halfwidth variants → ASCII equivalents
      - Ligatures (ﬁ → fi)
      - Superscripts (² → 2)
      - Composed accent forms remain stable

    Non-ASCII scripts (Devanagari, Tamil, Gujarati, Arabic) are preserved
    because NFKC does not transliterate between scripts.
    """
    return unicodedata.normalize("NFKC", text)


# ─────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────

def normalize_business_name(name: Optional[str]) -> Optional[str]:
    """Normalize a business name for entity-resolution blocking and matching.

    Transformations applied (in order):
    1.  Missing check        → return None
    2.  NFKC unicode         → canonical unicode form
    3.  Lowercase            → uniform case
    4.  & / + → "and"        → reduce symbol variants
    5.  Legal suffix map     → consistent suffix tokens (pvt ltd, llc, …)
    6.  Strip punctuation    → remove . , - ' " ( ) except inside words
    7.  Collapse whitespace  → single spaces only
    8.  Strip edges          → no leading/trailing spaces

    NOT applied:
    - Transliteration (Hindi → Roman). Kept for downstream embedding models.
    - Stopword removal. Short words carry identity information.
    - Stemming / lemmatization. Morphological changes risk destroying names.

    Args:
        name: Raw business_name string from dataset.

    Returns:
        Normalized string, or None if input is missing/empty.
    """
    if _is_missing(name):
        return None

    s = _nfkc(str(name))

    # Lowercase
    s = s.lower()

    # Replace & and + with "and"
    s = s.replace("&", " and ")
    s = s.replace("+", " and ")

    # Apply legal suffix normalization (compound patterns first, then single)
    for pattern, replacement in _LEGAL_SUFFIX_RE:
        s = pattern.sub(replacement, s)

    # Remove repeated special punctuation  e.g. "---", "..."
    s = _MULTI_PUNCT_RE.sub("", s)

    # Remove common punctuation that adds noise but not identity.
    # Keep: letters, digits, spaces, and Unicode word characters.
    # Remove: . , - _ ' " ( ) / \ @ # * ^ ~ ` ; : ! ?
    s = re.sub(r"[.,\-_'\"()/\\@#*^~`;:!?]", " ", s)

    # Collapse whitespace
    s = _MULTI_SPACE_RE.sub(" ", s).strip()

    # If normalization left nothing meaningful, return None
    if not s:
        return None

    return s


def normalize_business_address(address: Optional[str]) -> Optional[str]:
    """Normalize a business address for entity-resolution blocking and matching.

    Transformations applied (in order):
    1.  Missing check          → return None
    2.  NFKC unicode           → canonical unicode form
    3.  Lowercase              → uniform case
    4.  Address abbreviations  → expand road-type abbreviations (Rd→road, St→street)
    5.  Punctuation removal    → remove . , # etc. (keep digits and letters)
    6.  Collapse whitespace    → single spaces
    7.  Strip edges

    NOT applied:
    - Geocoding / address standardization (no external API used)
    - State-code expansion  (e.g. CA → California) — risks ambiguity
    - ZIP code reformatting — kept as-is for numeric matching
    - Inventing missing street numbers or city names

    Limitation:
    - "St" is expanded to "street" even when it means "Saint" (e.g. St. Louis).
      This is acceptable because both reference and noisy records receive the
      same transformation, preserving relative consistency.

    Args:
        address: Raw business_address string from dataset.

    Returns:
        Normalized string, or None if input is missing/empty.
    """
    if _is_missing(address):
        return None

    s = _nfkc(str(address))

    # Lowercase
    s = s.lower()

    # Apply address abbreviation expansion
    for pattern, replacement in _ADDRESS_ABBREV_RE:
        s = pattern.sub(replacement, s)

    # Remove punctuation that creates token-boundary noise.
    # Keep commas as they separate address components — then normalize to space.
    s = re.sub(r"[.,\-_'\"()/\\@*^~`;:!?]", " ", s)

    # Normalize commas → space (address components separated by space is fine for BM25)
    s = s.replace(",", " ")

    # Collapse whitespace
    s = _MULTI_SPACE_RE.sub(" ", s).strip()

    if not s:
        return None

    return s


def normalize_country(country: Optional[str]) -> Optional[str]:
    """Normalize a country field.

    Transformations:
    1.  Missing check  → return None
    2.  NFKC unicode   → canonical unicode form
    3.  Strip          → remove leading/trailing whitespace
    4.  Lowercase      → uniform case

    NOT applied:
    - Country code expansion (US → United States) — not needed for matching
    - Filtering / whitelist checking — test set may contain France or others

    The normalized value is ONLY used as a feature signal.
    It is never used to DROP records from the pipeline.

    Args:
        country: Raw country string from dataset.

    Returns:
        Normalized string (e.g. "us", "india", "france"), or None if missing.
    """
    if _is_missing(country):
        return None

    s = _nfkc(str(country))
    s = s.strip().lower()

    if not s:
        return None

    return s


# ─────────────────────────────────────────────────────────────────────
# VECTORIZED WRAPPERS (for pandas apply)
# ─────────────────────────────────────────────────────────────────────

def normalize_name_series(series):
    """Apply normalize_business_name to an entire pandas Series efficiently."""
    return series.apply(normalize_business_name)


def normalize_address_series(series):
    """Apply normalize_business_address to an entire pandas Series efficiently."""
    return series.apply(normalize_business_address)


def normalize_country_series(series):
    """Apply normalize_country to an entire pandas Series efficiently."""
    return series.apply(normalize_country)
