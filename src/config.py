"""
config.py — Central configuration for the Amazon ML Challenge pipeline.

All paths, constants, and normalization settings live here so that every
other module imports from a single source of truth.
"""

import os

# ─── Root Paths ──────────────────────────────────────────────────────
PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_TRAIN_DIR = os.path.join(PROJECT_ROOT, "DATASETS", "TRAIN")
RAW_TEST_DIR  = os.path.join(PROJECT_ROOT, "DATASETS", "TEST")

CLEAN_DIR       = os.path.join(PROJECT_ROOT, "clean_dataset")
CLEAN_TRAIN_DIR = os.path.join(CLEAN_DIR, "train")
CLEAN_TEST_DIR  = os.path.join(CLEAN_DIR, "test")
CANDIDATES_DIR  = os.path.join(PROJECT_ROOT, "candidates")
FEATURES_DIR    = os.path.join(PROJECT_ROOT, "features")
SUBMISSION_DIR  = os.path.join(PROJECT_ROOT, "output")
REPORTS_DIR     = os.path.join(PROJECT_ROOT, "reports")

CANDIDATES_TRAIN_FILE = os.path.join(CANDIDATES_DIR, "candidates_train.parquet")
CANDIDATES_TEST_FILE  = os.path.join(CANDIDATES_DIR, "candidates_test.parquet")
FEATURES_TRAIN_FILE   = os.path.join(FEATURES_DIR,   "features_train.parquet")
FEATURES_TEST_FILE    = os.path.join(FEATURES_DIR,   "features_test.parquet")
SUBMISSION_FILE       = os.path.join(SUBMISSION_DIR, "matching_results.tsv")
CANDIDATE_PAIRS_FILE  = os.path.join(SUBMISSION_DIR, "candidate_pairs.tsv")
BLOCKING_REPORT       = os.path.join(REPORTS_DIR, "blocking_report.csv")
PIPELINE_REPORT       = os.path.join(REPORTS_DIR, "pipeline_report.json")

# ─── Raw File Names ───────────────────────────────────────────────────
RAW_SOURCE_FILES = {
    "source1": "train_source1.tsv",
    "source2": "train_source2.tsv",
    "source3": "train_source3.tsv",
}
RAW_GT_FILE       = "train_ground_truth.tsv"

TEST_SOURCE_FILES = {
    "source1": "test_source1.tsv",
    "source2": "test_source2.tsv",
    "source3": "test_source3.tsv",
}

# ─── Clean File Names ─────────────────────────────────────────────────
CLEAN_SOURCE_FILES = {
    "source1": "train_source1_clean.tsv",
    "source2": "train_source2_clean.tsv",
    "source3": "train_source3_clean.tsv",
}
CLEAN_GT_FILE = "train_ground_truth.tsv"   # copied as-is, no changes

CLEAN_TEST_FILES = {
    "source1": "test_source1_clean.tsv",
    "source2": "test_source2_clean.tsv",
    "source3": "test_source3_clean.tsv",
}

# ─── Processing ───────────────────────────────────────────────────────
# Chunk size for reading large TSV files.
# 100 000 rows ≈ 30–50 MB in memory; safe on any modern machine.
CHUNK_SIZE = 100_000

# TSV separator
SEP = "\t"

# ─── Column Names ─────────────────────────────────────────────────────
ENTITY_ID_COL    = "entity_id"
NAME_COL         = "business_name"
ADDRESS_COL      = "business_address"
COUNTRY_COL      = "country"

NAME_CLEAN_COL    = "business_name_clean"
ADDRESS_CLEAN_COL = "business_address_clean"
COUNTRY_CLEAN_COL = "country_clean"

SOURCE_COLS       = [ENTITY_ID_COL, NAME_COL, ADDRESS_COL, COUNTRY_COL]
CLEAN_EXTRA_COLS  = [NAME_CLEAN_COL, ADDRESS_CLEAN_COL, COUNTRY_CLEAN_COL]
OUTPUT_COLS       = SOURCE_COLS + CLEAN_EXTRA_COLS

# ─── Candidate / Feature Column Names ────────────────────────────────
S1_ID_COL         = "s1_id"           # Source 1 entity ID in candidate pairs
CAND_ID_COL       = "cand_id"         # Candidate entity ID from S2 or S3
SOURCE_TAG_COL    = "source"          # Which source the candidate comes from
MATCH_SCORE_COL   = "match_score"     # Model output probability
IS_MATCH_COL      = "is_match"        # Binary prediction (0 or 1)

# ─── Submission Column Names ─────────────────────────────────────────
SUBMISSION_ENTITY_COL   = "entity_id"
SUBMISSION_MATCHED_COL  = "matched_entity_ids"

# ─── Normalization Settings ───────────────────────────────────────────

# Legal suffix canonical mapping.
# Maps regex pattern (applied after lowercasing) → canonical form.
# Order matters: compound forms (e.g. "pvt ltd") must come before single tokens.
# All patterns are applied left-to-right in the order listed.
LEGAL_SUFFIX_MAP = [
    # ── Compound Indian suffixes (must precede single tokens) ──────────
    (r"\bprivate\s+limited\b",       "pvt ltd"),
    (r"\bpvt\.?\s+ltd\.?\b",         "pvt ltd"),
    (r"\bpvt\.?\s+limited\b",        "pvt ltd"),
    (r"\bprivate\s+ltd\.?\b",        "pvt ltd"),
    (r"\bprv\.?\s+ltd\.?\b",         "pvt ltd"),
    # ── Single token legal suffixes ───────────────────────────────────
    (r"\blimited\b",                  "ltd"),
    (r"\bltd\.?\b",                   "ltd"),
    (r"\bprivate\b",                  "pvt"),
    (r"\bpvt\.?\b",                   "pvt"),
    (r"\bincorporated\b",             "inc"),
    (r"\binc\.?\b",                   "inc"),
    (r"\bcorporation\b",              "corp"),
    (r"\bcorp\.?\b",                  "corp"),
    (r"\bcompany\b",                  "co"),
    # "co" alone is too short and risky — leave it as-is unless prefixed
    (r"\bl\.?l\.?c\.?\b",            "llc"),
    (r"\bl\.?l\.?p\.?\b",            "llp"),
    (r"\bplc\.?\b",                   "plc"),
    (r"\bgmbh\.?\b",                  "gmbh"),
    (r"\bsarl\.?\b",                  "sarl"),
    (r"\bs\.?a\.?s\.?\b",            "sas"),
    (r"\bs\.?a\.?\b",                "sa"),
    (r"\bb\.?v\.?\b",                "bv"),
    (r"\bn\.?v\.?\b",                "nv"),
    (r"\basse?ts?\b",                 "assets"),   # common shorthand
]

# Address abbreviation expansion mapping.
# Maps regex pattern → full form.
# Applied after lowercasing.
ADDRESS_ABBREV_MAP = [
    # ── Road types ────────────────────────────────────────────────────
    (r"\broad\b",         "road"),   # already full — keep stable
    (r"\brd\.?\b",        "road"),
    (r"\bstreet\b",       "street"),
    (r"\bst\.?\b",        "street"),  # NOTE: also maps "Saint" — acceptable
    (r"\bavenue\b",       "avenue"),
    (r"\bave\.?\b",       "avenue"),
    (r"\bboulevard\b",    "boulevard"),
    (r"\bblvd\.?\b",      "boulevard"),
    (r"\bdrive\b",        "drive"),
    (r"\bdr\.?\b",        "drive"),
    (r"\blane\b",         "lane"),
    (r"\bln\.?\b",        "lane"),
    (r"\bcourt\b",        "court"),
    (r"\bct\.?\b",        "court"),
    (r"\bcircle\b",       "circle"),
    (r"\bcir\.?\b",       "circle"),
    (r"\bplace\b",        "place"),
    (r"\bpl\.?\b",        "place"),
    (r"\bterrace\b",      "terrace"),
    (r"\bter\.?\b",       "terrace"),
    (r"\bhighway\b",      "highway"),
    (r"\bhwy\.?\b",       "highway"),
    (r"\bfreeway\b",      "freeway"),
    (r"\bfwy\.?\b",       "freeway"),
    (r"\bparkway\b",      "parkway"),
    (r"\bpkwy\.?\b",      "parkway"),
    (r"\bexpressway\b",   "expressway"),
    (r"\bexpy\.?\b",      "expressway"),
    (r"\btrail\b",        "trail"),
    (r"\btrl\.?\b",       "trail"),
    (r"\balley\b",        "alley"),
    (r"\baly\.?\b",       "alley"),
    # ── Unit / suite ──────────────────────────────────────────────────
    (r"\bapartment\b",    "apt"),
    (r"\bapt\.?\b",       "apt"),
    (r"\bsuite\b",        "ste"),
    (r"\bste\.?\b",       "ste"),
    (r"\bunit\b",         "unit"),
    (r"\bfloor\b",        "fl"),
    (r"\bfl\.?\b",        "fl"),
    # ── Number markers ────────────────────────────────────────────────
    (r"\bnumber\b",       "no"),
    (r"\bno\.?\b",        "no"),
    (r"#",               "no "),    # "#123" → "no 123"
    # ── Compass directions ────────────────────────────────────────────
    (r"\bnorth\b",        "n"),
    (r"\bsouth\b",        "s"),
    (r"\beast\b",         "e"),
    (r"\bwest\b",         "w"),
    (r"\bnortheast\b",    "ne"),
    (r"\bnorthwest\b",    "nw"),
    (r"\bsoutheast\b",    "se"),
    (r"\bsouthwest\b",    "sw"),
]

# Missing value sentinel strings (case-insensitive).
MISSING_SENTINELS = {"", "nan", "null", "none", "na", "n/a", "nil", "missing"}

# Report file path
NORMALIZATION_REPORT = os.path.join(REPORTS_DIR, "normalization_report.csv")
