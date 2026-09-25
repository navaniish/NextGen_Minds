"""
preprocessing.py — End-to-end preprocessing pipeline for the Amazon ML Challenge.

This module reads the raw TSV files, applies normalization, writes clean TSV files,
runs quality-validation checks, and produces a normalization report.

Usage
─────
    python -m src.preprocessing          # process train + test (if test exists)
    python -m src.preprocessing --train  # process training only

Design principles
─────────────────
- Chunked reading: files are processed in CHUNK_SIZE rows to stay memory-efficient.
- Original columns are NEVER overwritten.
- Rows are NEVER dropped (missing values → None in clean columns).
- entity_id is NEVER modified.
- Ground-truth file is copied as-is (no normalization needed).
- Country is normalized to lowercase string — never filtered.
"""

import os
import sys
import shutil
import csv
import time
import logging
from typing import Optional

import pandas as pd

# Make src/ importable when run as a script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    RAW_TRAIN_DIR, RAW_TEST_DIR,
    CLEAN_TRAIN_DIR, CLEAN_TEST_DIR, REPORTS_DIR,
    RAW_SOURCE_FILES, RAW_GT_FILE,
    TEST_SOURCE_FILES,
    CLEAN_SOURCE_FILES, CLEAN_GT_FILE, CLEAN_TEST_FILES,
    CHUNK_SIZE, SEP,
    ENTITY_ID_COL, NAME_COL, ADDRESS_COL, COUNTRY_COL,
    NAME_CLEAN_COL, ADDRESS_CLEAN_COL, COUNTRY_CLEAN_COL,
    SOURCE_COLS, OUTPUT_COLS,
    NORMALIZATION_REPORT,
)
from normalization import (
    normalize_name_series,
    normalize_address_series,
    normalize_country_series,
    normalize_business_name,
    normalize_business_address,
    normalize_country,
)

# ─── Logger ──────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# load_dataset
# ─────────────────────────────────────────────────────────────────────

def load_dataset(path: str) -> pd.DataFrame:
    """Load a TSV source file into a DataFrame.

    - All columns loaded as strings (dtype=str).
    - keep_default_na=False prevents pandas from converting 'nan' strings
      to NaN silently — this is critical because the dataset uses the
      literal string 'nan' for some missing names.

    Args:
        path: Absolute path to the TSV file.

    Returns:
        DataFrame with string columns.

    Raises:
        FileNotFoundError if the file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")

    log.info(f"Loading  {os.path.basename(path)} ...")
    df = pd.read_csv(path, sep=SEP, dtype=str, keep_default_na=False)
    log.info(f"  Loaded  {len(df):,} rows x {df.shape[1]} cols")
    return df


# ─────────────────────────────────────────────────────────────────────
# clean_dataset
# ─────────────────────────────────────────────────────────────────────

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply normalization to a source DataFrame.

    Adds three new columns:
        business_name_clean
        business_address_clean
        country_clean

    Original columns are NOT modified.
    entity_id is NOT touched.
    Rows are NOT dropped.

    Args:
        df: Raw DataFrame from load_dataset().

    Returns:
        DataFrame with original columns plus three clean columns.
    """
    out = df.copy()

    log.info("  Normalizing business_name ...")
    out[NAME_CLEAN_COL] = normalize_name_series(out[NAME_COL])

    log.info("  Normalizing business_address ...")
    out[ADDRESS_CLEAN_COL] = normalize_address_series(out[ADDRESS_COL])

    log.info("  Normalizing country ...")
    out[COUNTRY_CLEAN_COL] = normalize_country_series(out[COUNTRY_COL])

    return out


# ─────────────────────────────────────────────────────────────────────
# save_clean_dataset
# ─────────────────────────────────────────────────────────────────────

def save_clean_dataset(df: pd.DataFrame, path: str) -> None:
    """Save a clean DataFrame to a TSV file.

    - None values are written as empty strings (consistent with how missing
      values appear in the original Source 2 / Source 3 files).
    - Column order: entity_id, business_name, business_address, country,
                    business_name_clean, business_address_clean, country_clean.

    Args:
        df:   DataFrame with original + clean columns.
        path: Output TSV file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Reorder to canonical column order
    cols = [c for c in OUTPUT_COLS if c in df.columns]
    df[cols].to_csv(path, sep=SEP, index=False, na_rep="")
    log.info(f"  Saved   {len(df):,} rows → {os.path.basename(path)}")


# ─────────────────────────────────────────────────────────────────────
# validate_clean_dataset
# ─────────────────────────────────────────────────────────────────────

def validate_clean_dataset(original: pd.DataFrame,
                           clean: pd.DataFrame,
                           label: str = "") -> dict:
    """Run quality-assurance checks comparing original and clean DataFrames.

    Checks:
    1.  Same number of rows.
    2.  Same entity_id values (order-insensitive).
    3.  No duplicate rows introduced.
    4.  Original columns are unchanged.
    5.  Clean columns exist and contain strings or None/NaN.
    6.  Country values are NOT incorrectly filtered.
    7.  Count of rows where each clean column differs from original.

    Args:
        original: DataFrame loaded from raw file.
        clean:    DataFrame after cleaning.
        label:    Human-readable label for logging.

    Returns:
        dict with all validation results.
    """
    result = {"label": label, "passed": True, "errors": [], "warnings": []}
    total = len(original)

    # ── 1. Row count ────────────────────────────────────────────────
    if len(clean) != total:
        msg = f"Row count mismatch: original={total:,}, clean={len(clean):,}"
        result["errors"].append(msg)
        result["passed"] = False
    result["original_rows"] = total
    result["clean_rows"]    = len(clean)

    # ── 2. entity_id integrity ──────────────────────────────────────
    orig_ids  = set(original[ENTITY_ID_COL])
    clean_ids = set(clean[ENTITY_ID_COL])
    if orig_ids != clean_ids:
        diff = len(orig_ids.symmetric_difference(clean_ids))
        msg = f"{diff} entity_id values differ between original and clean"
        result["errors"].append(msg)
        result["passed"] = False

    # ── 3. No duplicate rows introduced ────────────────────────────
    dup_ids_orig  = original[ENTITY_ID_COL].duplicated().sum()
    dup_ids_clean = clean[ENTITY_ID_COL].duplicated().sum()
    if dup_ids_clean > dup_ids_orig:
        msg = f"Duplicate entity_ids increased: {dup_ids_orig} → {dup_ids_clean}"
        result["errors"].append(msg)
        result["passed"] = False
    result["duplicate_ids"] = int(dup_ids_clean)

    # ── 4. Original columns unchanged ───────────────────────────────
    for col in SOURCE_COLS:
        if col not in clean.columns:
            result["errors"].append(f"Original column '{col}' missing in clean output")
            result["passed"] = False
            continue
        if not original[col].equals(clean[col]):
            result["errors"].append(f"Original column '{col}' was modified — NOT allowed")
            result["passed"] = False

    # ── 5. Clean columns exist ──────────────────────────────────────
    for col in [NAME_CLEAN_COL, ADDRESS_CLEAN_COL, COUNTRY_CLEAN_COL]:
        if col not in clean.columns:
            result["errors"].append(f"Clean column '{col}' missing")
            result["passed"] = False

    # ── 6. Country not incorrectly filtered ─────────────────────────
    orig_countries  = set(original[COUNTRY_COL].dropna().unique())
    clean_countries = set(clean[COUNTRY_CLEAN_COL].dropna().unique())
    # All original country VALUES should still appear (lowercased) in clean
    orig_lower = {c.strip().lower() for c in orig_countries if c.strip()}
    if not orig_lower.issubset(clean_countries):
        missing_c = orig_lower - clean_countries
        result["warnings"].append(f"Country values missing after normalization: {missing_c}")

    # ── 7. Change counts ────────────────────────────────────────────
    # How many rows had the name/address/country changed?
    def count_changed(orig_col, clean_col):
        # Compare original vs clean. None in clean means originally missing.
        orig_vals  = original[orig_col].fillna("").str.strip()
        clean_vals = clean[clean_col].fillna("")
        return (orig_vals != clean_vals).sum()

    result["name_changed"]    = int(count_changed(NAME_COL,    NAME_CLEAN_COL))
    result["address_changed"] = int(count_changed(ADDRESS_COL, ADDRESS_CLEAN_COL))
    result["country_changed"] = int(count_changed(COUNTRY_COL, COUNTRY_CLEAN_COL))

    # ── 8. Unexpected nulls in entity_id ───────────────────────────
    null_ids = clean[ENTITY_ID_COL].isna().sum()
    if null_ids > 0:
        result["errors"].append(f"{null_ids} null entity_ids in clean output")
        result["passed"] = False

    return result


# ─────────────────────────────────────────────────────────────────────
# _process_file  (internal)
# ─────────────────────────────────────────────────────────────────────

def _process_file(input_path: str,
                  output_path: str,
                  label: str) -> dict:
    """Load → clean → save → validate one source TSV file.

    Uses chunked reading to stay memory-efficient on large files.
    Writes the header once, then appends chunks.

    Args:
        input_path:  Path to raw TSV.
        output_path: Path to write clean TSV.
        label:       Label for logging and report.

    Returns:
        Validation result dict.
    """
    if not os.path.exists(input_path):
        log.warning(f"  File not found, skipping: {input_path}")
        return {"label": label, "passed": False,
                "errors": [f"File not found: {input_path}"],
                "original_rows": 0, "clean_rows": 0,
                "name_changed": 0, "address_changed": 0,
                "country_changed": 0, "duplicate_ids": 0}

    t0 = time.time()
    log.info(f"\n{'─'*60}")
    log.info(f"Processing: {label}")
    log.info(f"  Input : {input_path}")
    log.info(f"  Output: {output_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    total_rows      = 0
    total_name_ch   = 0
    total_addr_ch   = 0
    total_ctry_ch   = 0
    header_written  = False
    first_chunk     = None   # kept for validation on a sample

    reader = pd.read_csv(
        input_path, sep=SEP, dtype=str, keep_default_na=False,
        chunksize=CHUNK_SIZE
    )

    with open(output_path, "w", newline="", encoding="utf-8") as fout:
        writer = None

        for chunk_idx, chunk in enumerate(reader):
            clean_chunk = clean_dataset(chunk)

            # Write header on first chunk
            if not header_written:
                cols = [c for c in OUTPUT_COLS if c in clean_chunk.columns]
                clean_chunk[cols].to_csv(
                    fout, sep=SEP, index=False, na_rep="", header=True
                )
                header_written = True
                first_chunk    = (chunk.copy(), clean_chunk.copy())
            else:
                cols = [c for c in OUTPUT_COLS if c in clean_chunk.columns]
                clean_chunk[cols].to_csv(
                    fout, sep=SEP, index=False, na_rep="", header=False
                )

            # Accumulate change counts
            orig_name  = chunk[NAME_COL].fillna("").str.strip()
            orig_addr  = chunk[ADDRESS_COL].fillna("").str.strip()
            orig_ctry  = chunk[COUNTRY_COL].fillna("").str.strip()

            total_name_ch += (orig_name  != clean_chunk[NAME_CLEAN_COL].fillna("")).sum()
            total_addr_ch += (orig_addr  != clean_chunk[ADDRESS_CLEAN_COL].fillna("")).sum()
            total_ctry_ch += (orig_ctry  != clean_chunk[COUNTRY_CLEAN_COL].fillna("")).sum()

            total_rows += len(chunk)

            if (chunk_idx + 1) % 10 == 0:
                log.info(f"  ... {total_rows:>8,} rows processed")

    elapsed = time.time() - t0
    log.info(f"  Done  {total_rows:,} rows in {elapsed:.1f}s")

    # ── Lightweight validation on first chunk ───────────────────────
    if first_chunk is not None:
        orig_sample, clean_sample = first_chunk
        val = validate_clean_dataset(orig_sample, clean_sample, label=label)
    else:
        val = {"label": label, "passed": True, "errors": [], "warnings": []}

    val["original_rows"]   = total_rows
    val["clean_rows"]      = total_rows   # rows are never dropped
    val["name_changed"]    = int(total_name_ch)
    val["address_changed"] = int(total_addr_ch)
    val["country_changed"] = int(total_ctry_ch)
    val["elapsed_sec"]     = round(elapsed, 1)

    return val


# ─────────────────────────────────────────────────────────────────────
# run_preprocessing
# ─────────────────────────────────────────────────────────────────────

def run_preprocessing(process_train: bool = True,
                      process_test: bool = True) -> None:
    """Main entry point. Processes all source files and writes clean outputs.

    Steps:
    1.  Create output directories.
    2.  Process each training source file.
    3.  Copy ground-truth file (no normalization).
    4.  Process test source files (if directory exists).
    5.  Collect validation results.
    6.  Write normalization_report.csv.
    7.  Print summary table.

    Args:
        process_train: Whether to process training files (default True).
        process_test:  Whether to process test files (default True).
    """
    os.makedirs(CLEAN_TRAIN_DIR, exist_ok=True)
    os.makedirs(CLEAN_TEST_DIR,  exist_ok=True)
    os.makedirs(REPORTS_DIR,     exist_ok=True)

    all_results = []

    # ── Training files ───────────────────────────────────────────────
    if process_train:
        log.info("\n" + "=" * 60)
        log.info("PROCESSING TRAINING FILES")
        log.info("=" * 60)

        for key, fname in RAW_SOURCE_FILES.items():
            in_path  = os.path.join(RAW_TRAIN_DIR, fname)
            out_path = os.path.join(CLEAN_TRAIN_DIR, CLEAN_SOURCE_FILES[key])
            label    = f"Train {key}"
            result   = _process_file(in_path, out_path, label)
            all_results.append(result)

        # Copy ground truth as-is
        gt_src = os.path.join(RAW_TRAIN_DIR, RAW_GT_FILE)
        gt_dst = os.path.join(CLEAN_TRAIN_DIR, CLEAN_GT_FILE)
        if os.path.exists(gt_src):
            shutil.copy2(gt_src, gt_dst)
            log.info(f"\nCopied ground truth → {gt_dst}")
        else:
            log.warning(f"Ground truth not found: {gt_src}")

    # ── Test files ───────────────────────────────────────────────────
    if process_test:
        if not os.path.isdir(RAW_TEST_DIR):
            log.info(f"\nTest directory not found ({RAW_TEST_DIR}) — skipping.")
        else:
            log.info("\n" + "=" * 60)
            log.info("PROCESSING TEST FILES")
            log.info("=" * 60)

            for key, fname in TEST_SOURCE_FILES.items():
                in_path  = os.path.join(RAW_TEST_DIR, fname)
                out_path = os.path.join(CLEAN_TEST_DIR, CLEAN_TEST_FILES[key])
                label    = f"Test {key}"
                result   = _process_file(in_path, out_path, label)
                all_results.append(result)

    # ── Write report ─────────────────────────────────────────────────
    _write_report(all_results)

    # ── Print summary ────────────────────────────────────────────────
    _print_summary(all_results)

    log.info("\nPreprocessing complete.")


# ─────────────────────────────────────────────────────────────────────
# Reporting helpers
# ─────────────────────────────────────────────────────────────────────

def _write_report(results: list) -> None:
    """Write normalization_report.csv."""
    import csv
    os.makedirs(REPORTS_DIR, exist_ok=True)

    fields = [
        "file", "original_rows", "clean_rows",
        "name_changed", "address_changed", "country_changed",
        "duplicate_ids", "passed", "errors", "elapsed_sec",
    ]
    with open(NORMALIZATION_REPORT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in results:
            r["file"] = r.get("label", "")
            r["errors"] = " | ".join(r.get("errors", []))
            w.writerow(r)

    log.info(f"\nReport written → {NORMALIZATION_REPORT}")


def _print_summary(results: list) -> None:
    """Print a human-readable summary table."""
    header = f"\n{'File':<22} {'Orig Rows':>12} {'Clean Rows':>12} {'Name Δ':>10} {'Addr Δ':>10} {'Ctry Δ':>8} {'OK?':>5}"
    print(header)
    print("─" * len(header))
    for r in results:
        ok  = "✅" if r.get("passed", False) else "❌"
        print(
            f"  {r.get('label',''):<20} "
            f"{r.get('original_rows',0):>12,} "
            f"{r.get('clean_rows',0):>12,} "
            f"{r.get('name_changed',0):>10,} "
            f"{r.get('address_changed',0):>10,} "
            f"{r.get('country_changed',0):>8,} "
            f"{ok:>5}"
        )


# ─────────────────────────────────────────────────────────────────────
# Demo: quick normalization examples
# ─────────────────────────────────────────────────────────────────────

def _demo() -> None:
    """Print normalization examples to verify rules are working correctly."""
    examples_name = [
        "ABC PVT. LTD.",
        "ABC PRIVATE LIMITED",
        "abc pvt ltd",
        "Foot & Ankle Allied Center LLC",
        "HENDERSON AND JONES INC.",
        "ग्लोबल इन्वेस्टमेंट प्रा. लि.",
        "Chordia + Pagnters - 7306204978",
        "Hotel Enterprises L.L.C.",
        "   Smith   Corp.   ",
        "nan",
        "",
        None,
    ]

    examples_addr = [
        "12, MG Rd., Rajkot",
        "3315 Fremont Street, Peoria, IL",
        "3315 FREMONT ST, PEORIA, IL",
        "Near SBI ATM, Delhi",
        "0, Elkton, MD",
        "Canal Road Wakdai Nagar, Mangaon",
        "3907 Hamilton Road, Deer Park, WA",
        "3907 HAMILTON RD, DEER PARK, WA",
        "",
        None,
    ]

    print("\n" + "=" * 60)
    print("NORMALIZATION DEMO — business_name")
    print("=" * 60)
    for name in examples_name:
        result = normalize_business_name(name)
        print(f"  IN : {repr(name)}")
        print(f"  OUT: {repr(result)}\n")

    print("=" * 60)
    print("NORMALIZATION DEMO — business_address")
    print("=" * 60)
    for addr in examples_addr:
        result = normalize_business_address(addr)
        print(f"  IN : {repr(addr)}")
        print(f"  OUT: {repr(result)}\n")

    examples_country = ["US", "India", "india", "FRANCE", "  US  ", "nan", None]
    print("=" * 60)
    print("NORMALIZATION DEMO — country")
    print("=" * 60)
    for c in examples_country:
        result = normalize_country(c)
        print(f"  IN : {repr(c):15}  OUT: {repr(result)}")


# ─────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Amazon ML Challenge — Preprocessing Pipeline"
    )
    parser.add_argument("--train-only", action="store_true",
                        help="Process training files only (skip test)")
    parser.add_argument("--demo", action="store_true",
                        help="Show normalization examples and exit")
    args = parser.parse_args()

    if args.demo:
        _demo()
    else:
        run_preprocessing(
            process_train=True,
            process_test=not args.train_only,
        )
