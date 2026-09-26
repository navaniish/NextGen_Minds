"""
validation.py — Pipeline-wide data validation for the Amazon ML Challenge.

Author: Navaneeswar (Pipeline Architecture & Integration)

Provides validation functions that check the structural integrity of data
at each stage of the pipeline. These are called by pipeline.py before
passing data to the next stage.

Design principles
─────────────────
- Never modifies data. Only inspects and reports.
- Every validation function returns a ValidationResult (passed, errors, warnings).
- Failures raise PipelineValidationError so the pipeline halts cleanly.
- Warnings are logged but do not stop the pipeline.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Set

import pandas as pd

# Allow running standalone or as part of the src/ package
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    ENTITY_ID_COL, NAME_COL, ADDRESS_COL, COUNTRY_COL,
    NAME_CLEAN_COL, ADDRESS_CLEAN_COL, COUNTRY_CLEAN_COL,
    S1_ID_COL, CAND_ID_COL, SOURCE_TAG_COL,
    MATCH_SCORE_COL, IS_MATCH_COL,
    SUBMISSION_ENTITY_COL, SUBMISSION_MATCHED_COL,
    SOURCE_COLS, OUTPUT_COLS,
)

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Result Type
# ─────────────────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    """Holds the outcome of a single validation check."""
    stage: str
    passed: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def fail(self, msg: str) -> None:
        self.passed = False
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def log_summary(self) -> None:
        status = "✅ PASSED" if self.passed else "❌ FAILED"
        log.info(f"  [{self.stage}] Validation {status}")
        for e in self.errors:
            log.error(f"    ERROR: {e}")
        for w in self.warnings:
            log.warning(f"    WARN : {w}")
        for k, v in self.stats.items():
            log.info(f"    STAT : {k} = {v}")


class PipelineValidationError(Exception):
    """Raised when a critical validation check fails."""
    pass


# ─────────────────────────────────────────────────────────────────────
# Stage 1: Validate Cleaned Dataset (Output of Saranya's Stage 3)
# ─────────────────────────────────────────────────────────────────────

def validate_clean_data(
    df: pd.DataFrame,
    label: str,
    expected_min_rows: int = 1,
) -> ValidationResult:
    """Validate a cleaned source DataFrame before feeding into the pipeline.

    Checks:
    - Required columns exist (original + clean columns)
    - No null entity_id values
    - Row count >= expected_min_rows
    - Clean columns are present
    - No duplicate entity_ids (warns, does not fail)

    Args:
        df:               Cleaned source DataFrame.
        label:            Human-readable label for logging (e.g. "Source1 Train").
        expected_min_rows: Minimum acceptable row count.

    Returns:
        ValidationResult
    """
    r = ValidationResult(stage=f"CleanData[{label}]")
    r.stats["rows"] = len(df)
    r.stats["cols"] = list(df.columns)

    # ── Required columns ──────────────────────────────────────────────
    required = [ENTITY_ID_COL, NAME_COL, ADDRESS_COL, COUNTRY_COL,
                NAME_CLEAN_COL, ADDRESS_CLEAN_COL, COUNTRY_CLEAN_COL]
    for col in required:
        if col not in df.columns:
            r.fail(f"Missing required column: '{col}'")

    # ── Null entity IDs ───────────────────────────────────────────────
    if ENTITY_ID_COL in df.columns:
        null_ids = df[ENTITY_ID_COL].isna().sum()
        if null_ids > 0:
            r.fail(f"{null_ids:,} null entity_ids found")
        r.stats["null_entity_ids"] = int(null_ids)

        dup_ids = df[ENTITY_ID_COL].duplicated().sum()
        if dup_ids > 0:
            r.warn(f"{dup_ids:,} duplicate entity_ids (may be acceptable for S2/S3)")
        r.stats["duplicate_entity_ids"] = int(dup_ids)

    # ── Row count ─────────────────────────────────────────────────────
    if len(df) < expected_min_rows:
        r.fail(f"Too few rows: expected >= {expected_min_rows:,}, got {len(df):,}")

    # ── Country coverage ──────────────────────────────────────────────
    if COUNTRY_CLEAN_COL in df.columns:
        countries = df[COUNTRY_CLEAN_COL].dropna().unique().tolist()
        r.stats["countries"] = sorted(countries)

    r.log_summary()
    return r


# ─────────────────────────────────────────────────────────────────────
# Stage 2: Validate Candidate Pairs (Output of Karthik's Blocking)
# ─────────────────────────────────────────────────────────────────────

def validate_candidates(
    candidates: pd.DataFrame,
    s1_ids: Set[str],
    label: str,
) -> ValidationResult:
    """Validate the candidate pair DataFrame produced by the blocking stage.

    Checks:
    - Required columns: s1_id, cand_id, source
    - All s1_ids from Source 1 have at least one candidate (warns if not)
    - No null s1_id or cand_id values
    - Source column contains valid values (source2, source3)
    - No duplicate (s1_id, cand_id) pairs

    Args:
        candidates: DataFrame with candidate pairs.
        s1_ids:     Set of all entity_ids from Source 1 (ground truth set).
        label:      Human-readable label for logging.

    Returns:
        ValidationResult
    """
    r = ValidationResult(stage=f"Candidates[{label}]")
    r.stats["candidate_pairs"] = len(candidates)

    # ── Required columns ──────────────────────────────────────────────
    required = [S1_ID_COL, CAND_ID_COL, SOURCE_TAG_COL]
    for col in required:
        if col not in candidates.columns:
            r.fail(f"Missing required column: '{col}'")

    if not r.passed:
        r.log_summary()
        return r

    # ── Null values ───────────────────────────────────────────────────
    null_s1 = candidates[S1_ID_COL].isna().sum()
    null_cand = candidates[CAND_ID_COL].isna().sum()
    if null_s1 > 0:
        r.fail(f"{null_s1:,} null values in s1_id column")
    if null_cand > 0:
        r.fail(f"{null_cand:,} null values in cand_id column")

    # ── Source values ─────────────────────────────────────────────────
    valid_sources = {"source2", "source3"}
    actual_sources = set(candidates[SOURCE_TAG_COL].dropna().unique())
    invalid_sources = actual_sources - valid_sources
    if invalid_sources:
        r.warn(f"Unexpected source values: {invalid_sources}")
    r.stats["sources_found"] = sorted(actual_sources)

    # ── Coverage: every S1 entity should have at least one candidate ──
    s1_covered = set(candidates[S1_ID_COL].unique())
    uncovered = s1_ids - s1_covered
    coverage_pct = 100.0 * len(s1_covered) / len(s1_ids) if s1_ids else 0.0
    r.stats["s1_coverage_pct"] = round(coverage_pct, 2)
    if uncovered:
        r.warn(f"{len(uncovered):,} Source 1 entities have NO candidates "
               f"({100.0 - coverage_pct:.1f}% uncovered)")

    # ── Duplicate pairs ───────────────────────────────────────────────
    dup_pairs = candidates.duplicated(subset=[S1_ID_COL, CAND_ID_COL]).sum()
    if dup_pairs > 0:
        r.warn(f"{dup_pairs:,} duplicate (s1_id, cand_id) pairs")
    r.stats["duplicate_pairs"] = int(dup_pairs)

    r.log_summary()
    return r


# ─────────────────────────────────────────────────────────────────────
# Stage 3: Validate Feature Matrix (Output of Feature Extraction)
# ─────────────────────────────────────────────────────────────────────

def validate_features(
    features: pd.DataFrame,
    candidates: pd.DataFrame,
    label: str,
) -> ValidationResult:
    """Validate the feature matrix produced by the feature extraction stage.

    Checks:
    - Required columns: s1_id, cand_id, source
    - Same row count as the candidate DataFrame it was derived from
    - No NaN in s1_id or cand_id (key columns must be intact)
    - At least one numeric feature column exists beyond the key columns

    Args:
        features:   Feature DataFrame (one row per candidate pair).
        candidates: Candidate pair DataFrame from blocking stage.
        label:      Human-readable label.

    Returns:
        ValidationResult
    """
    r = ValidationResult(stage=f"Features[{label}]")
    r.stats["rows"] = len(features)
    r.stats["cols"] = len(features.columns)

    # ── Required key columns ──────────────────────────────────────────
    required = [S1_ID_COL, CAND_ID_COL, SOURCE_TAG_COL]
    for col in required:
        if col not in features.columns:
            r.fail(f"Missing required column: '{col}'")

    # ── Row count matches candidates ──────────────────────────────────
    if len(features) != len(candidates):
        r.fail(
            f"Row count mismatch: features has {len(features):,} rows, "
            f"candidates has {len(candidates):,} rows"
        )

    # ── No null keys ──────────────────────────────────────────────────
    for col in [S1_ID_COL, CAND_ID_COL]:
        if col in features.columns:
            nulls = features[col].isna().sum()
            if nulls > 0:
                r.fail(f"{nulls:,} null values in key column '{col}'")

    # ── At least one numeric feature ──────────────────────────────────
    key_cols = set(required)
    numeric_cols = [
        c for c in features.columns
        if c not in key_cols and pd.api.types.is_numeric_dtype(features[c])
    ]
    if not numeric_cols:
        r.fail("No numeric feature columns found. Feature extraction may have failed.")
    r.stats["feature_columns"] = numeric_cols[:10]  # log first 10 names

    r.log_summary()
    return r


# ─────────────────────────────────────────────────────────────────────
# Stage 4: Validate Model Predictions
# ─────────────────────────────────────────────────────────────────────

def validate_predictions(
    predictions: pd.DataFrame,
    features: pd.DataFrame,
    label: str,
) -> ValidationResult:
    """Validate the prediction DataFrame from the ML model.

    Checks:
    - Required columns: s1_id, cand_id, match_score, is_match
    - Row count matches feature matrix
    - match_score in [0, 1] range
    - is_match is binary (0 or 1)

    Args:
        predictions: Model output DataFrame.
        features:    Feature matrix it was derived from.
        label:       Human-readable label.

    Returns:
        ValidationResult
    """
    r = ValidationResult(stage=f"Predictions[{label}]")
    r.stats["rows"] = len(predictions)

    # ── Required columns ──────────────────────────────────────────────
    required = [S1_ID_COL, CAND_ID_COL, MATCH_SCORE_COL, IS_MATCH_COL]
    for col in required:
        if col not in predictions.columns:
            r.fail(f"Missing required column: '{col}'")

    if not r.passed:
        r.log_summary()
        return r

    # ── Row count ─────────────────────────────────────────────────────
    if len(predictions) != len(features):
        r.fail(
            f"Row count mismatch: predictions={len(predictions):,}, "
            f"features={len(features):,}"
        )

    # ── match_score in [0, 1] ─────────────────────────────────────────
    score_col = predictions[MATCH_SCORE_COL]
    out_of_range = ((score_col < 0) | (score_col > 1)).sum()
    if out_of_range > 0:
        r.fail(f"{out_of_range:,} match_score values outside [0, 1] range")
    r.stats["match_score_min"] = float(score_col.min())
    r.stats["match_score_max"] = float(score_col.max())

    # ── is_match is binary ───────────────────────────────────────────
    is_match_vals = set(predictions[IS_MATCH_COL].unique())
    if not is_match_vals.issubset({0, 1, 0.0, 1.0}):
        r.fail(f"is_match contains non-binary values: {is_match_vals}")
    r.stats["predicted_matches"] = int((predictions[IS_MATCH_COL] == 1).sum())
    r.stats["predicted_non_matches"] = int((predictions[IS_MATCH_COL] == 0).sum())

    r.log_summary()
    return r


# ─────────────────────────────────────────────────────────────────────
# Stage 5: Validate Final Submission File
# ─────────────────────────────────────────────────────────────────────

def validate_submission(
    submission: pd.DataFrame,
    s1_ids: Set[str],
    label: str = "Test",
) -> ValidationResult:
    """Validate the final submission DataFrame before writing to disk.

    Checks (all hard requirements from the challenge):
    - Contains exactly two columns: entity_id, matched_entity_ids
    - Every Source 1 entity_id is represented exactly once
    - No extra entity_ids that weren't in Source 1
    - matched_entity_ids is a comma-separated string (or empty for singletons)
    - No modification to entity_id values

    Args:
        submission: Final submission DataFrame.
        s1_ids:     Set of all entity_ids from Source 1 (test set).
        label:      Human-readable label.

    Returns:
        ValidationResult
    """
    r = ValidationResult(stage=f"Submission[{label}]")
    r.stats["rows"] = len(submission)

    # ── Required columns ──────────────────────────────────────────────
    required = [SUBMISSION_ENTITY_COL, SUBMISSION_MATCHED_COL]
    for col in required:
        if col not in submission.columns:
            r.fail(f"Missing required column: '{col}'")

    if not r.passed:
        r.log_summary()
        return r

    # ── Every S1 entity is represented exactly once ───────────────────
    sub_ids = set(submission[SUBMISSION_ENTITY_COL].tolist())
    missing_ids = s1_ids - sub_ids
    extra_ids = sub_ids - s1_ids

    if missing_ids:
        r.fail(f"{len(missing_ids):,} Source 1 entity_ids missing from submission")
    if extra_ids:
        r.fail(f"{len(extra_ids):,} unexpected entity_ids in submission (not from Source 1)")

    dup_rows = submission.duplicated(subset=[SUBMISSION_ENTITY_COL]).sum()
    if dup_rows > 0:
        r.fail(f"{dup_rows:,} duplicate entity_id rows in submission")

    r.stats["total_entities"] = len(submission)
    r.stats["entities_with_matches"] = int(
        submission[SUBMISSION_MATCHED_COL].fillna("").str.strip().ne("").sum()
    )
    r.stats["singleton_entities"] = int(
        submission[SUBMISSION_MATCHED_COL].fillna("").str.strip().eq("").sum()
    )

    r.log_summary()
    return r


# ─────────────────────────────────────────────────────────────────────
# Utility: Raise if any result failed
# ─────────────────────────────────────────────────────────────────────

def assert_all_passed(results: List[ValidationResult]) -> None:
    """Raise PipelineValidationError if any validation result failed.

    Call this after collecting multiple validation results to halt the
    pipeline cleanly with a clear error message.

    Args:
        results: List of ValidationResult objects to check.

    Raises:
        PipelineValidationError if any result has passed=False.
    """
    failures = [r for r in results if not r.passed]
    if failures:
        messages = []
        for r in failures:
            for e in r.errors:
                messages.append(f"[{r.stage}] {e}")
        raise PipelineValidationError(
            f"Pipeline validation failed at {len(failures)} stage(s):\n" +
            "\n".join(f"  • {m}" for m in messages)
        )
