"""
pipeline.py — Central orchestration for the Amazon ML Challenge pipeline.

Author: Navaneeswar (Pipeline Architecture & Integration Lead)

This module owns the EXECUTION FLOW of the pipeline. It:
  1. Loads the already-cleaned datasets (Stage 3 output, by Saranya).
  2. Calls the blocking module    (Stage 4, owned by Karthik).
  3. Calls feature extraction     (Stage 5, owned by the features team).
  4. Calls ML model prediction    (Stage 6, owned by the model team).
  5. Assembles and validates the submission file.
  6. Writes the final matching_results.tsv.

Design principles
─────────────────
- This file is the ONLY file that imports across all team modules.
- Other modules expose clean function interfaces; this file calls them.
- Each stage is validated before proceeding to the next.
- No business logic lives here — only orchestration.
- Reproducible: all seeds, paths, and parameters come from config.py.
- Fail-fast: a failed validation raises PipelineValidationError.

Usage
─────
    python -m src.pipeline --mode test    # predict on test set only
    python -m src.pipeline --mode train   # full train + eval cycle
    python -m src.pipeline --mode both    # train + test (default)
    python -m src.pipeline --skip-blocking  # reuse existing candidates
"""

import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime
from typing import Optional, Dict, Any

import pandas as pd

# Ensure src/ is on the path when running as a script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    # Paths — data
    CLEAN_TRAIN_DIR, CLEAN_TEST_DIR,
    CLEAN_SOURCE_FILES, CLEAN_TEST_FILES, CLEAN_GT_FILE,
    # Paths — pipeline outputs
    CANDIDATES_TRAIN_FILE, CANDIDATES_TEST_FILE,
    FEATURES_TRAIN_FILE, FEATURES_TEST_FILE,
    SUBMISSION_DIR, SUBMISSION_FILE,
    REPORTS_DIR, PIPELINE_REPORT,
    # Column names
    ENTITY_ID_COL,
    S1_ID_COL, CAND_ID_COL, SOURCE_TAG_COL,
    MATCH_SCORE_COL, IS_MATCH_COL,
    SUBMISSION_ENTITY_COL, SUBMISSION_MATCHED_COL,
    # I/O
    SEP,
)
from validation import (
    ValidationResult,
    PipelineValidationError,
    validate_clean_data,
    validate_candidates,
    validate_features,
    validate_predictions,
    validate_submission,
    assert_all_passed,
)

# ─── Logger ──────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

# ─── Pipeline-level constants ─────────────────────────────────────────
# Matching score threshold: pairs above this are classified as matches.
# Adjust this once Karthik's model training gives us the optimal threshold.
DEFAULT_THRESHOLD = 0.5


# ═════════════════════════════════════════════════════════════════════
# STAGE 0 — Load Clean Data
# ═════════════════════════════════════════════════════════════════════

def load_clean_data(data_dir: str, source_files: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    """Load the three cleaned source DataFrames from disk.

    This stage consumes the output of Saranya's normalization pipeline
    (Stage 3). The files are already cleaned — no re-normalization happens here.

    Args:
        data_dir:     Directory containing the clean TSV files.
        source_files: Dict mapping source key ("source1", ...) → filename.

    Returns:
        Dict with keys "source1", "source2", "source3" mapped to DataFrames.

    Raises:
        FileNotFoundError if any clean file is missing.
    """
    log.info(f"Loading clean data from: {data_dir}")
    data = {}
    for key, fname in source_files.items():
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Clean file not found: {path}\n"
                f"Run `python -m src.preprocessing` first to generate clean files."
            )
        log.info(f"  Reading {fname} ...")
        data[key] = pd.read_csv(path, sep=SEP, dtype=str, keep_default_na=False)
        log.info(f"  Loaded  {len(data[key]):,} rows")

    return data


def load_ground_truth(data_dir: str) -> Optional[pd.DataFrame]:
    """Load the ground truth file (train only; test has no GT).

    Args:
        data_dir: Clean train directory.

    Returns:
        DataFrame with columns [entity_id, matched_entity_ids], or None.
    """
    gt_path = os.path.join(data_dir, CLEAN_GT_FILE)
    if not os.path.exists(gt_path):
        log.warning(f"Ground truth not found at {gt_path}. Skipping.")
        return None
    gt = pd.read_csv(gt_path, sep=SEP, dtype=str, keep_default_na=False)
    log.info(f"  Loaded ground truth: {len(gt):,} rows")
    return gt


# ═════════════════════════════════════════════════════════════════════
# STAGE 1 — Candidate Generation (Karthik's Blocking Module)
# ═════════════════════════════════════════════════════════════════════

def run_candidate_generation(
    data: Dict[str, pd.DataFrame],
    mode: str,
    skip_if_exists: bool = False,
) -> pd.DataFrame:
    """Generate candidate entity pairs via the blocking module.

    INTERFACE ONLY — Implementation owned by Karthik.

    This function delegates entirely to Karthik's blocking module.
    Navaneeswar's responsibility:
      ✅ Call the blocking module with the correct data.
      ✅ Validate its output before returning.
      ❌ Do NOT implement blocking logic here.

    Expected output schema from blocking module:
        s1_id       (str)  — Source 1 entity ID
        cand_id     (str)  — Candidate entity ID from Source 2 or 3
        source      (str)  — "source2" or "source3"
        shared_keys (int)  — Number of blocking keys matched (blocking score)

    Args:
        data:            Dict of cleaned DataFrames (source1, source2, source3).
        mode:            "train" or "test" — determines output file path.
        skip_if_exists:  If True and candidate file already exists, reload it.

    Returns:
        DataFrame of candidate pairs.
    """
    out_file = CANDIDATES_TRAIN_FILE if mode == "train" else CANDIDATES_TEST_FILE

    if skip_if_exists and os.path.exists(out_file):
        log.info(f"  [Blocking] Reloading existing candidates from {out_file}")
        return pd.read_parquet(out_file)

    log.info(f"  [Blocking] Delegating to Karthik's blocking module ...")

    # ── Interface point: import Karthik's blocking module ─────────────
    try:
        from blocking import generate_candidates_for_dataset
    except ImportError:
        raise ImportError(
            "Could not import Karthik's blocking module (blocking.py).\n"
            "Ensure blocking.py is present in src/ and exposes "
            "generate_candidates_for_dataset()."
        )

    source_files = CLEAN_SOURCE_FILES if mode == "train" else CLEAN_TEST_FILES
    data_dir     = CLEAN_TRAIN_DIR    if mode == "train" else CLEAN_TEST_DIR

    candidates = generate_candidates_for_dataset(
        data_dir=data_dir,
        source_files=source_files,
    )

    # ── Persist ──────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    candidates.to_parquet(out_file, index=False)
    log.info(f"  [Blocking] Saved {len(candidates):,} candidate pairs → {out_file}")

    return candidates


# ═════════════════════════════════════════════════════════════════════
# STAGE 2 — Feature Extraction
# ═════════════════════════════════════════════════════════════════════

def run_feature_extraction(
    candidates: pd.DataFrame,
    data: Dict[str, pd.DataFrame],
    mode: str,
) -> pd.DataFrame:
    """Extract similarity features for each candidate pair.

    INTERFACE ONLY — Implementation owned by the Features team.

    Expected output schema:
        s1_id            (str)   — Source 1 entity ID
        cand_id          (str)   — Candidate entity ID
        source           (str)   — "source2" or "source3"
        name_jaro        (float) — Jaro-Winkler similarity of clean names
        name_token_sort  (float) — Token-sort ratio
        addr_token       (float) — Address token overlap
        country_match    (int)   — 1 if countries match, 0 otherwise
        ... (any additional features the feature team adds)

    Args:
        candidates: DataFrame of candidate pairs from blocking stage.
        data:       Dict of cleaned source DataFrames (for field lookup).
        mode:       "train" or "test".

    Returns:
        Feature DataFrame (one row per candidate pair).
    """
    out_file = FEATURES_TRAIN_FILE if mode == "train" else FEATURES_TEST_FILE

    log.info(f"  [Features] Delegating to features module ...")

    # ── Interface point: import the features module ───────────────────
    try:
        from features import extract_features
    except ImportError:
        raise ImportError(
            "Could not import the features module (features.py).\n"
            "Ensure features.py is present in src/ and exposes "
            "extract_features(candidates, data) -> pd.DataFrame."
        )

    features = extract_features(candidates=candidates, data=data)

    # ── Persist ──────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    features.to_parquet(out_file, index=False)
    log.info(f"  [Features] Saved {len(features):,} feature rows → {out_file}")

    return features


# ═════════════════════════════════════════════════════════════════════
# STAGE 3 — ML Model Prediction
# ═════════════════════════════════════════════════════════════════════

def run_model_prediction(
    features: pd.DataFrame,
    mode: str,
    threshold: float = DEFAULT_THRESHOLD,
) -> pd.DataFrame:
    """Run the trained ML model on the feature matrix.

    INTERFACE ONLY — Implementation owned by the ML Model team.

    Expected output schema:
        s1_id        (str)   — Source 1 entity ID
        cand_id      (str)   — Candidate entity ID
        source       (str)   — "source2" or "source3"
        match_score  (float) — Predicted match probability in [0, 1]
        is_match     (int)   — Binary prediction: 1 if match_score >= threshold

    Args:
        features:  Feature DataFrame from feature extraction stage.
        mode:      "train" or "test".
        threshold: Score cutoff for binary classification.

    Returns:
        Prediction DataFrame.
    """
    log.info(f"  [Model] Delegating to ML model module (threshold={threshold}) ...")

    # ── Interface point: import the ML model module ───────────────────
    try:
        from model import predict
    except ImportError:
        raise ImportError(
            "Could not import the ML model module (model.py).\n"
            "Ensure model.py is present in src/ and exposes "
            "predict(features) -> pd.DataFrame."
        )

    predictions = predict(features=features)

    # Apply threshold to generate binary prediction if model didn't
    if IS_MATCH_COL not in predictions.columns:
        predictions[IS_MATCH_COL] = (
            predictions[MATCH_SCORE_COL] >= threshold
        ).astype(int)

    return predictions


# ═════════════════════════════════════════════════════════════════════
# STAGE 4 — Assemble Submission File
# ═════════════════════════════════════════════════════════════════════

def generate_submission(
    predictions: pd.DataFrame,
    s1_df: pd.DataFrame,
) -> pd.DataFrame:
    """Assemble the final matching_results.tsv from model predictions.

    Rules enforced here:
    - Every Source 1 entity_id must appear in the output (even singletons).
    - matched_entity_ids is a comma-separated list of matched IDs.
    - Singletons (no matches) get an empty matched_entity_ids field.
    - entity_id values are NEVER modified.

    Args:
        predictions: Model prediction DataFrame (s1_id, cand_id, is_match).
        s1_df:       The full Source 1 DataFrame (ensures all IDs are covered).

    Returns:
        Submission DataFrame with columns [entity_id, matched_entity_ids].
    """
    log.info("  [Submission] Assembling final submission file ...")

    # Filter to predicted matches only
    matches = predictions[predictions[IS_MATCH_COL] == 1][[S1_ID_COL, CAND_ID_COL]]

    # Aggregate: group by s1_id, collect all matched candidate IDs
    match_groups = (
        matches
        .groupby(S1_ID_COL)[CAND_ID_COL]
        .apply(lambda ids: ",".join(sorted(ids)))
        .reset_index()
        .rename(columns={S1_ID_COL: SUBMISSION_ENTITY_COL,
                          CAND_ID_COL: SUBMISSION_MATCHED_COL})
    )

    # Build a complete submission — all Source 1 IDs must be present
    all_s1_ids = s1_df[[ENTITY_ID_COL]].rename(
        columns={ENTITY_ID_COL: SUBMISSION_ENTITY_COL}
    )
    submission = all_s1_ids.merge(match_groups, on=SUBMISSION_ENTITY_COL, how="left")
    submission[SUBMISSION_MATCHED_COL] = (
        submission[SUBMISSION_MATCHED_COL].fillna("")
    )

    log.info(
        f"  [Submission] {len(submission):,} entities | "
        f"{submission[SUBMISSION_MATCHED_COL].ne('').sum():,} with matches | "
        f"{submission[SUBMISSION_MATCHED_COL].eq('').sum():,} singletons"
    )

    return submission


def save_submission(submission: pd.DataFrame, path: str) -> None:
    """Write the submission DataFrame to a TSV file.

    Args:
        submission: Final submission DataFrame.
        path:       Output file path (matching_results.tsv).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    submission.to_csv(path, sep=SEP, index=False)
    log.info(f"  [Submission] Saved → {path}")


# ═════════════════════════════════════════════════════════════════════
# PIPELINE ORCHESTRATOR
# ═════════════════════════════════════════════════════════════════════

def run_pipeline(
    mode: str = "both",
    skip_blocking: bool = False,
    threshold: float = DEFAULT_THRESHOLD,
) -> None:
    """End-to-end pipeline orchestrator.

    Execution flow:
        load_data()
            ↓  validate_clean_data()
        run_candidate_generation()
            ↓  validate_candidates()
        run_feature_extraction()
            ↓  validate_features()
        run_model_prediction()
            ↓  validate_predictions()
        generate_submission()
            ↓  validate_submission()
        save_submission()

    Args:
        mode:          "train" | "test" | "both"
        skip_blocking: Reuse saved candidates if they exist.
        threshold:     Match score classification threshold.
    """
    pipeline_start = time.time()
    run_modes = []
    if mode in ("train", "both"):
        run_modes.append("train")
    if mode in ("test", "both"):
        run_modes.append("test")

    log.info("=" * 65)
    log.info("  Amazon ML Challenge — Business Entity Resolution Pipeline")
    log.info(f"  Author     : Navaneeswar (Pipeline Lead)")
    log.info(f"  Mode       : {mode}")
    log.info(f"  Threshold  : {threshold}")
    log.info(f"  Started    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 65)

    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    pipeline_report: Dict[str, Any] = {"mode": mode, "stages": {}}

    for current_mode in run_modes:
        log.info(f"\n{'─' * 65}")
        log.info(f"  MODE: {current_mode.upper()}")
        log.info(f"{'─' * 65}")

        source_files = CLEAN_SOURCE_FILES if current_mode == "train" else CLEAN_TEST_FILES
        data_dir     = CLEAN_TRAIN_DIR    if current_mode == "train" else CLEAN_TEST_DIR

        # ── Stage 0: Load clean data ──────────────────────────────────
        log.info("\n[Stage 0] Loading clean data ...")
        t0 = time.time()
        data = load_clean_data(data_dir=data_dir, source_files=source_files)

        # Load ground truth for train mode
        ground_truth = None
        if current_mode == "train":
            ground_truth = load_ground_truth(CLEAN_TRAIN_DIR)

        # Validate each source
        validation_results = []
        for key, df in data.items():
            vr = validate_clean_data(df, label=f"{key} [{current_mode}]")
            validation_results.append(vr)
        assert_all_passed(validation_results)

        s1_ids = set(data["source1"][ENTITY_ID_COL].tolist())
        pipeline_report["stages"][f"{current_mode}_load"] = {
            "s1_rows": len(data["source1"]),
            "s2_rows": len(data["source2"]),
            "s3_rows": len(data["source3"]),
            "elapsed_sec": round(time.time() - t0, 1),
        }

        # ── Stage 1: Candidate Generation ────────────────────────────
        log.info(f"\n[Stage 1] Candidate Generation ({current_mode}) ...")
        t1 = time.time()
        candidates = run_candidate_generation(
            data=data,
            mode=current_mode,
            skip_if_exists=skip_blocking,
        )
        vr_cands = validate_candidates(candidates, s1_ids, label=current_mode)
        assert_all_passed([vr_cands])
        pipeline_report["stages"][f"{current_mode}_blocking"] = {
            "candidate_pairs": len(candidates),
            "s1_coverage_pct": vr_cands.stats.get("s1_coverage_pct"),
            "elapsed_sec": round(time.time() - t1, 1),
        }

        # ── Stage 2: Feature Extraction ───────────────────────────────
        log.info(f"\n[Stage 2] Feature Extraction ({current_mode}) ...")
        t2 = time.time()
        features = run_feature_extraction(
            candidates=candidates,
            data=data,
            mode=current_mode,
        )
        vr_feats = validate_features(features, candidates, label=current_mode)
        assert_all_passed([vr_feats])
        pipeline_report["stages"][f"{current_mode}_features"] = {
            "feature_rows": len(features),
            "feature_cols": vr_feats.stats.get("feature_columns"),
            "elapsed_sec": round(time.time() - t2, 1),
        }

        # ── Stage 3: Model Prediction ─────────────────────────────────
        log.info(f"\n[Stage 3] Model Prediction ({current_mode}) ...")
        t3 = time.time()
        predictions = run_model_prediction(
            features=features,
            mode=current_mode,
            threshold=threshold,
        )
        vr_preds = validate_predictions(predictions, features, label=current_mode)
        assert_all_passed([vr_preds])
        pipeline_report["stages"][f"{current_mode}_prediction"] = {
            "predicted_matches": vr_preds.stats.get("predicted_matches"),
            "elapsed_sec": round(time.time() - t3, 1),
        }

        # ── Stage 4: Generate & Validate Submission ───────────────────
        if current_mode == "test":
            log.info(f"\n[Stage 4] Generating Submission ...")
            t4 = time.time()
            submission = generate_submission(
                predictions=predictions,
                s1_df=data["source1"],
            )
            vr_sub = validate_submission(submission, s1_ids, label="Test")
            assert_all_passed([vr_sub])

            save_submission(submission, SUBMISSION_FILE)
            pipeline_report["stages"]["submission"] = {
                "file": SUBMISSION_FILE,
                "total_entities": vr_sub.stats.get("total_entities"),
                "entities_with_matches": vr_sub.stats.get("entities_with_matches"),
                "singleton_entities": vr_sub.stats.get("singleton_entities"),
                "elapsed_sec": round(time.time() - t4, 1),
            }

    # ── Write pipeline report ─────────────────────────────────────────
    pipeline_report["total_elapsed_sec"] = round(time.time() - pipeline_start, 1)
    pipeline_report["completed_at"] = datetime.now().isoformat()
    with open(PIPELINE_REPORT, "w") as f:
        json.dump(pipeline_report, f, indent=2)
    log.info(f"\nPipeline report saved → {PIPELINE_REPORT}")

    log.info("\n" + "=" * 65)
    log.info(f"  Pipeline COMPLETE in {pipeline_report['total_elapsed_sec']}s")
    log.info("=" * 65)


# ═════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Amazon ML Challenge — Business Entity Resolution Pipeline\n"
                    "Author: Navaneeswar (Pipeline Architecture & Integration Lead)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.pipeline                        # run full train + test pipeline
  python -m src.pipeline --mode test            # run test prediction only
  python -m src.pipeline --mode train           # run training eval only
  python -m src.pipeline --skip-blocking        # reuse cached candidate pairs
  python -m src.pipeline --threshold 0.6        # adjust match threshold
        """
    )
    parser.add_argument(
        "--mode",
        choices=["train", "test", "both"],
        default="both",
        help="Pipeline mode: train / test / both (default: both)",
    )
    parser.add_argument(
        "--skip-blocking",
        action="store_true",
        help="Skip candidate generation and reuse saved candidates if they exist.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Match score classification threshold (default: {DEFAULT_THRESHOLD})",
    )

    args = parser.parse_args()

    try:
        run_pipeline(
            mode=args.mode,
            skip_blocking=args.skip_blocking,
            threshold=args.threshold,
        )
    except PipelineValidationError as e:
        log.error(f"\n❌ Pipeline halted due to validation failure:\n{e}")
        sys.exit(1)
    except FileNotFoundError as e:
        log.error(f"\n❌ Pipeline halted — file not found:\n{e}")
        sys.exit(1)
    except ImportError as e:
        log.error(f"\n❌ Pipeline halted — missing team module:\n{e}")
        sys.exit(1)
