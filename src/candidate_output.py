"""
candidate_output.py — Standalone executable script for Stage 4 Blocking output.

Generates the required final deliverables:
1. output/candidate_pairs.tsv — TSV format candidate pairs for test set
2. reports/candidate_validation.txt — Detailed validation report

Usage:
    python3 src/candidate_output.py
"""

import os
import sys
import time
import heapq
import logging
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    CLEAN_TEST_DIR, CLEAN_TEST_FILES,
    SUBMISSION_DIR, CANDIDATE_PAIRS_FILE, REPORTS_DIR,
    ENTITY_ID_COL, NAME_COL, ADDRESS_COL,
    NAME_CLEAN_COL, ADDRESS_CLEAN_COL, SEP
)
from blocking import (
    extract_blocking_keys,
    build_inverted_index,
)

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

CANDIDATE_VALIDATION_REPORT = os.path.join(REPORTS_DIR, "candidate_validation.txt")


def generate_candidate_pairs_streaming(
    data_dir: str,
    source_files: Dict[str, str],
    output_path: str,
    max_candidates_per_source: int = 25,
) -> Tuple[int, int, int]:
    """Stream candidate pairs directly to output_path TSV without memory overhead.

    Args:
        data_dir: Directory containing cleaned source TSV files.
        source_files: Dict mapping source key to filename.
        output_path: Destination path for output/candidate_pairs.tsv.
        max_candidates_per_source: Max top candidates from S2 and S3 per S1 record.

    Returns:
        Tuple of (total_s1_records, s1_with_candidates, total_candidate_pairs)
    """
    log.info(f"Loading cleaned test dataset from {data_dir} ...")
    s1_path = os.path.join(data_dir, source_files["source1"])
    s2_path = os.path.join(data_dir, source_files["source2"])
    s3_path = os.path.join(data_dir, source_files["source3"])

    df1 = pd.read_csv(s1_path, sep=SEP, dtype=str, keep_default_na=False)
    df2 = pd.read_csv(s2_path, sep=SEP, dtype=str, keep_default_na=False)
    df3 = pd.read_csv(s3_path, sep=SEP, dtype=str, keep_default_na=False)
    log.info(f"Loaded: S1 ({len(df1):,} rows), S2 ({len(df2):,} rows), S3 ({len(df3):,} rows)")

    valid_s2_ids: Set[str] = set(df2[ENTITY_ID_COL])
    valid_s3_ids: Set[str] = set(df3[ENTITY_ID_COL])
    valid_cand_ids: Set[str] = valid_s2_ids | valid_s3_ids

    log.info("Building inverted index for Source 2 and Source 3 ...")
    index2, pruned2 = build_inverted_index(df2)
    index3, pruned3 = build_inverted_index(df3)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    s1_names = df1.get(NAME_CLEAN_COL, df1.get(NAME_COL, pd.Series([""] * len(df1))))
    s1_addrs = df1.get(ADDRESS_CLEAN_COL, df1.get(ADDRESS_COL, pd.Series([""] * len(df1))))

    log.info(f"Streaming candidate pairs directly to {output_path} ...")
    total_pairs = 0
    s1_with_cands = 0

    start_time = time.time()
    with open(output_path, "w", encoding="utf-8", buffering=2*1024*1024) as out_f:
        # Header required by challenge specification
        out_f.write("source1_entity_id\tcandidate_entity_ids\n")

        for idx, (s1_id, name, addr) in enumerate(zip(df1[ENTITY_ID_COL], s1_names, s1_addrs)):
            if idx > 0 and idx % 250_000 == 0:
                elapsed = time.time() - start_time
                log.info(f"Processed {idx:,} / {len(df1):,} S1 records ({idx/len(df1):.1%}) in {elapsed:.1f}s ...")

            s1_keys = extract_blocking_keys(name, addr)

            # Match S2
            s2_counts = defaultdict(int)
            for k in s1_keys:
                if k not in pruned2:
                    for cand_id in index2.get(k, []):
                        s2_counts[cand_id] += 1

            # Match S3
            s3_counts = defaultdict(int)
            for k in s1_keys:
                if k not in pruned3:
                    for cand_id in index3.get(k, []):
                        s3_counts[cand_id] += 1

            top_s2 = [cid for cid, _ in heapq.nlargest(max_candidates_per_source, s2_counts.items(), key=lambda x: x[1])] if s2_counts else []
            top_s3 = [cid for cid, _ in heapq.nlargest(max_candidates_per_source, s3_counts.items(), key=lambda x: x[1])] if s3_counts else []

            # Combine candidates, deduplicate while preserving rank order
            cands = []
            seen = set()
            for cid in top_s2 + top_s3:
                if cid not in seen and cid != s1_id and cid in valid_cand_ids:
                    seen.add(cid)
                    cands.append(cid)

            cand_str = ",".join(cands)
            out_f.write(f"{s1_id}\t{cand_str}\n")

            if cands:
                s1_with_cands += 1
                total_pairs += len(cands)

    elapsed_total = time.time() - start_time
    log.info(
        f"Completed streaming candidate_pairs.tsv → {output_path} in {elapsed_total:.1f}s "
        f"({len(df1):,} S1 records, {s1_with_cands:,} with candidates, {total_pairs:,} total candidate pairs)"
    )
    return len(df1), s1_with_cands, total_pairs


def validate_candidate_pairs_file(
    tsv_path: str,
    test_s1_df: pd.DataFrame,
    test_s2_df: pd.DataFrame,
    test_s3_df: pd.DataFrame,
    report_path: str,
) -> bool:
    """Run 10-point comprehensive validation on candidate_pairs.tsv and write report.

    Args:
        tsv_path: Path to output/candidate_pairs.tsv
        test_s1_df: DataFrame of clean test Source 1
        test_s2_df: DataFrame of clean test Source 2
        test_s3_df: DataFrame of clean test Source 3
        report_path: Path to write reports/candidate_validation.txt

    Returns:
        True if all critical validation assertions pass, False otherwise.
    """
    log.info(f"Validating {tsv_path} ...")
    assertions = []
    all_passed = True

    # 1. File existence and size check
    if not os.path.exists(tsv_path):
        assertions.append(("FILE_EXISTS", "FAIL", f"File does not exist: {tsv_path}"))
        all_passed = False
        return False
    
    file_size_bytes = os.path.getsize(tsv_path)
    file_size_mb = file_size_bytes / (1024 * 1024)
    if file_size_bytes == 0:
        assertions.append(("FILE_NOT_EMPTY", "FAIL", "File is empty (0 bytes)"))
        all_passed = False
    else:
        assertions.append(("FILE_NOT_EMPTY", "PASS", f"File exists ({file_size_mb:.2f} MB)"))

    # Stream read TSV line by line to prevent high memory consumption
    s1_expected_ids = list(test_s1_df[ENTITY_ID_COL])
    total_s1_expected = len(s1_expected_ids)
    s1_id_set = set(s1_expected_ids)
    s2_id_set = set(test_s2_df[ENTITY_ID_COL])
    s3_id_set = set(test_s3_df[ENTITY_ID_COL])
    valid_cand_id_set = s2_id_set | s3_id_set

    s1_actual_ids = []
    seen_s1_actual = set()
    s1_sequence_matches = True
    duplicate_s1_count = 0
    s1_in_cand_count = 0
    invalid_cand_count = 0
    total_cand_pairs = 0
    cand_counts_per_s1 = []
    s2_cand_count = 0
    s3_cand_count = 0
    s1_with_zero_cands = 0
    s1_with_cands = 0
    sample_rows = []

    with open(tsv_path, "r", encoding="utf-8") as f:
        header_line = f.readline()
        expected_header = "source1_entity_id\tcandidate_entity_ids\n"
        if header_line == expected_header:
            assertions.append(("HEADER_FORMAT", "PASS", "Header matches required: 'source1_entity_id\\tcandidate_entity_ids'"))
        else:
            assertions.append(("HEADER_FORMAT", "FAIL", f"Header mismatch: {repr(header_line)}"))
            all_passed = False

        for line_idx, line in enumerate(f):
            line_str = line.rstrip("\r\n")
            parts = line_str.split("\t")
            if len(parts) != 2:
                assertions.append(("TAB_FORMAT", "FAIL", f"Line {line_idx+2} does not have exactly 2 tab-separated columns"))
                all_passed = False
                s1_id = parts[0] if parts else ""
                cand_str = ""
            else:
                s1_id, cand_str = parts[0], parts[1]

            if line_idx < total_s1_expected and s1_id != s1_expected_ids[line_idx]:
                s1_sequence_matches = False

            if s1_id in seen_s1_actual:
                duplicate_s1_count += 1
            else:
                seen_s1_actual.add(s1_id)
            s1_actual_ids.append(s1_id)

            if len(sample_rows) < 5:
                sample_rows.append((s1_id, cand_str))

            if not cand_str:
                s1_with_zero_cands += 1
                cand_counts_per_s1.append(0)
                continue

            cands = [c.strip() for c in cand_str.split(",") if c.strip()]
            num_cands = len(cands)
            cand_counts_per_s1.append(num_cands)
            total_cand_pairs += num_cands
            s1_with_cands += 1

            for c in cands:
                if c in s1_id_set:
                    s1_in_cand_count += 1
                if c not in valid_cand_id_set:
                    invalid_cand_count += 1
                if c in s2_id_set:
                    s2_cand_count += 1
                elif c in s3_id_set:
                    s3_cand_count += 1

    total_s1_actual = len(s1_actual_ids)
    if total_s1_actual == total_s1_expected:
        assertions.append(("ROW_COUNT_MATCH", "PASS", f"Exactly {total_s1_actual:,} rows matching test Source 1 count"))
    else:
        assertions.append(("ROW_COUNT_MATCH", "FAIL", f"Row count mismatch: expected {total_s1_expected:,}, got {total_s1_actual:,}"))
        all_passed = False

    if s1_sequence_matches:
        assertions.append(("S1_ID_SEQUENCE", "PASS", "Source 1 entity IDs match original dataset in exact order"))
    else:
        assertions.append(("S1_ID_SEQUENCE", "FAIL", "Source 1 entity IDs do not match original sequence"))
        all_passed = False

    if duplicate_s1_count == 0:
        assertions.append(("S1_ID_UNIQUENESS", "PASS", f"All {total_s1_actual:,} Source 1 entity IDs are unique (zero duplicates)"))
    else:
        assertions.append(("S1_ID_UNIQUENESS", "FAIL", f"Found {duplicate_s1_count:,} duplicate Source 1 IDs!"))
        all_passed = False

    if s1_in_cand_count == 0:
        assertions.append(("NO_SELF_REFERENCING", "PASS", "Zero Source 1 entity IDs present in candidate_entity_ids column"))
    else:
        assertions.append(("NO_SELF_REFERENCING", "FAIL", f"Found {s1_in_cand_count:,} Source 1 IDs in candidate_entity_ids"))
        all_passed = False

    if invalid_cand_count == 0:
        assertions.append(("CANDIDATE_ID_VALIDITY", "PASS", f"All {total_cand_pairs:,} candidate IDs strictly belong to Source 2 or Source 3"))
    else:
        assertions.append(("CANDIDATE_ID_VALIDITY", "FAIL", f"Found {invalid_cand_count:,} candidate IDs not in Source 2 or 3"))
        all_passed = False

    mean_cands = (total_cand_pairs / total_s1_actual) if total_s1_actual > 0 else 0.0
    min_cands = min(cand_counts_per_s1) if cand_counts_per_s1 else 0
    max_cands = max(cand_counts_per_s1) if cand_counts_per_s1 else 0

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("AMAZON ML CHALLENGE — STAGE 4 BLOCKING DELIVERABLE VALIDATION REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write("1. FILE INFORMATION\n")
        f.write(f"   Target File Path:  {tsv_path}\n")
        f.write(f"   File Size:         {file_size_mb:.2f} MB ({file_size_bytes:,} bytes)\n")
        f.write(f"   Total Header Row:  1 row ('source1_entity_id\\tcandidate_entity_ids')\n")
        f.write(f"   Total Data Rows:   {total_s1_actual:,} rows\n\n")

        f.write("2. CANDIDATE PAIR COVERAGE & STATISTICS\n")
        f.write(f"   Total Source 1 Records:          {total_s1_actual:,}\n")
        f.write(f"   S1 Records with Candidates (>0): {s1_with_cands:,} ({s1_with_cands/total_s1_actual:.2%})\n")
        f.write(f"   S1 Records with Zero Candidates: {s1_with_zero_cands:,} ({s1_with_zero_cands/total_s1_actual:.2%})\n")
        f.write(f"   Total Candidate Pairs Generated: {total_cand_pairs:,}\n")
        f.write(f"   Average Candidates per S1:       {mean_cands:.2f}\n")
        f.write(f"   Min Candidates per S1:           {min_cands}\n")
        f.write(f"   Max Candidates per S1:           {max_cands}\n")
        f.write(f"   Source 2 Candidates:             {s2_cand_count:,} ({s2_cand_count/total_cand_pairs:.2%})\n" if total_cand_pairs > 0 else "   Source 2 Candidates:             0\n")
        f.write(f"   Source 3 Candidates:             {s3_cand_count:,} ({s3_cand_count/total_cand_pairs:.2%})\n\n" if total_cand_pairs > 0 else "   Source 3 Candidates:             0\n\n")

        f.write("3. VALIDATION ASSERTIONS SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'ASSERTION NAME':<25} | {'STATUS':<6} | {'DETAILS':<45}\n")
        f.write("-" * 80 + "\n")
        for name, status, details in assertions:
            f.write(f"{name:<25} | {status:<6} | {details:<45}\n")
        f.write("-" * 80 + "\n\n")

        f.write("4. SAMPLE ENTRIES VERIFICATION (First 5 Rows)\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'source1_entity_id':<20} | {'candidate_entity_ids (truncated if long)'}\n")
        f.write("-" * 80 + "\n")
        for s1, cands in sample_rows:
            cands_disp = cands[:60] + "..." if len(cands) > 60 else (cands if cands else "<EMPTY>")
            f.write(f"{s1:<20} | {cands_disp}\n")
        f.write("-" * 80 + "\n\n")

        f.write("OVERALL VALIDATION STATUS: " + ("PASSED SUCCESSFUL" if all_passed else "FAILED") + "\n")

    log.info(f"Validation report saved → {report_path}")
    return all_passed


def main():
    log.info("Starting optimized streaming candidate output generation process ...")

    s1_path = os.path.join(CLEAN_TEST_DIR, CLEAN_TEST_FILES["source1"])
    s2_path = os.path.join(CLEAN_TEST_DIR, CLEAN_TEST_FILES["source2"])
    s3_path = os.path.join(CLEAN_TEST_DIR, CLEAN_TEST_FILES["source3"])

    if not os.path.exists(s1_path) or not os.path.exists(s2_path) or not os.path.exists(s3_path):
        log.error(f"Clean test files missing in {CLEAN_TEST_DIR}. Please run preprocessing first.")
        sys.exit(1)

    # Stream write output/candidate_pairs.tsv directly
    generate_candidate_pairs_streaming(
        data_dir=CLEAN_TEST_DIR,
        source_files=CLEAN_TEST_FILES,
        output_path=CANDIDATE_PAIRS_FILE,
        max_candidates_per_source=25,
    )

    log.info("Loading test dataframes for post-generation validation ...")
    test_s1_df = pd.read_csv(s1_path, sep=SEP, dtype=str, keep_default_na=False)
    test_s2_df = pd.read_csv(s2_path, sep=SEP, dtype=str, keep_default_na=False)
    test_s3_df = pd.read_csv(s3_path, sep=SEP, dtype=str, keep_default_na=False)

    log.info("Running validation on generated candidate_pairs.tsv ...")
    passed = validate_candidate_pairs_file(
        tsv_path=CANDIDATE_PAIRS_FILE,
        test_s1_df=test_s1_df,
        test_s2_df=test_s2_df,
        test_s3_df=test_s3_df,
        report_path=CANDIDATE_VALIDATION_REPORT,
    )

    if passed:
        log.info("SUCCESS! All validation checks passed. Deliverables generated.")
    else:
        log.error("Validation failed! Please check reports/candidate_validation.txt")
        sys.exit(1)


if __name__ == "__main__":
    main()
