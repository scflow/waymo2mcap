#!/usr/bin/env python3
"""
Auto Waymo TFRecord to MCAP Batch Converter.
Automatically scans the tfrecord directory, validates any existing MCAP files,
skips already valid ones, and converts missing/corrupted ones.
Can be executed without any arguments by default.
"""

import os
import sys
import glob
import time
import argparse
from pathlib import Path

# Ensure workspace and src are in path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / "src"))

from mcap.reader import make_reader
import waymo_to_mcap


def validate_mcap(mcap_path: str, min_messages: int = 100) -> tuple[bool, str]:
    """Validate an MCAP file for structural integrity and essential topics.
    
    Returns:
        (is_valid, reason_or_summary)
    """
    if not os.path.exists(mcap_path):
        return False, "File does not exist"

    size_mb = os.path.getsize(mcap_path) / (1024 * 1024)
    if size_mb < 1.0:
        return False, f"File size too small ({size_mb:.2f} MB)"

    try:
        with open(mcap_path, "rb") as f:
            reader = make_reader(f)
            summary = reader.get_summary()
            if summary is None:
                return False, "Corrupted or incomplete file (no MCAP summary footer)"

            stats = summary.statistics
            msg_count = stats.message_count
            channel_count = stats.channel_count

            if msg_count < min_messages:
                return False, f"Insufficient messages ({msg_count} < {min_messages})"

            channels = {c.topic for c in summary.channels.values()}
            essential_topics = ["/tf", "/camera/FRONT/compressed", "/lidar/points"]
            missing_topics = [t for t in essential_topics if t not in channels]
            if missing_topics:
                return False, f"Missing essential topics: {missing_topics}"

            return True, f"{msg_count:,} msgs, {channel_count} channels, {size_mb:.1f} MB"
    except Exception as e:
        return False, f"Reader error: {e}"


def find_default_dirs() -> tuple[Path, Path]:
    """Auto-detect input tfrecord and output mcap directories."""
    # Check candidates for input tfrecord dir
    candidates_input = [
        SCRIPT_DIR / "data" / "tfrecord",
        SCRIPT_DIR / "data",
        Path("data/tfrecord"),
        Path("data"),
        Path(".")
    ]
    input_dir = None
    for cand in candidates_input:
        if cand.exists() and (list(cand.glob("*.tfrecord")) or list(cand.glob("*.tfrecord*"))):
            input_dir = cand
            break

    if input_dir is None:
        input_dir = SCRIPT_DIR / "data" / "tfrecord"

    # Default output dir
    if input_dir.name == "tfrecord":
        output_dir = input_dir.parent / "mcap"
    else:
        output_dir = input_dir / "mcap"

    return input_dir.resolve(), output_dir.resolve()


def auto_convert(
    input_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    force: bool = False,
    dry_run: bool = False
):
    def_in, def_out = find_default_dirs()
    in_dir = Path(input_dir).resolve() if input_dir else def_in
    out_dir = Path(output_dir).resolve() if output_dir else def_out

    print("=" * 65)
    print(" Waymo TFRecord -> MCAP Auto Batch Converter")
    print(f"   Input Dir:  {in_dir}")
    print(f"   Output Dir: {out_dir}")
    print("=" * 65)

    if not in_dir.exists():
        print(f"Error: Input directory does not exist: {in_dir}", file=sys.stderr)
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # Find all tfrecord files
    tfrecord_patterns = ["*.tfrecord", "*.tfrecord*"]
    found_files = []
    for pat in tfrecord_patterns:
        for f in in_dir.glob(pat):
            if f.is_file() and not f.name.startswith("."):
                if f not in found_files:
                    found_files.append(f)

    found_files.sort()
    total = len(found_files)

    if total == 0:
        print(f"No TFRecord files found in {in_dir}")
        return

    print(f"Found {total} TFRecord file(s). Starting check...\n")

    stats = {
        "skipped": 0,
        "converted": 0,
        "failed": 0,
        "re_converted": 0
    }
    start_total_time = time.time()

    for idx, tf_file in enumerate(found_files, 1):
        target_stem = tf_file.stem
        # Handle cases like .tfrecord-00000-of-00266
        if ".tfrecord" in target_stem:
            target_stem = target_stem.split(".tfrecord")[0]
        mcap_file = out_dir / f"{target_stem}.mcap"

        print(f"[{idx}/{total}] Checking: {tf_file.name}")

        is_valid = False
        status_reason = "File does not exist"

        if mcap_file.exists() and not force:
            is_valid, status_reason = validate_mcap(str(mcap_file))

        if is_valid:
            print(f"  --> [SKIP] Already converted & verified: {status_reason}")
            stats["skipped"] += 1
            print("-" * 60)
            continue

        if mcap_file.exists() and not is_valid:
            print(f"  --> [INVALID] Existing MCAP failed verification ({status_reason}). Re-converting...")
            stats["re_converted"] += 1
        elif force and mcap_file.exists():
            print(f"  --> [FORCE] Re-converting existing file...")
            stats["re_converted"] += 1
        else:
            print(f"  --> [NEW] Target MCAP does not exist. Converting...")

        if dry_run:
            print("  --> [DRY-RUN] Skipping actual conversion.")
            print("-" * 60)
            continue

        try:
            t0 = time.time()
            waymo_to_mcap.convert_tfrecord_to_mcap(
                input_path=str(tf_file),
                output_path=str(mcap_file),
                merge_lidar=True,
                split_lidar=False,
                include_second_return=True,
                include_raw_frame=False
            )
            elapsed = time.time() - t0

            # Verify immediately after conversion
            ok, ver_msg = validate_mcap(str(mcap_file))
            if ok:
                print(f"  --> [SUCCESS] Converted and verified in {elapsed:.2f}s ({ver_msg})")
                stats["converted"] += 1
            else:
                print(f"  --> [WARNING] Converted but verification failed: {ver_msg}", file=sys.stderr)
                stats["failed"] += 1
        except Exception as e:
            print(f"  --> [ERROR] Conversion failed: {e}", file=sys.stderr)
            stats["failed"] += 1

        print("-" * 60)

    total_time = time.time() - start_total_time
    print("\n" + "=" * 65)
    print(" Batch Conversion Summary:")
    print(f"   Total scanned:     {total}")
    print(f"   Skipped (valid):   {stats['skipped']}")
    print(f"   Converted:         {stats['converted']}")
    if stats["re_converted"]:
        print(f"   Re-converted:      {stats['re_converted']}")
    if stats["failed"]:
        print(f"   Failed:            {stats['failed']}")
    print(f"   Total time:        {total_time:.2f}s")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(
        description="Auto Waymo TFRecord to MCAP Converter (default: scans data/tfrecord/ and writes data/mcap/)"
    )
    parser.add_argument(
        "--input-dir", "-i",
        type=str,
        default=None,
        help="Directory containing .tfrecord files (default: auto-detected data/tfrecord)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Directory to save .mcap files (default: auto-detected data/mcap)"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force re-conversion even if MCAP already exists and is valid"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and validate existing files without performing conversions"
    )

    args = parser.parse_args()
    auto_convert(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        force=args.force,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
