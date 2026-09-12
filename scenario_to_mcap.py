#!/usr/bin/env python3
"""
Waymo Open Motion Dataset (WOMD) Scenario to MCAP Converter.
Converts scenario_pb2.Scenario into Foxglove-compatible MCAP files.
Supports multi-agent tracks, SDC ego pose & velocity, ground truth trajectory forecast,
high-definition map layers, and dynamic traffic signals.
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / "src"))

import tfrecord
from waymo_open_dataset.protos import scenario_pb2
from converter.scenario_util import convert_scenario_to_mcap


def main():
    parser = argparse.ArgumentParser(
        description="Convert Waymo Open Motion Dataset (WOMD) Scenario TFRecord to MCAP"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="data/uncompressed_scenario_training_training.tfrecord-00000-of-01000",
        help="Path to scenario TFRecord file"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="data/mcap/scenarios",
        help="Directory to save output MCAP files (default: data/mcap/scenarios)"
    )
    parser.add_argument(
        "--max-scenarios", "-n",
        type=int,
        default=1,
        help="Number of scenarios to convert (default: 1)"
    )
    parser.add_argument(
        "--scenario-id",
        type=str,
        default=None,
        help="Specific scenario ID to convert"
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Starting index of scenario in the TFRecord file (default: 0)"
    )

    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        print(f"Error: Input file does not exist: {input_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print(" Waymo Open Motion Dataset (WOMD) Scenario -> MCAP Converter")
    print(f"   Input:      {input_path}")
    print(f"   Output Dir: {out_dir}")
    print("=" * 65)

    reader = tfrecord.tfrecord_iterator(str(input_path))
    converted = 0

    for idx, record in enumerate(reader):
        if idx < args.index:
            continue

        scenario = scenario_pb2.Scenario()
        scenario.ParseFromString(record)

        if args.scenario_id and scenario.scenario_id != args.scenario_id:
            continue

        print(f"\n[{converted + 1}/{args.max_scenarios}] Processing scenario index {idx}:")
        out_mcap = out_dir / f"scenario_{scenario.scenario_id}.mcap"
        convert_scenario_to_mcap(scenario, str(out_mcap))
        converted += 1

        if converted >= args.max_scenarios:
            break

    print("\n" + "=" * 65)
    print(f" Conversion Finished! Converted {converted} scenario(s).")
    print("=" * 65)


if __name__ == "__main__":
    main()
