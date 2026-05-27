"""
CLI wrapper for the Claude data-preprocessing agent.

Usage
-----
    python scripts/preprocess_data.py data/raw/my_data.csv
    python scripts/preprocess_data.py data/raw/my_data.csv --out data/processed/clean.csv
    python scripts/preprocess_data.py data/raw/my_data.csv --model claude-sonnet-4-6 --verbose
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data_preprocessing_agent import preprocess_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess an arbitrary CSV to the canonical DR pipeline schema using Claude."
    )
    parser.add_argument("input", help="Path to the raw CSV file")
    parser.add_argument(
        "--out", "-o",
        default=None,
        help="Output path for the cleaned CSV (default: <input>_clean.csv)",
    )
    parser.add_argument(
        "--model", "-m",
        default="claude-sonnet-4-6",
        help="Claude model to use (default: claude-sonnet-4-6)",
    )
    parser.add_argument(
        "--report", "-r",
        default=None,
        help="Path to write the quality report JSON (optional)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print the full quality report to stdout",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.out) if args.out else input_path.with_name(input_path.stem + "_clean.csv")

    print(f"Running Claude preprocessing agent on: {input_path.name}")
    print(f"Model: {args.model}")
    print()

    df_clean, quality_report, briefing = preprocess_csv(
        input_path,
        model=args.model,
    )

    # Save cleaned data
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(out_path, index=False)
    print(f"Cleaned data saved to: {out_path}")
    print(f"Shape: {df_clean.shape}")
    print()

    # Print briefing
    print("=== Agent Briefing ===")
    print(briefing)
    print()

    # Optionally print full quality report
    if args.verbose and quality_report:
        print("=== Quality Report ===")
        print(json.dumps(quality_report, indent=2, ensure_ascii=False))
        print()

    # Save quality report
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(quality_report, indent=2, ensure_ascii=False))
        print(f"Quality report saved to: {report_path}")
    elif quality_report:
        issues = quality_report.get("issues", [])
        if issues:
            print("Issues detected:")
            for issue in issues:
                print(f"  • {issue}")


if __name__ == "__main__":
    main()
