"""Usage: python scripts/validate_manifest.py path/to/ham10000_master_manifest.csv"""

import argparse
import json

import pandas as pd

from ham10000_hybrid.data_validation import validate_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate HAM10000 seven-class manifest")
    parser.add_argument("manifest_csv", help="Path to metadata + saved train/validation/test splits")
    args = parser.parse_args()
    df = pd.read_csv(args.manifest_csv)
    print(json.dumps(validate_manifest(df), indent=2))


if __name__ == "__main__":
    main()
