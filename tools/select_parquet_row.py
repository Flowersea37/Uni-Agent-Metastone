"""Write a selected contiguous parquet slice without modifying the source."""

import argparse
from pathlib import Path

import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--index", type=int, default=1)
    parser.add_argument("--count", type=int, default=1)
    args = parser.parse_args()
    table = pq.read_table(args.source)
    if args.index < 0 or args.count < 1 or args.index + args.count > len(table):
        raise IndexError(
            f"row slice [{args.index}, {args.index + args.count}) outside parquet with {len(table)} rows"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table.slice(args.index, args.count), args.output)
    print(f"Selected rows [{args.index}, {args.index + args.count}): {args.output}")


if __name__ == "__main__":
    main()
