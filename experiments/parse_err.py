"""Parse Python tracebacks from a SLURM .err log file.

Each line in .err files is prefixed with "NODE: [rankN]:" from torchrun's
multi-process stderr multiplexer. Multiple ranks emit interleaved tracebacks.
This script de-interleaves them, deduplicates by exception message, and prints
one clean traceback per unique error.

Usage:
    uv run experiments/parse_err.py logs/gipfel-fp8_760m-3305904.err
    uv run experiments/parse_err.py logs/gipfel-fp8_760m-3305904.err --rank 14
    uv run experiments/parse_err.py logs/gipfel-fp8_760m-3305904.err --all
"""

import argparse
import re
import sys
from collections import defaultdict


# Matches the log prefix:  "3: [default2]:[rank14]: some text"
# or simpler prefixes like "0: some text" without rank tags
# Format: "NODE: [defaultGPU]:[rankGLOBAL_RANK]: text"
# NODE is the SLURM node index, GPU is the local GPU index on that node.
_PREFIX_RE = re.compile(r"^\d+: \[default\d+\]:(?:\[rank(\d+)\]: )?(.*)")


def _strip_prefix(line: str) -> tuple[int | None, str]:
    """Return (rank, text) with the torchrun prefix removed."""
    m = _PREFIX_RE.match(line)
    if m:
        rank_str, text = m.group(1), m.group(2)
        rank = int(rank_str) if rank_str is not None else None
        return rank, text
    return None, line


def _collect_traceback_lines(lines: list[str]) -> dict[int, list[str]]:
    """
    Walk through lines and group traceback text by rank.

    Returns {rank: [lines_of_that_traceback, ...]} where each entry is a
    complete Traceback block (from "Traceback (most recent call last):" to the
    final exception line).
    """
    # We may see multiple tracebacks per rank (e.g. chained exceptions).
    # Store all of them; we'll keep the last one per rank as most informative.
    per_rank: dict[int, list[list[str]]] = defaultdict(list)
    active: dict[int, list[str]] = {}

    for raw in lines:
        rank, text = _strip_prefix(raw.rstrip())
        if rank is None:
            continue

        if text.startswith("Traceback (most recent call last):"):
            active[rank] = [text]
        elif rank in active:
            active[rank].append(text)
            # A traceback ends at the exception line: a line that doesn't start
            # with whitespace and isn't "Traceback" nor a "File …" line.
            if text and not text.startswith(" ") and not text.startswith("Traceback"):
                per_rank[rank].append(active.pop(rank))

    # Flush any still-open tracebacks (file ended mid-traceback)
    for rank, tb in active.items():
        per_rank[rank].append(tb)

    return {rank: tbs[-1] for rank, tbs in per_rank.items() if tbs}


def _exception_key(tb: list[str]) -> str:
    """Return the exception type from the last non-empty line of a traceback.

    Uses only the exception class name (before the first colon) so that
    per-rank details like GPU IDs don't prevent deduplication.
    """
    for line in reversed(tb):
        stripped = line.strip()
        if stripped:
            return stripped.split(":")[0]
    return ""


def parse_err(path: str) -> dict[int, list[str]]:
    with open(path) as f:
        lines = f.readlines()
    return _collect_traceback_lines(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("err_file", metavar="FILE", help=".err log file to parse")
    parser.add_argument(
        "--rank",
        type=int,
        metavar="N",
        help="Show traceback only for this rank",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Show one traceback per rank (default: deduplicate by exception message)",
    )
    args = parser.parse_args()

    tracebacks = parse_err(args.err_file)

    if not tracebacks:
        print("No Python tracebacks found.", file=sys.stderr)
        sys.exit(0)

    if args.rank is not None:
        if args.rank not in tracebacks:
            print(f"No traceback found for rank {args.rank}.", file=sys.stderr)
            sys.exit(1)
        tb = tracebacks[args.rank]
        print(f"=== rank {args.rank} ===")
        print("\n".join(tb))
        return

    if args.all:
        for rank in sorted(tracebacks):
            print(f"=== rank {rank} ===")
            print("\n".join(tracebacks[rank]))
            print()
        return

    # Default: deduplicate — show one representative traceback per unique exception.
    seen: dict[str, int] = {}
    for rank in sorted(tracebacks):
        key = _exception_key(tracebacks[rank])
        if key not in seen:
            seen[key] = rank

    if len(seen) == 1:
        rank = next(iter(seen.values()))
        print(f"=== rank {rank} (same error on {len(tracebacks)} rank(s)) ===")
        print("\n".join(tracebacks[rank]))
    else:
        for key, rank in seen.items():
            same = sum(1 for tb in tracebacks.values() if _exception_key(tb) == key)
            print(f"=== rank {rank} ({same} rank(s) with this error) ===")
            print("\n".join(tracebacks[rank]))
            print()


if __name__ == "__main__":
    main()
