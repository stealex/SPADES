#!/usr/bin/env python3
"""Generate cumulative CIFRA config files from 82Se_2nu_2betaMinus_cifra.yaml.

For each N in 1..len(e1plus), produces a config file containing the first N
entries of both 'e1plus' and 'melems', with the output location updated to
reflect N.  All other content is preserved verbatim (including comments).

Usage:
    python scripts/gen_cifra_configs.py [--input CONFIG] [--outdir OUTDIR]

Defaults:
    --input  config_files/82Se_2nu_2betaMinus_cifra.yaml
    --outdir config_files/
"""

import argparse
import ast
import re
import sys
from pathlib import Path


def parse_inline_array(line: str) -> list:
    """Extract the Python list from a YAML line like '    key: [v1, v2, ...]'."""
    match = re.search(r":\s*(\[.*\])", line)
    if not match:
        raise ValueError(f"Cannot find inline array in line: {line!r}")
    return ast.literal_eval(match.group(1))


def replace_inline_array(line: str, values: list) -> str:
    """Replace the inline array in *line* with *values*, keeping the key prefix."""
    prefix = re.match(r".*?:\s*", line).group(0)
    formatted = ", ".join(repr(v) if isinstance(v, str) else str(v) for v in values)
    return f"{prefix}[{formatted}]\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="config_files/82Se_2nu_2betaMinus_cifra.yaml",
                        help="Source config file")
    parser.add_argument("--outdir", default="config_files",
                        help="Directory where generated files are written")
    args = parser.parse_args()

    src = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    lines = src.read_text().splitlines(keepends=True)

    # Identify lines containing e1plus and melems arrays
    e1plus_idx = melems_idx = None
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("e1plus:") and "[" in line:
            e1plus_idx = i
        elif stripped.startswith("melems:") and "[" in line:
            melems_idx = i

    if e1plus_idx is None or melems_idx is None:
        sys.exit("Could not find 'e1plus' or 'melems' array lines in the config file.")

    e1plus_all = parse_inline_array(lines[e1plus_idx])
    melems_all = parse_inline_array(lines[melems_idx])

    if len(e1plus_all) != len(melems_all):
        print(f"Warning: e1plus has {len(e1plus_all)} entries but melems has "
              f"{len(melems_all)}.  Will iterate up to min({len(e1plus_all)}, "
              f"{len(melems_all)}).", file=sys.stderr)

    n_max = min(len(e1plus_all), len(melems_all))
    stem = src.stem  # e.g. "82Se_2nu_2betaMinus_cifra"

    for n in range(1, n_max + 1):
        new_lines = list(lines)
        new_lines[e1plus_idx] = replace_inline_array(lines[e1plus_idx], e1plus_all[:n])
        new_lines[melems_idx] = replace_inline_array(lines[melems_idx], melems_all[:n])

        # Update output location and file_prefix to avoid collisions
        tag = f"n{n:02d}"
        for i, line in enumerate(new_lines):
            # location: <value>
            if re.match(r"\s*location:\s*", line):
                new_lines[i] = re.sub(r"(location:\s*).*", rf"\g<1>{stem}_{tag}", line) + "\n"
                new_lines[i] = new_lines[i].replace("\n\n", "\n")
            # file_prefix: <value>
            if re.match(r"\s*file_prefix:\s*", line):
                new_lines[i] = re.sub(r"(file_prefix:\s*).*", rf"\g<1>{stem}_{tag}", line) + "\n"
                new_lines[i] = new_lines[i].replace("\n\n", "\n")

        out_path = outdir / f"{stem}_{tag}.yaml"
        out_path.write_text("".join(new_lines))
        print(f"Written {out_path}")

    print(f"\nDone – {n_max} files generated in '{outdir}'.")


if __name__ == "__main__":
    main()
