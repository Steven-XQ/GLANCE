#!/usr/bin/env python3
"""Parse per-cell log files and aggregate to CSV / Markdown.

Two subcommands:
  append   - extract metrics from one cell's log file and append a row to results.csv
  finalize - read results.csv, compute mean+/-std per (dataset, variant), write a Markdown table
"""
import argparse
import csv
import os
import re
import sys
from collections import defaultdict
from statistics import mean, pstdev

# Log lines look like:
#   [INFO] ours wde ---> 0.4067...
#   [INFO] ours fde ---> 0.2103...
#   [INFO] ours SIM ---> 0.2079...
#   [INFO] ours AUC-J ---> 0.7331...
#   [INFO] ours NSS ---> 0.8583...
METRIC_RE = re.compile(r"ours\s+(wde|fde|SIM|AUC-J|NSS)\s+--->\s+([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)")
METRIC_KEY = {"wde": "wde", "fde": "fde", "SIM": "sim", "AUC-J": "auc_j", "NSS": "nss"}
ORDERED = ["wde", "fde", "sim", "auc_j", "nss"]
VARIANT_ORDER = [
    "baseline",
    "v1", "v2", "v3", "v4", "v5", "v6", "v7",
    "v8", "v9", "v10", "v11", "v12",
]


def parse_log(path):
    metrics = {}
    with open(path, "r", errors="replace") as f:
        for line in f:
            m = METRIC_RE.search(line)
            if not m:
                continue
            key = METRIC_KEY[m.group(1)]
            metrics[key] = float(m.group(2))  # last occurrence wins
    return metrics


def cmd_append(args):
    metrics = parse_log(args.log)
    missing = [k for k in ORDERED if k not in metrics]
    if missing:
        print(f"[aggregate] missing metrics {missing} in {args.log}", file=sys.stderr)
        sys.exit(1)

    header_needed = not os.path.exists(args.csv) or os.path.getsize(args.csv) == 0
    with open(args.csv, "a", newline="") as f:
        w = csv.writer(f)
        if header_needed:
            w.writerow(["dataset", "variant", "seed", "epochs"] + ORDERED)
        w.writerow([args.dataset, args.variant, args.seed, args.epochs] + [metrics[k] for k in ORDERED])
    print(f"[aggregate] appended: {args.dataset} {args.variant} seed={args.seed} -> {args.csv}")


def cmd_finalize(args):
    by_group = defaultdict(lambda: defaultdict(list))  # (dataset, variant) -> metric -> [values]
    with open(args.csv) as f:
        r = csv.DictReader(f)
        for row in r:
            key = (row["dataset"], row["variant"])
            for m in ORDERED:
                by_group[key][m].append(float(row[m]))

    datasets = sorted({k[0] for k in by_group})
    variants_seen = {k[1] for k in by_group}
    ordered_variants = [v for v in VARIANT_ORDER if v in variants_seen]
    ordered_variants += sorted(v for v in variants_seen if v not in VARIANT_ORDER)

    header_cells = ["Variant", "n",
                    "WDE ↓", "FDE ↓",
                    "SIM ↑", "AUC-J ↑", "NSS ↑"]

    lines = []
    for ds in datasets:
        lines.append(f"## {ds.upper()} — 35 epochs, mean ± std over seeds\n")
        lines.append("| " + " | ".join(header_cells) + " |")
        lines.append("|" + "---|" * len(header_cells))
        for v in ordered_variants:
            stats = by_group.get((ds, v))
            if not stats:
                continue
            n = len(stats["wde"])
            row = [v, str(n)]
            for m in ORDERED:
                xs = stats[m]
                mu = mean(xs)
                sd = pstdev(xs) if len(xs) > 1 else 0.0
                row.append(f"{mu:.3f} ± {sd:.3f}")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    out = "\n".join(lines)
    with open(args.out, "w") as f:
        f.write(out)
    print(out)
    print(f"[aggregate] wrote {args.out}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    pa = sub.add_parser("append")
    pa.add_argument("--log", required=True)
    pa.add_argument("--dataset", required=True)
    pa.add_argument("--variant", required=True)
    pa.add_argument("--seed", required=True)
    pa.add_argument("--epochs", required=True)
    pa.add_argument("--csv", required=True)
    pa.set_defaults(func=cmd_append)

    pf = sub.add_parser("finalize")
    pf.add_argument("--csv", required=True)
    pf.add_argument("--out", required=True)
    pf.set_defaults(func=cmd_finalize)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
