#!/usr/bin/env python3
"""
plot_trials_debug.py  – debugging version
  • scans every column for fields like '12s:0.034'
  • prints detailed messages so we can trace what it sees
"""

import sys, re, csv, pathlib
from pathlib import Path
import matplotlib.pyplot as plt

DEBUG = True            # turn off once it works

if len(sys.argv) < 2:
    print("Usage: plot_trials_debug.py file1.csv [file2.csv …]")
    sys.exit(1)

def dbg(msg):
    if DEBUG:
        print(msg)

# ─── extract (sec,err) pairs from a CSV row ───────────────────
# ---------------------------------------------------------------------
def parse_row(row, row_idx, file_label):
    """
    Extract (sec, err) pairs from one CSV row, tolerant of extra spaces
    or text.  Accepts any cell that contains the substring 's:'.
    """
    pairs = []
    for field in row:
        if "s:" not in field:
            continue
        try:
            sec_str, err_str = field.split("s:", 1)
            sec = int(sec_str.strip())
            err = float(err_str.strip())
            pairs.append((sec, err))
        except ValueError:
            dbg(f"[{file_label}] row {row_idx}: could not parse '{field}'")
    dbg(f"[{file_label}] row {row_idx}: found {len(pairs)} pairs")
    return pairs
# ---------------------------------------------------------------------



plt.rcParams.update({"figure.dpi": 120, "font.size": 10})
colors = plt.cm.tab10.colors
impl_handles = {}

for file_idx, csv_path in enumerate(sys.argv[1:], 1):
    csv_path = pathlib.Path(csv_path)
    if not csv_path.exists():
        print(f"[warn] {csv_path} not found, skipping.")
        continue

    with csv_path.open(newline='') as fh:
        rows = [row for row in csv.reader(fh) if row]

    if not rows:
        print(f"[warn] {csv_path} empty, skipping.")
        continue

    impl_name = rows[0][0]
    dbg(f"Processing file {csv_path}  (impl = {impl_name})")

    fig, ax = plt.subplots()
    ax.set_title(f"{impl_name} – per‑trial error")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Error (m)")
    ax.grid(True, alpha=0.3)

    colour = colors[(file_idx-1) % len(colors)]
    plotted_any = False

    for r_idx, row in enumerate(rows, 1):
        pairs = parse_row(row, r_idx, impl_name)
        if not pairs:
            continue
        t, e = zip(*pairs)
        handle, = ax.plot(t, e, color=colour, alpha=0.7, linewidth=1)
        plotted_any = True

    if not plotted_any:
        dbg(f"No valid 'Ns:err' pairs found in {csv_path}")

    if plotted_any:
        impl_handles[impl_name] = (handle, colour)
        out_png = csv_path.with_name(csv_path.stem + "_trial.png")   # or “…_trials.png”
        fig.tight_layout()
        fig.savefig(out_png)
        print(f"wrote {out_png}")
    else:
        plt.close(fig)

# ─── combined overlay if we have multiple impls ───────────────
if len(impl_handles) > 1:
    fig, ax = plt.subplots()
    ax.set_title("Combined trials")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Error (m)")
    ax.grid(True, alpha=0.3)

    for file_idx, csv_path in enumerate(sys.argv[1:], 1):
        csv_path = pathlib.Path(csv_path)
        with csv_path.open(newline='') as fh:
            rows = [row for row in csv.reader(fh) if row]
        impl_name = rows[0][0]
        colour = impl_handles[impl_name][1]

        for row in rows:
            pairs = parse_row(row, 0, impl_name)   # reuse helper
            if not pairs:
                continue
            t, e = zip(*pairs)
            ax.plot(t, e, color=colour, alpha=0.7, linewidth=1,
                    label=impl_name if row[1] == '1' else None)

    ax.legend()
    fig.tight_layout()
    fig.savefig("combined_rib.png")
    print("wrote combined.png")
else:
    dbg("combined plot skipped (only one impl)")
