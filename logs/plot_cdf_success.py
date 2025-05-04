#!/usr/bin/env python3
"""
plot_cdf_success_flexible.py – CDF of success times (robust column order)

Changes (2025‑05‑03)
───────────────────
• **Success percentage is now relative to *all* trials**, not just the
  successful ones.  A stalled / timed‑out run therefore leaves the curve
  below 100 %, visualising reliability.
• Still autodetects the status column, extracts finish time in the same
  three‑tier order.

Usage
-----
    python plot_cdf_success_flexible.py pid.csv mpc.csv
Produces `<impl>_cdf.png` and a combined `cdf_success.png`.
"""
import sys, csv, pathlib, math
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.dpi": 120, "font.size": 10})

if len(sys.argv) < 2:
    print("Usage: plot_cdf_success_flexible.py file1.csv [file2.csv …]")
    sys.exit(1)

STATUS_SET = {"success", "timeout", "stall", "fail", "failure"}
COLORS = plt.cm.tab10.colors

impl2times:   dict[str, list[float]] = defaultdict(list)   # successful
impl2total:   dict[str, int]          = defaultdict(int)    # all trials

# ───────────────────────────────── ingest CSVs ───────────────────────
for csv_path in map(pathlib.Path, sys.argv[1:]):
    if not csv_path.exists():
        print(f"[warn] {csv_path} not found – skipping")
        continue

    with csv_path.open(newline="") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if not row:
                continue

            impl = row[0].strip()
            impl2total[impl] += 1              # count every trial

            # locate status column dynamically
            status_idx = None; status_val = None
            for i, cell in enumerate(row):
                v = cell.strip().lower()
                if v in STATUS_SET:
                    status_idx, status_val = i, v
                    break
            if status_val != "success":
                continue

            # -------- success time extraction ------------------------
            t_finish = None
            # 1) first finite numeric cell **after** status
            for cell in row[status_idx + 1:]:
                try:
                    num = float(cell)
                    if math.isfinite(num):
                        t_finish = num; break
                except ValueError:
                    continue
            # 2) total_sec (col‑2)
            if t_finish is None and len(row) > 2:
                try:
                    num = float(row[2])
                    if math.isfinite(num):
                        t_finish = num
                except ValueError:
                    pass
            # 3) max "Ns:err" timestamp
            if t_finish is None:
                ts = [int(c.split('s:')[0]) for c in row if 's:' in c]
                if ts:
                    t_finish = max(ts)
            if t_finish is not None:
                impl2times[impl].append(t_finish)

# ───────────────────────── plots ─────────────────────────────────────
if not impl2times:
    print("No successful trials found.")
    sys.exit(0)

# per‑file -----------------------------------------------------------
for idx, impl in enumerate(impl2total):
    total_n = impl2total[impl]
    succ_times = np.array(impl2times.get(impl, []), dtype=float)
    succ_times.sort()
    if succ_times.size == 0:
        print(f"[warn] {impl}: 0 successes; skipping individual plot")
        continue
    y = np.arange(1, succ_times.size + 1) / total_n * 100.0  # denom = total trials

    fig, ax = plt.subplots()
    ax.step(succ_times, y, where="post", color=COLORS[idx % 10])
    ax.set_title(f"CDF – {impl}  (success {len(succ_times)}/{total_n})")
    ax.set_xlabel("Time to finish (s)")
    ax.set_ylabel("Success of all trials (%)")
    ax.set_ylim(0, 100); ax.grid(True, alpha=.3)
    fig.tight_layout(); fig.savefig(f"{impl}_cdf.png")
    print(f"wrote {impl}_cdf.png")

# combined -----------------------------------------------------------
if len(impl2total) > 1:
    fig, ax = plt.subplots()
    ax.set_title("CDF of success times – all implementations")
    ax.set_xlabel("Time to finish (s)")
    ax.set_ylabel("Success of all trials (%)")
    ax.set_ylim(0, 100); ax.grid(True, alpha=.3)

    for idx, impl in enumerate(sorted(impl2total)):
        total_n = impl2total[impl]
        times = np.array(impl2times.get(impl, []), dtype=float)
        times.sort()
        if times.size == 0:
            continue
        y = np.arange(1, times.size + 1) / total_n * 100.0
        ax.step(times, y, where="post", color=COLORS[idx % 10], label=f"{impl} ({len(times)}/{total_n})")

    ax.legend(); fig.tight_layout(); fig.savefig("cdf_success.png")
    print("wrote cdf_success.png")
