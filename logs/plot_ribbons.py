#!/usr/bin/env python3
"""
plot_trials_ribbon.py
────────────────────────────────────────────────────────
Visualise per‑second cube‑goal error traces.
• Thin line for each trial
• Thick line = mean across trials
• Shaded ±1 σ ribbon
• log‑scale y‑axis to fit both mm‑level and stalled runs

Usage
-----
python plot_trials_ribbon.py  run_pid.csv  run_mpc.csv

Outputs
-------
run_pid.png       – all PID trials
run_mpc.png       – all MPC trials
combined.png      – overlay of the mean±σ ribbon for each impl
```"""
import sys, csv, re, pathlib, numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.dpi": 120, "font.size": 10})

if len(sys.argv) < 2:
    print("Usage: plot_trials_ribbon.py file1.csv [file2.csv …]")
    sys.exit(1)

# ─── helpers ─────────────────────────────────────────────────
PAT = re.compile(r"^(\d+)s:([\d.]+)")

def row_pairs(row):
    """return list[(sec,int), err(float)] from a CSV row"""
    pairs = []
    for cell in row:
        m = PAT.match(cell.strip())
        if m:
            sec, err = int(m.group(1)), float(m.group(2))
            pairs.append((sec, err))
    return pairs

colors = plt.cm.tab10.colors
impl_color = {}
impl_ribbon_data = {}

for file_idx, csv_path in enumerate(sys.argv[1:], 1):
    p = pathlib.Path(csv_path)
    if not p.exists():
        print(f"[warn] {p} not found, skipping.")
        continue

    with p.open(newline="") as fh:
        rows = [r for r in csv.reader(fh) if r]
    if not rows:
        print(f"[warn] {p} empty; skipping.")
        continue

    impl = rows[0][0]
    colour = colors[(file_idx-1) % len(colors)]
    impl_color[impl] = colour

    # --- gather trials --------------------------------------
    trials = []
    for row in rows:
        pairs = row_pairs(row)
        if pairs:
            trials.append(pairs)
    if not trials:
        print(f"[warn] no 'Ns:err' pairs in {p}")
        continue

    # time grid (union of all seconds)
    all_secs = sorted({sec for tr in trials for sec, _ in tr})
    err_matrix = np.full((len(trials), len(all_secs)), np.nan)
    sec_idx = {s:i for i,s in enumerate(all_secs)}
    for r, tr in enumerate(trials):
        for sec, err in tr:
            err_matrix[r, sec_idx[sec]] = err
    mean = np.nanmean(err_matrix, 0)
    std  = np.nanstd (err_matrix, 0)
    impl_ribbon_data[impl] = (all_secs, mean, std)

    # --- per‑file plot --------------------------------------
    fig, ax = plt.subplots()
    ax.set_title(f"{impl} – per‑trial error")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Error (m)"); ax.grid(True, alpha=.3)
    # all finite errors in this file
    valid_err = err_matrix[np.isfinite(err_matrix)]

    ymin = max(valid_err.min(), 1e-4)     # clip at 1e‑4 so log(0) never occurs
    ymax = valid_err.max() * 1.05         # 5 % head‑room

    ax.set_yscale("log")
    ax.set_ylim(ymin, ymax)

    # (optional) tighten x‑axis too
    ax.set_xlim(0, all_secs[-1])


    # thin lines
    for tr in trials:
        t, e = zip(*tr)
        e = np.maximum(e, 1e-4)   # avoid log(0)
        ax.plot(t, e, color=colour, alpha=0.4, linewidth=0.8)
    # mean ± σ ribbon
    ax.plot(all_secs, mean, color=colour, linewidth=2)
    ax.fill_between(all_secs, mean-std, mean+std, color=colour, alpha=.25)

    fig.tight_layout()
    out_png = p.with_name(p.stem + "_ribbon.png")   # or “…_trials.png”
    fig.savefig(out_png)
    print(f"wrote {out_png}")

# ─── combined overlay (mean ± σ only) ───────────────────────
if len(impl_ribbon_data) > 1:
    fig, ax = plt.subplots()
    ax.set_title("Combined mean ±1 σ")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Error (m)")
    ax.grid(True, alpha=.3); ax.set_yscale("log")

    for impl, (secs, mean, std) in impl_ribbon_data.items():
        c = impl_color[impl]
        ax.plot(secs, mean, color=c, linewidth=2, label=impl)
        ax.fill_between(secs, mean-std, mean+std, color=c, alpha=.25)

    ax.legend(); fig.tight_layout(); fig.savefig("combined_ribbon.png")
    print("wrote combined.png")
