"""Supplementary statistical analysis for COGS 185 Final Project report."""

import csv
import math
from collections import defaultdict

CSV_PATH = "results/all_results.csv"

def load_csv():
    rows = []
    with open(CSV_PATH) as f:
        for row in csv.DictReader(f):
            row["clip_score"] = float(row["clip_score"])
            row["gen_time_sec"] = float(row["gen_time_sec"])
            rows.append(row)
    return rows

def mean(vals):
    return sum(vals) / len(vals)

def std(vals):
    m = mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))

def sep(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

rows = load_csv()

# ── A: Per-prompt × per-CFG mean CLIP ──────────────────────────────────────
sep("A: Per-Prompt × CFG Scale Mean CLIP")
cfg_rows = [r for r in rows if r["experiment"] == "exp1_cfg_scale"]
prompts = sorted(set(r["prompt"] for r in cfg_rows))
cfg_vals = sorted(set(int(r["value"]) for r in cfg_rows))

header = f"{'CFG':<5}" + "".join(f"  {p[:20]:<22}" for p in prompts) + "  Mean"
print(header)
for cfg in cfg_vals:
    pmeans = []
    row_str = f"{cfg:<5}"
    for p in prompts:
        vals = [r["clip_score"] for r in cfg_rows if int(r["value"]) == cfg and r["prompt"] == p]
        m = mean(vals)
        pmeans.append(m)
        row_str += f"  {m:.4f}               "
    row_str += f"  {mean(pmeans):.4f}"
    print(row_str)

# Per-prompt total gain (CFG=1 to CFG=15)
print("\nPer-prompt gain CFG=1 → CFG=15:")
for p in prompts:
    v1 = mean([r["clip_score"] for r in cfg_rows if int(r["value"]) == 1 and r["prompt"] == p])
    v15 = mean([r["clip_score"] for r in cfg_rows if int(r["value"]) == 15 and r["prompt"] == p])
    print(f"  {p[:40]:<42} Δ={v15-v1:+.4f}")

# ── B: Efficiency Pareto (Exp 2) ────────────────────────────────────────────
sep("B: Steps Efficiency Analysis")
step_rows = [r for r in rows if r["experiment"] == "exp2_steps"]
steps_vals = sorted(set(int(r["value"]) for r in step_rows))

print(f"{'Steps':<8} {'CLIP':<8} {'Time(s)':<10} {'CLIP/s':<10} {'%CLIP_max':<12} {'%Time_rel_10'}")
clips = {}
times = {}
for s in steps_vals:
    c = mean([r["clip_score"] for r in step_rows if int(r["value"]) == s])
    t = mean([r["gen_time_sec"] for r in step_rows if int(r["value"]) == s])
    clips[s] = c
    times[s] = t

max_clip = max(clips.values())
base_time = times[10]
for s in steps_vals:
    pct_clip = clips[s] / max_clip * 100
    pct_time = times[s] / base_time * 100
    print(f"{s:<8} {clips[s]:.4f}   {times[s]:<10.2f} {clips[s]/times[s]:.4f}     {pct_clip:.1f}%        {pct_time:.1f}%")

print(f"\nEfficiency ratio collapse (CLIP/s at 10 steps vs 50 steps): "
      f"{clips[10]/times[10]:.4f} → {clips[50]/times[50]:.4f} "
      f"({(clips[10]/times[10])/(clips[50]/times[50]):.1f}x degradation)")
print(f"Steps 30 vs 50 CLIP difference: {abs(clips[30]-clips[50]):.4f} ({times[50]-times[30]:.1f}s extra cost)")

# ── C: Coefficient of Variation ─────────────────────────────────────────────
sep("C: Coefficient of Variation by Experiment")
print(f"{'Exp':<20} {'Value':<30} {'Mean':<8} {'Std':<8} {'CV%'}")
for exp in ["exp1_cfg_scale", "exp2_steps", "exp3_scheduler", "exp5_model"]:
    exp_rows = [r for r in rows if r["experiment"] == exp]
    values = sorted(set(r["value"] for r in exp_rows))
    for v in values:
        cs = [r["clip_score"] for r in exp_rows if r["value"] == v]
        m, s = mean(cs), std(cs)
        cv = s / m * 100
        print(f"{exp:<20} {v[:28]:<30} {m:.4f}   {s:.4f}   {cv:.2f}%")

# ── D: Between vs Within Condition Variance (Exp 3 Scheduler) ───────────────
sep("D: Variance Decomposition — Exp 3 Scheduler")
sched_rows = [r for r in rows if r["experiment"] == "exp3_scheduler"]
schedulers = sorted(set(r["value"] for r in sched_rows))
all_clips = [r["clip_score"] for r in sched_rows]
grand_mean = mean(all_clips)
total_var = std(all_clips) ** 2

# Between-condition variance (variance of group means)
group_means = [mean([r["clip_score"] for r in sched_rows if r["value"] == s]) for s in schedulers]
between_var = std(group_means) ** 2

within_var = total_var - between_var
print(f"Total variance in Exp3 CLIP scores: {total_var:.8f}")
print(f"Between-scheduler variance:         {between_var:.8f}  ({between_var/total_var*100:.1f}% of total)")
print(f"Within-scheduler variance:          {within_var:.8f}  ({within_var/total_var*100:.1f}% of total)")
print("→ Within-condition (seed/prompt) variance dominates; scheduler choice is second-order.")

# ── E: Cross-experiment DDIM consistency ────────────────────────────────────
sep("E: Cross-Experiment DDIM Baseline Consistency")
# Exp2 steps=20 DDIM vs Exp3 DDIM — same model, same settings, same prompts/seeds
exp2_ddim = sorted(
    [(r["prompt"], r["seed"], r["clip_score"]) for r in rows
     if r["experiment"] == "exp2_steps" and r["value"] == "20"],
    key=lambda x: (x[0], x[1])
)
exp3_ddim = sorted(
    [(r["prompt"], r["seed"], r["clip_score"]) for r in rows
     if r["experiment"] == "exp3_scheduler" and r["value"] == "DDIM"],
    key=lambda x: (x[0], x[1])
)
print(f"{'Prompt[:25]':<27} {'Seed':<6} {'Exp2 CLIP':<12} {'Exp3 CLIP':<12} {'Match'}")
for (p2, s2, c2), (p3, s3, c3) in zip(exp2_ddim, exp3_ddim):
    match = "✓" if abs(c2 - c3) < 1e-9 else "✗"
    print(f"{p2[:25]:<27} {s2:<6} {c2:<12.4f} {c3:<12.4f} {match}")

# ── F: Model × Prompt Interaction ───────────────────────────────────────────
sep("F: Model × Prompt Interaction (Exp 5)")
model_rows = [r for r in rows if r["experiment"] == "exp5_model"]
models = sorted(set(r["value"] for r in model_rows))
model_labels = {m: "SD v1.4" if "v1-4" in m else "SD v1.5" for m in models}
prompts_m = sorted(set(r["prompt"] for r in model_rows))

print(f"{'Prompt[:30]':<32}", end="")
for m in models:
    print(f"  {model_labels[m]:<10}", end="")
print("  Δ(v1.5−v1.4)")
for p in prompts_m:
    print(f"{p[:30]:<32}", end="")
    vals = {}
    for m in models:
        cs = [r["clip_score"] for r in model_rows if r["value"] == m and r["prompt"] == p]
        vals[m] = mean(cs)
        print(f"  {vals[m]:.4f}    ", end="")
    delta = vals[models[1]] - vals[models[0]]  # v1.5 - v1.4
    print(f"  {delta:+.4f}")

# ── Summary Table for Paper ─────────────────────────────────────────────────
sep("SUMMARY: All Experiments (for paper Table 1)")
print(f"{'Experiment':<20} {'Variable':<15} {'N_cond':<8} {'CLIP range':<15} {'Best value':<15} {'Best CLIP'}")
for exp, var in [("exp1_cfg_scale","cfg_scale"),("exp2_steps","steps"),
                 ("exp3_scheduler","scheduler"),("exp5_model","model")]:
    exp_rows = [r for r in rows if r["experiment"] == exp]
    values = sorted(set(r["value"] for r in exp_rows))
    group_clips = {v: mean([r["clip_score"] for r in exp_rows if r["value"] == v]) for v in values}
    best_v = max(group_clips, key=group_clips.get)
    worst_v = min(group_clips, key=group_clips.get)
    clip_range = group_clips[best_v] - group_clips[worst_v]
    print(f"{exp:<20} {var:<15} {len(values):<8} {clip_range:.4f}         {best_v[:12]:<15} {group_clips[best_v]:.4f}")

print("\nDone. Use these tables in your report.")
