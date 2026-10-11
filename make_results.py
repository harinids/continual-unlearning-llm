"""Regenerate the matched-forgetting figure and table from results/phase3_utility/*_run.json.
Run from the repo root:  python make_results.py   (CPU only, no GPU needed)."""
import json, glob, math, csv, os, statistics as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

D = "results/phase3_utility/"
GRID = [1.5, 2, 3, 4, 5, 7, 10]

def traj(f):
    d = json.load(open(f)); ob = d["overall_baseline"]; pm = d["persistence_matrix"]; pts = []
    for t in sorted(pm, key=int):
        G = sum(a["forget_ppl"] for a in pm[t].values()) / len(pm[t]) / ob["forget_ppl"]
        r = d["request_results"][int(t)]["final"]
        pts.append((G, r["retain_ppl"] / ob["retain_ppl"], r["eval_ppl"] / ob["eval_ppl"]))
    return pts

def at(pts, g):                                   # interpolate (log G) the first time forgetting reaches g
    for (g0, a0, b0), (g1, a1, b1) in zip(pts, pts[1:]):
        if g0 <= g <= g1 and g1 > g0:
            w = (math.log(g) - math.log(g0)) / (math.log(g1) - math.log(g0))
            return a0 + w * (a1 - a0), b0 + w * (b1 - b0)
    return None

def files(pat): return [f for f in sorted(glob.glob(D + pat + "_run.json")) if "diag" not in f]

GROUPS = {   # name -> glob
 "gpt2": {
  "Geometry (ours)":            "gpt2_geometry_s1v2_lr1x_ctlfix_seed*_wiki",
  "Geometry, projected update": "gpt2_geometry_s1v2_pu_ctlfix_seed*_wiki",
  "Random subspace (control)":  "gpt2_geometry_s1v2_randsub_ctlfix_seed*_wiki",
  "GA + EWC":                   "gpt2_safetystack_s1v2_lr1x_ctlfix_seed*_wiki",
  "EWC-GA, fixed lambda 1e5":   "gpt2_ewcgafix_lam100000_s1v2_lr1x_ctlfix_seed*_wiki",
  "RMU (10x lr)":               "gpt2_rmu2_s1v2_lr10x_ctlfix_seed*_wiki"},
 "qwen05": {
  "Geometry (ours)": "qwen_geometry_s1v1_lr6e-06_ctlfix_seed*_wiki",
  "GA + EWC":        "qwen_safetystack_s1v1_lr6e-06_ctlfix_seed*_wiki",
  "RMU (1x lr)":     "qwen_rmu2_s1v1_lr1x_ctlfix_seed*_wiki"},
 "hx": {
  "Geometry (2x lr)": "gpt2hx_geometry_hx1_lr2x_ctlfix_seed*_wiki",
  "Geometry (3x lr)": "gpt2hx_geometry_hx1_lr3x_ctlfix_seed*_wiki",
  "Random subspace (control)": "gpt2hx_geometry_hx1_randsub_ctlfix_seed*_wiki",
  "GA + EWC": "gpt2hx_safetystack_hx1_lr1x_ctlfix_seed*_wiki"},
 "qwen15": {
  "Geometry (ours)": "qwen15_geometry_s1v1_lr6e-06_ctlfix_seed*_wiki",
  "Geometry, lr 12e-6": "qwen15_geometry_s1v1_lr1.2e-05_ctlfix_seed*_wiki",
  "Geometry, lr 18e-6": "qwen15_geometry_s1v1_lr1.8e-05_ctlfix_seed*_wiki",
  "GA + EWC":        "qwen15_safetystack_s1v1_lr6e-06_ctlfix_seed*_wiki"}}
KS = {"k = 0 (random)": "gpt2_geometry_s1v2_randsub_ctlfix_seed*_wiki", "k = 5": "gpt2_geometry_s1v2_k5_ctlfix_seed*_wiki",
      "k = 10": "gpt2_geometry_s1v2_lr1x_ctlfix_seed*_wiki", "k = 20": "gpt2_geometry_s1v2_k20_ctlfix_seed*_wiki"}
COL = {"Geometry (ours)": "#0EA88A", "Geometry, projected update": "#0B6E5A", "Random subspace (control)": "#9AA5B8",
       "GA + EWC": "#D64550", "RMU (1x lr)": "#6C5CE7", "Geometry (2x lr)": "#0EA88A", "Geometry (3x lr)": "#0B6E5A",
       "Geometry, lr 12e-6": "#0B6E5A", "Geometry, lr 18e-6": "#7FD1BF", "EWC-GA, fixed lambda 1e5": "#F2A33A", "RMU (10x lr)": "#6C5CE7"}

def curve(pat, col):
    fs = files(pat); xs, m, lo, hi, n = [], [], [], [], []
    for g in GRID:
        v = [at(traj(f), g) for f in fs]; v = [x[col] for x in v if x]
        if len(v) >= 2 or (len(v) == 1 and len(fs) == 1):
            xs.append(g); m.append(st.mean(v)); lo.append(min(v)); hi.append(max(v)); n.append(len(v))
    return xs, m, lo, hi, n, len(fs)

rows = []
for fam, groups in GROUPS.items():
    for name, pat in groups.items():
        for g in GRID:
            fs = files(pat); v = [at(traj(f), g) for f in fs]; v = [x for x in v if x]
            if v: rows.append([fam, name, g, round(st.mean(x[0] for x in v), 3), round(st.mean(x[1] for x in v), 3), len(v), len(fs)])
for name, pat in KS.items():
    for g in (2, 3, 5):
        fs = files(pat); v = [at(traj(f), g) for f in fs]; v = [x for x in v if x]
        if v: rows.append(["k_ladder", name, g, round(st.mean(x[0] for x in v), 3), round(st.mean(x[1] for x in v), 3), len(v), len(fs)])
with open("matched_G_table.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["family", "method", "G", "retain_drift_x", "eval_drift_x", "n_seeds_reaching_G", "n_seeds"]); w.writerows(rows)

fig, ax = plt.subplots(3, 3, figsize=(15, 12.2))
def panel(a, fam, col, title):
    for name, pat in GROUPS[fam].items():
        xs, m, lo, hi, n, tot = curve(pat, col)
        if not xs: continue
        a.plot(xs, m, "-o", ms=4, lw=2, color=COL.get(name, '#444444'), label=f"{name} (n={tot})"); a.fill_between(xs, lo, hi, color=COL.get(name, '#444444'), alpha=0.15)
    a.set_xscale("log"); a.xaxis.set_minor_formatter(NullFormatter()); a.set_xticks([2, 3, 5, 10]); a.set_xticklabels(["2", "3", "5", "10"]); a.axhline(1, color="k", lw=0.6, ls=":")
    a.set_xlabel("forgetting level G (forget perplexity / baseline)"); a.set_ylabel("drift (1.0 = undamaged)"); a.set_title(title, fontsize=11); a.grid(alpha=0.25)
panel(ax[0][0], "gpt2", 0, "GPT-2: retain drift vs forgetting"); panel(ax[0][1], "gpt2", 1, "GPT-2: held-out (eval) drift vs forgetting")
panel(ax[1][0], "qwen05", 0, "Qwen2.5-0.5B: retain drift"); panel(ax[1][1], "qwen05", 1, "Qwen2.5-0.5B: eval drift")
panel(ax[1][2], "qwen15", 0, "Qwen2.5-1.5B: retain drift (2 seeds)")
panel(ax[2][0], "hx", 0, "HateXplain (GPT-2): retain drift"); panel(ax[2][1], "hx", 1, "HateXplain (GPT-2): eval drift"); ax[2][2].axis("off"); ax[2][0].legend(fontsize=8, loc="upper left")
ax[0][0].legend(fontsize=7.5, loc="upper left"); ax[1][0].legend(fontsize=8, loc="upper left"); ax[1][2].legend(fontsize=8, loc="upper left")
a = ax[0][2]; labs = list(KS); vals = []
for name, pat in KS.items():
    v = [at(traj(f), 5) for f in files(pat)]; v = [x[0] for x in v if x]; vals.append(st.mean(v) if v else float("nan"))
a.bar(labs, vals, color=["#9AA5B8", "#7FD1BF", "#0EA88A", "#0B6E5A"]);
for i, x in enumerate(vals): a.text(i, x + 0.05, f"{x:.2f}x", ha="center", fontsize=9)
a.set_ylabel("retain drift at G = 5"); a.set_title("GPT-2: subspace size (3 seeds)", fontsize=11); a.grid(axis="y", alpha=0.25)
fig.suptitle("Damage per unit of forgetting: retain-subspace projection vs baselines (shaded = min-max over seeds; a curve stops where fewer than 2 seeds reach that G)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig("results_figure.png", dpi=170); print("wrote matched_G_table.csv and results_figure.png")
