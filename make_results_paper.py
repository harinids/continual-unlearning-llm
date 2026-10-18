import json, glob, math, csv, statistics as st
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

def at(pts, g):          # first time forgetting reaches g, interpolated in log G
    for (g0, a0, b0), (g1, a1, b1) in zip(pts, pts[1:]):
        if g0 <= g <= g1 and g1 > g0:
            w = (math.log(g) - math.log(g0)) / (math.log(g1) - math.log(g0))
            return a0 + w * (a1 - a0), b0 + w * (b1 - b0)
    return None

def files(pat): return [f for f in sorted(glob.glob(D + pat + "_run.json")) if "diag" not in f]

GROUPS = {
 "qwen05": {
  "Geometry (ours), lr 6e-6": "qwen_geometry_s1v1_lr6e-06_ctlfix_seed*_wiki",
  "Geometry (ours), lr 8e-6": "qwen_geometry_s1v1_lr8e-06_ctlfix_seed*_wiki",
  "GA + EWC":                 "qwen_safetystack_s1v1_lr6e-06_ctlfix_seed*_wiki",
  "NPO":                      "qwen_npo_s1v1_lr6e-06_ctlfix_seed*_wiki",
  "GradDiff + EWC, lr 6e-6":  "qwen_graddiff_s1v1_lr6e-06_ctlfix_seed*_wiki",
  "GradDiff + EWC, lr 8e-6":  "qwen_graddiff_s1v1_lr8e-06_ctlfix_seed*_wiki",
  "RMU":                      "qwen_rmu2_s1v1_lr1x_ctlfix_seed*_wiki",
  "Random subspace (control)": "qwen_randsub_s1v1_lr6e-06_ctlfix_seed*_wiki"},
 "qwen15": {
  "Geometry (ours), lr 6e-6": "qwen15_geometry_s1v1_lr6e-06_ctlfix_seed*_wiki",
  "Geometry (ours), lr 8e-6": "qwen15_geometry_s1v1_lr8e-06_ctlfix_seed*_wiki",
  "Geometry (ours), lr 1e-5": "qwen15_geometry_s1v1_lr1e-05_ctlfix_seed*_wiki",
  "GA + EWC":                 "qwen15_safetystack_s1v1_lr6e-06_ctlfix_seed*_wiki"},
  "GradDiff + EWC":            "qwen15_graddiff_s1v1_lr6e-06_ctlfix_seed*_wiki"},
 "gpt2": {   # discussion
  "Geometry (ours)": "gpt2_geometry_s1v2_lr1x_ctlfix_seed*_wiki",
  "GA + EWC":        "gpt2_safetystack_s1v2_lr1x_ctlfix_seed*_wiki",
  "NPO":             "gpt2_npo_s1v2_lr1x_ctlfix_seed*_wiki",
  "GradDiff + EWC":  "gpt2_graddiff_s1v2_lr1x_ctlfix_seed*_wiki",
  "EWC-GA, fixed lambda": "gpt2_ewcgafix_lam100000_s1v2_lr1x_ctlfix_seed*_wiki",
  "RMU":             "gpt2_rmu2_s1v2_lr10x_ctlfix_seed*_wiki",
  "Random subspace (control)": "gpt2_geometry_s1v2_randsub_ctlfix_seed*_wiki"},
 "hx": {     # discussion
  "Geometry (ours), lr 2x": "gpt2hx_geometry_hx1_lr2x_ctlfix_seed*_wiki",
  "Geometry (ours), lr 3x": "gpt2hx_geometry_hx1_lr3x_ctlfix_seed*_wiki",
  "GA + EWC":               "gpt2hx_safetystack_hx1_lr1x_ctlfix_seed*_wiki",
  "Random subspace (control)": "gpt2hx_geometry_hx1_randsub_ctlfix_seed*_wiki"}}

def col(n):
    if n.startswith("Geometry"): return "#0EA88A" if ("6e-6" in n or "2x" in n or n == "Geometry (ours)") else "#0B6E5A"
    return {"GA + EWC": "#D64550", "NPO": "#A0522D", "RMU": "#6C5CE7", "Random subspace (control)": "#9AA5B8",
            "EWC-GA, fixed lambda": "#F2A33A"}.get(n, "#C0508C" if n.startswith("GradDiff") else "#444444")

def curve(pat, c):
    fs = files(pat); xs, m, lo, hi = [], [], [], []
    for g in GRID:
        v = [at(traj(f), g) for f in fs]; v = [x[c] for x in v if x]
        if len(v) >= 2 or (len(v) == 1 and len(fs) == 1):
            xs.append(g); m.append(st.mean(v)); lo.append(min(v)); hi.append(max(v))
    return xs, m, lo, hi, len(fs)

rows = []
for fam, groups in GROUPS.items():
    for name, pat in groups.items():
        fs = files(pat)
        for g in GRID:
            v = [at(traj(f), g) for f in fs]; v = [x for x in v if x]
            if v: rows.append([fam, name, g, round(st.mean(x[0] for x in v), 3), round(st.mean(x[1] for x in v), 3), len(v), len(fs)])
with open("matched_G_table_paper.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["family", "method", "G", "retain_drift_x", "eval_drift_x", "n_seeds_reaching_G", "n_seeds"]); w.writerows(rows)

def panel(a, fam, c, title):
    for name, pat in GROUPS[fam].items():
        xs, m, lo, hi, tot = curve(pat, c)
        if not xs: continue
        ls = "--" if "control" in name else "-"
        a.plot(xs, m, ls, marker="o", ms=4, lw=2, color=col(name), label=f"{name} (n={tot})"); a.fill_between(xs, lo, hi, color=col(name), alpha=0.12)
    a.set_xscale("log"); a.xaxis.set_minor_formatter(NullFormatter()); a.set_xticks([2, 3, 5, 10]); a.set_xticklabels(["2", "3", "5", "10"])
    a.axhline(1, color="k", lw=0.6, ls=":"); a.set_xlabel("forgetting level G"); a.set_ylabel("drift (1.0 = undamaged)")
    a.set_title(title, fontsize=11); a.grid(alpha=0.25); a.legend(fontsize=7.5, loc="upper left")

fig, ax = plt.subplots(2, 3, figsize=(15, 8.6))
panel(ax[0][0], "qwen05", 0, "Qwen2.5-0.5B: retain drift"); panel(ax[0][1], "qwen05", 1, "Qwen2.5-0.5B: held-out (eval) drift")
panel(ax[0][2], "qwen15", 0, "Qwen2.5-1.5B: retain drift (1-3 seeds)")
panel(ax[1][0], "gpt2", 0, "Discussion, GPT-2: retain drift"); panel(ax[1][1], "gpt2", 1, "Discussion, GPT-2: eval drift")
panel(ax[1][2], "hx", 0, "Discussion, HateXplain (GPT-2): retain drift")
fig.suptitle("Damage per unit of forgetting: Geometry (ours) vs other unlearning methods (shaded = min-max over seeds; curve stops where <2 seeds reach that G)", fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig("results_figure_paper.png", dpi=170)
print("wrote matched_G_table_paper.csv and results_figure_paper.png")
