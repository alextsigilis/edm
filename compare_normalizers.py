"""Compare the hamming, exact and sampled normalisers on MVAD for one model type.

For every G, the model is fitted three times, once per normaliser, with the same seed. Each
restart therefore starts from the same k-modes initialisation, and the only difference
between the three fits is the normaliser used during training. BIC is always computed from
the hard central sequences, where all three normalisers give the same likelihood.

Two figures are produced:
  * BIC against G, one line per normaliser;
  * for each normaliser, its BIC-optimal model drawn in the style of Figure 7 of Murphy et
    al. (2021): every observed sequence is one row, coloured by state over time, and rows
    are grouped by MAP cluster, with similar sequences placed next to each other.

    python compare_normalizers.py --type UUN
    python compare_normalizers.py --type CU --covariates none --n-init 2
    python compare_normalizers.py --plot-only results/normalizers_UUN.csv
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from edm import NORMALIZERS, EDMType
from main import build_model, covariate_matrix, load_mvad, parse_covariates, sps

# Slots 1-3 of the validated categorical palette, each with its own marker, so the
# normalisers can still be told apart without colour (CVD, greyscale print).
STYLE = {
    "hamming": ("#2a78d6", "o"),
    "exact": ("#eb6834", "s"),
    "sampled": ("#1baf7a", "^"),
}
# State colours for the cluster plots. Blue, green, magenta and yellow follow the paper's
# figures for SC, EM, TR and JL. Six hues cannot all be told apart pairwise; the closest
# pair (magenta/red) is given to TR/HE, which rarely border each other. The SPS labels and
# the printed per-cluster table carry the same information as text.
STATE_STYLE = {
    "EM": ("EMployment", "#008300"),
    "FE": ("Further Education", "#1baf7a"),
    "HE": ("Higher Education", "#e34948"),
    "JL": ("JobLessness", "#eda100"),
    "SC": ("SChool", "#2a78d6"),
    "TR": ("TRaining", "#e87ba4"),
}
SURFACE, INK, INK_SECONDARY, INK_MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRIDLINE, BASELINE = "#e1e0d9", "#c3c2b7"

# Fit settings stored in every table row, so --plot-only can refit the same models.
SETTINGS = ("covariate_names", "weights", "gated_noise", "mc_samples", "n_init", "epochs",
            "lr", "seed")


# ---- data and fitting -------------------------------------------------------------------
def settings_from_args(args):
    return {"covariate_names": ",".join(parse_covariates(args.covariates)) or "none",
            "weights": not args.no_weights, "gated_noise": args.gated_noise,
            "mc_samples": args.mc_samples, "n_init": args.n_init, "epochs": args.epochs,
            "lr": args.lr, "seed": args.seed}


def settings_from_table(table, args):
    if not all(c in table.columns for c in SETTINGS):
        print("the table does not record its fit settings; refitting with the command-line options")
        return settings_from_args(args)
    row = table.iloc[0]
    return {"covariate_names": str(row["covariate_names"]), "weights": bool(row["weights"]),
            "gated_noise": bool(row["gated_noise"]), "mc_samples": int(row["mc_samples"]),
            "n_init": int(row["n_init"]), "epochs": int(row["epochs"]), "lr": float(row["lr"]),
            "seed": int(row["seed"])}


def load_data(csv, settings):
    S, w, df, alphabet = load_mvad(csv)
    names = parse_covariates(settings["covariate_names"])
    return {"S": S, "weights": w if settings["weights"] else None,
            "cov": covariate_matrix(df, names), "cov_names": names, "alphabet": alphabet,
            "months": list(df.loc[:, "Sep.93":"Jun.99"].columns)}


def fit_one(edm_type, G, normalizer, data, settings, device):
    """Fit one model; returns (model, covariates used) or (None, None) if G is not allowed."""
    S, cov = data["S"], data["cov"]
    n_cov = 0 if cov is None else cov.shape[1]
    model, with_cov = build_model(edm_type, G, S.shape[1], len(data["alphabet"]), n_cov,
                                  {"normalizer": normalizer, "gated_noise": settings["gated_noise"],
                                   "mc_samples": settings["mc_samples"]})
    if model is None:
        return None, None
    cov = cov if with_cov else None
    model.fit(S, data["weights"], cov, n_init=settings["n_init"], epochs=settings["epochs"],
              lr=settings["lr"], seed=settings["seed"], device=device)
    return model, cov


def sweep(edm_type, g_values, data, settings, device, table_path):
    """Fit every (G, normaliser) pair and save the table after each fit.

    Returns the table and, per normaliser, its best model as (G, bic, model, covariates).
    """
    rows, best = [], {}
    for G in g_values:
        for normalizer in NORMALIZERS:
            start = time.perf_counter()
            model, cov = fit_one(edm_type, G, normalizer, data, settings, device)
            if model is None:
                print(f"{edm_type.name} G={G}: not a valid model, skipped")
                break
            seconds = time.perf_counter() - start
            bic = model.bic()
            rows.append({"type": edm_type.name, "G": G, "normalizer": normalizer,
                         "covariates": cov is not None, "bic": bic,
                         "loglik": model.log_likelihood_, "n_params": model.n_parameters(),
                         "seconds": round(seconds, 2), **settings})
            pd.DataFrame(rows).to_csv(table_path, index=False)
            print(f"{edm_type.name:>4}  G={G:<3d} {normalizer:<8} BIC={bic:12.2f}  "
                  f"loglik={model.log_likelihood_:11.2f}  ({seconds:.1f}s)", flush=True)
            if normalizer not in best or bic > best[normalizer][1]:
                best[normalizer] = (G, bic, model, cov)
    return pd.DataFrame(rows), best


def refit_best(table, data, settings, device):
    """Refit each normaliser's BIC-optimal G (same seed and settings, so the same model)."""
    edm_type = EDMType[table["type"].iloc[0]]
    best = {}
    for normalizer in NORMALIZERS:
        rows = table[table["normalizer"] == normalizer]
        if rows.empty:
            continue
        G = int(rows.loc[rows["bic"].idxmax(), "G"])
        print(f"refitting {edm_type.name} G={G} with the {normalizer} normaliser", flush=True)
        model, cov = fit_one(edm_type, G, normalizer, data, settings, device)
        best[normalizer] = (G, model.bic(), model, cov)
    return best


def summarize(table):
    """BIC by G (rows) and normaliser (columns), then each normaliser's best G."""
    wide = table.pivot(index="G", columns="normalizer", values="bic")
    wide = wide[[n for n in NORMALIZERS if n in wide.columns]]
    print("\nBIC (higher is better)")
    print(wide.round(1).to_string())
    print("\nbest G per normaliser")
    for name in wide.columns:
        g = wide[name].idxmax()
        print(f"  {name:<8} G={g:<3d} BIC={wide.loc[g, name]:.2f}")


# ---- cluster layout (Figure 7) ----------------------------------------------------------
def hamming_matrix(A, B):
    return (A[:, None, :] != B[None, :, :]).sum(-1)


def seriate(D, start):
    """Greedy nearest-neighbour path through distance matrix D, a cheap TSP seriation."""
    n = D.shape[0]
    order, free = [start], torch.ones(n, dtype=torch.bool)
    free[start] = False
    for _ in range(n - 1):
        nxt = int(D[order[-1]].masked_fill(~free, torch.iinfo(D.dtype).max).argmin())
        order.append(nxt)
        free[nxt] = False
    return order


def cluster_bands(model, S, cov, alphabet):
    """Row order and bands (name, first row, size, SPS label) for a Figure 7 style plot.

    Clusters are ordered by a seriation of their central sequences, renumbered 1, 2, ... in
    that order, with the noise component last. Within a cluster, rows start from the member
    closest to the central sequence and follow a nearest-neighbour chain.
    """
    labels = model.predict(S, cov).cpu()
    centres = model.central_sequences().cpu()
    D = hamming_matrix(centres, centres)
    groups = [(k, centres[k]) for k in seriate(D, int(D.sum(1).argmax()))]
    if model.noise_component is not None:
        groups.append((model.noise_component, None))
    order, bands = [], []
    for k, centre in groups:
        idx = (labels == k).nonzero().flatten()
        if len(idx) == 0:
            continue
        members = S[idx]
        start = 0 if centre is None else int((members != centre).sum(-1).argmin())
        idx = idx[seriate(hamming_matrix(members, members), start)]
        if centre is None:
            bands.append(("Noise", len(order), len(idx), None))
        else:
            bands.append((str(sum(b[3] is not None for b in bands) + 1), len(order), len(idx),
                          sps(centre, alphabet)))
        order.extend(idx.tolist())
    return order, bands


def cluster_table(S, order, bands, alphabet):
    """Cluster sizes and mean months in each state, the text version of the plot."""
    rows = []
    for name, first, size, label in bands:
        members = S[order[first:first + size]]
        months = F.one_hot(members, len(alphabet)).sum(1).double().mean(0)
        rows.append({"cluster": name, "SPS": label or "-", "n": size,
                     **{a: round(float(m), 1) for a, m in zip(alphabet, months)}})
    return pd.DataFrame(rows)


# ---- plots ------------------------------------------------------------------------------
def _spread(values, min_gap, heights=None):
    """Move label positions apart so neighbours are at least min_gap (times the mean of their
    heights, in lines) apart, keeping the mean position."""
    heights = [1] * len(values) if heights is None else heights
    order = sorted(range(len(values)), key=lambda i: values[i])
    placed = [values[i] for i in order]
    for k in range(1, len(placed)):
        gap = min_gap * (heights[order[k]] + heights[order[k - 1]]) / 2
        placed[k] = max(placed[k], placed[k - 1] + gap)
    shift = (sum(values) - sum(placed)) / len(placed)
    out = [0.0] * len(values)
    for k, i in enumerate(order):
        out[i] = placed[k] + shift
    return out


def _wrap_sps(label, per_line=3):
    """Fit an SPS label on at most two lines: break after per_line spells and, if longer,
    keep the first and last spells around an ellipsis (the printed table has it in full)."""
    spells = label.split("-")
    if len(spells) > 2 * per_line:
        spells = spells[:per_line] + ["…"] + spells[-(per_line - 1):]
    return "-\n".join("-".join(spells[i:i + per_line]) for i in range(0, len(spells), per_line))


def _leader(moved):
    return {"arrowstyle": "-", "color": INK_MUTED, "lw": 0.8, "shrinkA": 0,
            "shrinkB": 4} if moved else None


def plot_bic(table, path):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    edm_type = table["type"].iloc[0]
    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    ends = []
    for name, (color, marker) in STYLE.items():
        rows = table[table["normalizer"] == name].sort_values("G")
        if rows.empty:
            continue
        best = rows.loc[rows["bic"].idxmax()]
        ax.plot(rows["G"], rows["bic"], color=color, lw=2, solid_joinstyle="round",
                solid_capstyle="round", marker=marker, markersize=8, markerfacecolor=color,
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=3,
                label=f"{name}  (best G = {int(best['G'])}, BIC {best['bic']:,.0f})")
        ends.append((name, rows["G"].iloc[-1], rows["bic"].iloc[-1]))

    # Direct labels at the right end of each line. Labels that would collide are spread
    # apart and joined to their line end with a thin leader line.
    lo, hi = ax.get_ylim()
    label_y = _spread([y for _, _, y in ends], min_gap=0.05 * (hi - lo))
    x_label = table["G"].max() + 0.35
    for (name, x, y), y_text in zip(ends, label_y):
        ax.annotate(name, xy=(x, y), xytext=(x_label, y_text), va="center", ha="left",
                    fontsize=11, color=INK, annotation_clip=False,
                    arrowprops=_leader(abs(y_text - y) > 0.01 * (hi - lo)))

    g_values = sorted(table["G"].unique())
    ax.set_xticks(g_values)
    ax.set_xlim(g_values[0] - 0.4, g_values[-1] + 1.5)      # room for the end labels
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_xlabel("Number of components (G)", fontsize=11, color=INK_SECONDARY)
    ax.set_ylabel("BIC (higher is better)", fontsize=11, color=INK_SECONDARY)
    ax.set_title(f"{edm_type}: BIC by number of components, per normaliser",
                 fontsize=13, color=INK, loc="left", pad=12)
    ax.grid(axis="y", color=GRIDLINE, lw=1, ls="-")
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelcolor=INK_SECONDARY, length=0, pad=6)
    legend = ax.legend(loc="best", frameon=False, fontsize=10)
    for text in legend.get_texts():
        text.set_color(INK)

    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    print(f"\nfigure saved to {path}")


def plot_clusters(best, data, edm_type, path):
    """One Figure 7 style panel per normaliser, each at its BIC-optimal G."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    S, alphabet, months = data["S"], data["alphabet"], data["months"]
    n, T = S.shape
    cmap = ListedColormap([STATE_STYLE[a][1] for a in alphabet])
    ticks = [0] + list(range(9, T, 10))                    # Sep.93, Jun.94, ..., Jun.99
    fig, axes = plt.subplots(1, len(best), figsize=(6.6 * len(best), 10), squeeze=False,
                             layout="constrained", facecolor=SURFACE)

    for ax, (normalizer, (G, bic, model, cov)) in zip(axes[0], best.items()):
        order, bands = cluster_bands(model, S, cov, alphabet)
        print(f"\n{normalizer}: G = {G}, BIC {bic:.2f} (mean months in each state)")
        print(cluster_table(S, order, bands, alphabet).to_string(index=False))

        ax.imshow(S[order].numpy(), cmap=cmap, vmin=-0.5, vmax=len(alphabet) - 0.5,
                  aspect="auto", interpolation="nearest", extent=(0, T, n, 0))
        for _, first, _, _ in bands[1:]:
            ax.axhline(first, color=INK, lw=1.2)
        centres = [first + size / 2 for _, first, size, _ in bands]
        ax.set_yticks(centres, [name for name, _, _, _ in bands])
        ax.set_xticks([t + 0.5 for t in ticks], [months[t] for t in ticks])

        # SPS label of each central sequence on the right, spread apart where bands are thin.
        labelled = [(y, _wrap_sps(label)) for y, (_, _, _, label) in zip(centres, bands) if label]
        label_y = _spread([y for y, _ in labelled], min_gap=0.028 * n,
                          heights=[label.count("\n") + 1 for _, label in labelled])
        for (y, label), y_text in zip(labelled, label_y):
            ax.annotate(label, xy=(T, y), xytext=(T + 1.5, y_text), ha="left", va="center",
                        fontsize=8.5, color=INK, annotation_clip=False,
                        arrowprops=_leader(abs(y_text - y) > 1))

        ax.set_title(f"{normalizer}: G = {G}, BIC {bic:,.0f}", fontsize=12, color=INK,
                     loc="left", pad=10)
        ax.set_xlabel("Time", fontsize=11, color=INK_SECONDARY)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(colors=INK_MUTED, labelcolor=INK_SECONDARY, length=0, pad=4)
        ax.tick_params(axis="x", labelsize=9, labelrotation=45)
        for tick in ax.get_xticklabels():
            tick.set_ha("right")
            tick.set_rotation_mode("anchor")

    axes[0][0].set_ylabel("Clusters", fontsize=11, color=INK_SECONDARY)
    fig.suptitle(f"{edm_type}: clusters at each normaliser's BIC-optimal G "
                 f"(rows are the n = {n} observed sequences)", fontsize=13, color=INK)
    fig.legend(handles=[Patch(facecolor=STATE_STYLE[a][1], label=STATE_STYLE[a][0])
                        for a in alphabet],
               loc="outside lower center", ncol=len(alphabet), frameon=False, fontsize=10)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    print(f"\nfigure saved to {path}")


# ---- entry point ------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--type", default="UUN", type=str.upper, choices=[t.name for t in EDMType],
                   help="model type (default UUN)")
    p.add_argument("--g-min", type=int, default=5)
    p.add_argument("--g-max", type=int, default=15)
    p.add_argument("--csv", default="mvad.csv")
    p.add_argument("--out", default="results")
    p.add_argument("--covariates", default="all",
                   help="'all', 'none' or comma-separated, e.g. GCSE5eq")
    p.add_argument("--no-weights", action="store_true", help="ignore the sampling weights")
    p.add_argument("--gated-noise", action="store_true", help="GN gate instead of the default NGN")
    p.add_argument("--mc-samples", type=int, default=32,
                   help="sequences drawn per component per step for the sampled normaliser")
    p.add_argument("--n-init", type=int, default=3)
    p.add_argument("--epochs", type=int, default=600)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--plot-only", metavar="TABLE",
                   help="plot a saved table; the best models are refitted for the cluster plot")
    p.add_argument("--no-show", action="store_true", help="save the figures without opening them")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        table = pd.read_csv(args.plot_only)
        stem = Path(args.plot_only).stem
        settings = settings_from_table(table, args)
        data = load_data(args.csv, settings)
        best = refit_best(table, data, settings, args.device)
    else:
        settings = settings_from_args(args)
        data = load_data(args.csv, settings)
        edm_type = EDMType[args.type]
        print(f"{edm_type.name}: G = {args.g_min}..{args.g_max}, normalisers {list(NORMALIZERS)}, "
              f"covariates={data['cov_names'] or 'none'}, "
              f"weights={'yes' if settings['weights'] else 'no'}\n")
        stem = f"normalizers_{edm_type.name}"
        table, best = sweep(edm_type, range(args.g_min, args.g_max + 1), data, settings,
                            args.device, out / f"{stem}.csv")
        if table.empty:
            raise SystemExit("nothing was fitted; check --type and the G range")

    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summarize(table)
    plot_bic(table, out / f"{stem}.png")
    plot_clusters(best, data, table["type"].iloc[0], out / f"{stem}_clusters.png")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
