"""Reproduce Figure 3 of Murphy et al. (2021) on the MVAD data with the PyTorch EDM family.

    python main.py                                   # all 8 types, G = 1..25, all covariates
    python main.py --g-max 15 --n-init 2             # quicker run
    python main.py --types UUN,CCN --covariates GCSE5eq
    python main.py --plot-only results/bic_table.csv # re-plot a saved table without refitting
"""
from __future__ import annotations

import argparse
import time
from itertools import groupby
from pathlib import Path

import pandas as pd
import torch

from edm import NORMALIZERS, EDModel, EDMType

# CSV state labels -> the short codes used in the paper's figures
STATES = {
    "employment": "EM",
    "FE": "FE",
    "HE": "HE",
    "joblessness": "JL",
    "school": "SC",
    "training": "TR",
}

# Paper name -> CSV column. Grammar and Location define the sampling weights, so they are
# not used as gating covariates (Section 2).
COVARIATES = {
    "Catholic": "catholic",
    "FMPR": "fmpr",
    "Funemp": "funemp",
    "GCSE5eq": "gcse5eq",
    "Gender": "male",
    "Livboth": "livboth",
}

# Figure 3 styling: colour plus one or two overlaid markers (marker, filled), mimicking R's pch
STYLE = {
    "CC":  ("#555555", [("s", False), ("x", True)]),   # ⊠
    "UC":  ("#1E90FF", [("o", False), ("x", True)]),   # ⊗
    "CU":  ("#E31A1C", [("D", False), ("+", True)]),   # ◇ with +
    "UU":  ("#00B000", [("s", False), ("+", True)]),   # ⊞
    "CCN": ("#6A5ACD", [("^", True)]),                 # ▲
    "UCN": ("#8B1C62", [("o", True)]),                 # ●
    "CUN": ("#FFC800", [("v", True)]),                 # ▼
    "UUN": ("#FF69B4", [("s", True)]),                 # ■
}


# ---- data -------------------------------------------------------------------------------
def load_mvad(path):
    """Sequences (n, 70) as integer codes, raw sampling weights, the DataFrame, the alphabet."""
    df = pd.read_csv(path)
    months = df.loc[:, "Sep.93":"Jun.99"]              # drop Jul/Aug 1993 as in the paper
    alphabet = list(STATES.values())
    code = {label: alphabet.index(short) for label, short in STATES.items()}
    S = months.apply(lambda col: col.map(code))
    if S.isna().any().any():
        unknown = set(months.to_numpy().ravel()) - set(STATES)
        raise ValueError(f"unexpected state labels in {path}: {sorted(unknown)}")
    S = torch.tensor(S.to_numpy(dtype="int64"))
    w = torch.tensor(df["weight"].to_numpy(), dtype=torch.float32)
    return S, w, df, alphabet


def parse_covariates(spec: str) -> list[str]:
    if spec == "all":
        return list(COVARIATES)
    if spec == "none":
        return []
    names = [s.strip() for s in spec.split(",") if s.strip()]
    unknown = set(names) - set(COVARIATES)
    if unknown:
        raise SystemExit(f"unknown covariates {sorted(unknown)}; choose from {list(COVARIATES)}")
    return names


def covariate_matrix(df, names):
    """yes/no columns -> (n, r) float tensor of 0/1 dummies, or None."""
    if not names:
        return None
    cols = [COVARIATES[name] for name in names]
    return torch.tensor((df[cols] == "yes").to_numpy(), dtype=torch.float32)


def sps(seq, alphabet):
    """State-permanence-sequence label, e.g. (SC,25)-(HE,45)."""
    return "-".join(f"({alphabet[k]},{len(list(run))})" for k, run in groupby(seq.tolist()))


# ---- fitting ----------------------------------------------------------------------------
def is_redundant(edm_type: EDMType, G: int) -> bool:
    # UC/UU at G = 1 equal CC/CU; UCN/UUN at G = 2 equal CCN/CUN (Section 3.3)
    return (edm_type.name in ("UC", "UU") and G == 1) or (edm_type.name in ("UCN", "UUN") and G == 2)


def build_model(edm_type, G, T, v, n_cov, model_kwargs):
    """Covariates enter the gate 'where allowable' (Section 5.1); otherwise fit without them."""
    for r in ((n_cov, 0) if n_cov else (0,)):
        try:
            return EDModel(edm_type, G, T, v, r, **model_kwargs), r > 0
        except ValueError:          # e.g. covariates with G = 1, or CCN with G = 1
            continue
    return None, False


def fit_family(S, weights, cov, n_states, edm_types, g_values, *, model_kwargs, fit_kwargs, table_path):
    T = S.shape[1]
    n_cov = 0 if cov is None else cov.shape[1]
    rows, best = [], None
    for edm_type in edm_types:
        for G in g_values:
            if is_redundant(edm_type, G):
                continue
            model, with_cov = build_model(edm_type, G, T, n_states, n_cov, model_kwargs)
            if model is None:
                continue
            start = time.perf_counter()
            model.fit(S, weights, cov if with_cov else None, **fit_kwargs)
            bic, k = model.bic(), model.n_parameters()
            rows.append({"type": edm_type.name, "G": G, "covariates": with_cov,
                         "bic": bic, "loglik": model.log_likelihood_, "n_params": k})
            pd.DataFrame(rows).to_csv(table_path, index=False)      # save as we go
            print(f"{edm_type.name:>4}  G={G:<3d} BIC={bic:12.2f}  k={k:5d}  "
                  f"({time.perf_counter() - start:.1f}s)", flush=True)
            if best is None or bic > best[0]:
                best = (bic, model, with_cov)
    return pd.DataFrame(rows), best


def summarize(model, S, cov, alphabet, cov_names):
    print(f"\nbest by BIC: {model.edm_type.name}, G={model.model.G}, BIC={model.bic():.2f}")
    labels = model.predict(S, cov).cpu()
    for k, seq in enumerate(model.central_sequences().cpu()):
        print(f"  cluster {k + 1:2d}: n={int((labels == k).sum()):4d}  {sps(seq, alphabet)}")
    if model.noise_component is not None:
        print(f"  noise     : n={int((labels == model.noise_component).sum()):4d}")
    coef = model.gating_coefficients()
    if coef is not None:
        index = [f"cluster {k}" for k in range(2, len(coef) + 2)]
        if model.model.has_noise and not model.model.gating.ngn:
            index[-1] = "noise"
        table = pd.DataFrame(coef.cpu().numpy(), index=index, columns=["(Intercept)", *cov_names])
        print("\ngating coefficients (reference = cluster 1):")
        print(table.round(2).to_string())


# ---- Figure 3 ---------------------------------------------------------------------------
def plot_bic(table, path, g_min=5, show=False):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    table = table[table["G"] >= g_min]
    fig, ax = plt.subplots(figsize=(8, 6.5))
    handles, labels = [], []
    for name, (color, markers) in STYLE.items():
        rows = table[table["type"] == name].sort_values("G")
        if rows.empty:
            continue
        ax.plot(rows["G"], rows["bic"], color=color, lw=1)
        parts = []
        for marker, filled in markers:
            kw = dict(linestyle="none", marker=marker, color=color, markersize=7,
                      markeredgewidth=1.2, markerfacecolor=color if filled else "none")
            ax.plot(rows["G"], rows["bic"], **kw)
            parts.append(Line2D([], [], **kw))
        handles.append(tuple(parts))          # tuple -> markers drawn on top of each other
        labels.append(name)

    ax.set_xlabel("Number of Components (G)", fontsize=13)
    ax.set_ylabel("BIC", fontsize=13)
    ax.set_xticks(sorted(table["G"].unique()))
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.tick_params(axis="y", labelrotation=90)
    ax.legend(handles, labels, ncol=2, loc="upper right", fontsize=14,
              handletextpad=0.3, columnspacing=1.0, frameon=True, edgecolor="black")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    print(f"\nfigure saved to {path}")
    if show:
        plt.show()


# ---- entry point ------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default="mvad.csv")
    p.add_argument("--out", default="results")
    p.add_argument("--types", default="all", help="'all' or comma-separated, e.g. UUN,CCN")
    p.add_argument("--g-min", type=int, default=1)
    p.add_argument("--g-max", type=int, default=25)
    p.add_argument("--plot-g-min", type=int, default=5, help="Figure 3 hides G < 5")
    p.add_argument("--covariates", default="all", help="'all', 'none' or comma-separated, e.g. GCSE5eq")
    p.add_argument("--no-weights", action="store_true", help="ignore the sampling weights")
    p.add_argument("--gated-noise", action="store_true", help="GN gate instead of the default NGN")
    p.add_argument("--normalizer", choices=NORMALIZERS, default="hamming",
                   help="'sampled' estimates ∇log Z by Monte Carlo (Deep Learning, eq. 18.13)")
    p.add_argument("--mc-samples", type=int, default=32,
                   help="sequences drawn per component per step with --normalizer sampled")
    p.add_argument("--n-init", type=int, default=3)
    p.add_argument("--epochs", type=int, default=600)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--plot-only", metavar="TABLE", help="plot a saved bic_table.csv without fitting")
    p.add_argument("--show", action="store_true", help="also open the figure in a window")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        table = pd.read_csv(args.plot_only)
    else:
        S, w, df, alphabet = load_mvad(args.csv)
        cov_names = parse_covariates(args.covariates)
        cov = covariate_matrix(df, cov_names)
        if args.types == "all":
            types = list(EDMType)
        else:
            types = [EDMType[t.strip().upper()] for t in args.types.split(",")]
        print(f"n={S.shape[0]}, T={S.shape[1]}, states={alphabet}, covariates={cov_names or 'none'}, "
              f"weights={'no' if args.no_weights else 'yes'}\n")

        table, best = fit_family(
            S, None if args.no_weights else w, cov, len(alphabet), types,
            range(args.g_min, args.g_max + 1),
            model_kwargs={"gated_noise": args.gated_noise, "normalizer": args.normalizer,
                          "mc_samples": args.mc_samples},
            fit_kwargs=dict(epochs=args.epochs, lr=args.lr, n_init=args.n_init,
                            seed=args.seed, device=args.device),
            table_path=out / "bic_table.csv",
        )
        if best is None:
            raise SystemExit("nothing was fitted; check --types and the G range")
        _, model, with_cov = best
        summarize(model, S, cov if with_cov else None, alphabet, cov_names if with_cov else [])

    plot_bic(table, out / "figure3_bic.png", g_min=args.plot_g_min, show=args.show)


if __name__ == "__main__":
    main()