"""Generate two publication-ready GoalPath-Bond EDA figures for Overleaf.

Run from repository root:
    python generate_overleaf_figures.py

By default, saves both PDFs in figures/ and commits/pushes only those PDFs
to origin/main. Use --local-only to generate without publishing.

Optional:
    python generate_overleaf_figures.py --data path/to/bonds.csv.zip --out figures --local-only
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # No GUI required; figures are saved directly.
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Support both the original repository layout and the reorganized code folder.
ROOT = Path(__file__).resolve().parent
if (ROOT / "02_Python_Code").is_dir():
    sys.path.insert(0, str(ROOT / "02_Python_Code"))
from process_investable_universe import load_data, DEFAULT_BONDS


def make_figure(series, filename):
    fig = plt.figure(figsize=(7.0, 2.85), layout="constrained")
    for i, (values, label, xlabel, trim) in enumerate(series):
        values = values[np.isfinite(values)]
        if values.empty:
            raise ValueError(f"No valid observations for {label}")

        median = values.median()
        # Trim only the displayed yield tails, not the dataset or median.
        if trim:
            lower, upper = values.quantile([0.005, 0.995])
            shown = values[values.between(lower, upper)]
        else:
            shown = values

        ax = fig.add_subplot(1, 2, i + 1)
        ax.hist(shown, bins=50,
                weights=np.full(len(shown), 100 / len(values)),
                color="#397db5", edgecolor="white", linewidth=0.32)
        ax.axvline(median, color="#222222", linestyle="--", linewidth=1,
                   label=f"Median: {median:.2f}")
        ax.set_title(f"({chr(97+i)}) {label}", loc="left", fontsize=10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Share of valid observations (%)")
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.grid(axis="y", alpha=0.2, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, fontsize=8)

        print(f"{label}: n={len(values):,}, plotted={len(shown):,}, "
              f"median={median:.4f}")

    fig.savefig(filename, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def publish_figures(files):
    """Publish only the generated PDFs; never push unrelated local commits."""
    def git(*args, capture=False):
        return subprocess.run(
            ["git", *args], cwd=ROOT, check=True, text=True,
            capture_output=capture
        )

    repo_root = Path(git("rev-parse", "--show-toplevel",
                         capture=True).stdout.strip()).resolve()
    paths = [str(p.resolve().relative_to(repo_root)) for p in files]

    # Do not push from another branch or include pre-existing local commits.
    branch = git("branch", "--show-current", capture=True).stdout.strip()
    if branch != "main":
        raise RuntimeError("Automatic publishing requires the main branch.")
    git("fetch", "origin", "main")
    head = git("rev-parse", "HEAD", capture=True).stdout.strip()
    remote = git("rev-parse", "FETCH_HEAD", capture=True).stdout.strip()
    if head != remote:
        raise RuntimeError(
            "Local main differs from origin/main. Sync your changes first; "
            "automatic publishing will not push unrelated commits."
        )

    # Explicit pathspecs prevent other working or staged files being committed.
    git("add", "--", *paths)
    changed = subprocess.run(
        ["git", "diff", "--cached", "--quiet", "--", *paths], cwd=ROOT
    )
    if changed.returncode == 0:
        print("Both PDFs are already up to date on GitHub.")
        return
    if changed.returncode != 1:
        raise RuntimeError("Could not check staged figure changes.")

    git("commit", "--only", "-m", "Update Overleaf bond distribution figures",
        "--", *paths)
    git("push", "origin", "main")
    print("Both PDFs committed and pushed to GitHub (origin/main).")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_BONDS)
    parser.add_argument("--out", type=Path, default=ROOT / "figures")
    parser.add_argument("--local-only", action="store_true",
                        help="Generate PDFs without committing or pushing.")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # Reuse the existing data loader, including its zipped-CSV support.
    _, df, _ = load_data(csv_path=args.data, usecols=[
        "cusip", "DATE", "MATURITY", "DURATION", "COUPON", "YIELD"
    ])
    df = df.loc[df["DATE"].between("2016-01-01", "2025-12-31")].copy()
    ttm = (df["MATURITY"] - df["DATE"]).dt.days / 365.25
    duration = df["DURATION"]
    coupon = df["COUPON"]  # Already expressed in percentage points.
    ytm = df["YIELD"] * 100  # Existing loader converts yields to decimals.

    make_figure([
        (duration[duration >= 0], "Modified duration", "Modified duration (years)", False),
        (ttm[ttm >= 0], "Remaining time to maturity", "Time to maturity (years)", False)
    ], args.out / "duration_maturity.pdf")

    make_figure([
        (coupon[coupon >= 0], "Annual coupon rate", "Coupon rate (%)", False),
        (ytm, "Yield to maturity", "Yield to maturity (%)", True)
    ], args.out / "coupon_yield.pdf")

    print(f"Figures saved in {args.out.resolve()}")
    files = [args.out / "duration_maturity.pdf", args.out / "coupon_yield.pdf"]
    if not args.local_only:
        try:
            publish_figures(files)
        except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
            print(f"GitHub upload failed: {exc}", file=sys.stderr)
            print("The PDFs are still saved locally. Check git status and sync "
                  "or push the figure commit manually.", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    plt.rcParams.update({
        "font.family": "serif", "font.size": 9, "pdf.fonttype": 42,
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8
    })
    main()
