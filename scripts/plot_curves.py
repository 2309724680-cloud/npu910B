"""Plot the four standard curves from a run's `*_curves.csv`.

    python scripts/plot_curves.py results/<run>_curves.csv -o results/charts

Reads the CSV only — it never recomputes metrics from the raw JSONL. That is
deliberate: charts and report tables then provably come from the same numbers,
so a discrepancy can't hide in a second implementation of the same formula.

Requires the charts extra:  pip install "llm-perfkit[charts]"
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

try:
    import matplotlib
except ImportError:
    sys.exit('matplotlib is required: pip install "llm-perfkit[charts]"')

matplotlib.use("Agg")  # headless: benchmark hosts rarely have a display
import matplotlib.pyplot as plt  # noqa: E402


def _num(row: dict, key: str) -> float | None:
    v = (row.get(key) or "").strip()
    if not v:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def load(path: str) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _by_parallelism(rows: list[dict], inp: float | None) -> list[dict]:
    """Rows of one input-length group, ordered by parallelism.

    Mixing length groups on one x-axis would put 1K and 128K points on the same
    line, where the drop reads as a scaling collapse rather than a length change.
    """
    out = [r for r in rows if _num(r, "input_tokens") == inp]
    return sorted(out, key=lambda r: _num(r, "concurrency") or 0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("csv", help="path to <run>_curves.csv")
    ap.add_argument("-o", "--out-dir", default="charts")
    ap.add_argument("--ttft-slo-ms", type=float, default=2000.0,
                    help="SLO line drawn on the TTFT chart")
    args = ap.parse_args()

    rows = load(args.csv)
    if not rows:
        sys.exit(f"no rows in {args.csv}")
    os.makedirs(args.out_dir, exist_ok=True)

    # The sweep group is the length with the most parallelism rungs.
    lengths = {_num(r, "input_tokens") for r in rows}
    sweep_len = max(lengths, key=lambda v: len(_by_parallelism(rows, v)))
    sweep = _by_parallelism(rows, sweep_len)
    par = [_num(r, "concurrency") for r in sweep]
    stem = os.path.basename(args.csv).replace("_curves.csv", "")
    made = []

    def save(fig, suffix: str) -> None:
        p = os.path.join(args.out_dir, f"{stem}_{suffix}.png")
        fig.tight_layout()
        fig.savefig(p, dpi=150)
        plt.close(fig)
        made.append(p)

    # C1 — overall output throughput vs parallelism. Flattening marks saturation.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(par, [_num(r, "output_tps") for r in sweep], "o-", color="#1f77b4")
    ax.set(xlabel="parallelism", ylabel="output TPS (tok/s)",
           title=f"C1 Output throughput vs parallelism (input={sweep_len:.0f})")
    ax.grid(alpha=.3)
    save(fig, "c1_output_tps")

    # C2 — Goodput with SLO compliance. Peak Goodput, not peak TPS, is the
    # operating point: past it, added load produces requests that miss the SLO.
    good = [_num(r, "goodput_rps") for r in sweep]
    if any(v is not None for v in good):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(par, good, "o-", color="#2ca02c", label="Goodput (req/s)")
        ax.set(xlabel="parallelism", ylabel="Goodput (req/s)", title="C2 Goodput vs parallelism")
        ax.grid(alpha=.3)
        ratio = [_num(r, "goodput_ratio") for r in sweep]
        comp = [v * 100 if v is not None else None for v in ratio]
        if any(v is not None for v in comp):
            ax2 = ax.twinx()
            ax2.plot(par, comp, "s--", color="#d62728", alpha=.7, label="SLO compliance (%)")
            ax2.set_ylabel("SLO compliance (%)")
            ax2.set_ylim(0, 105)
        peak = max(range(len(good)), key=lambda i: good[i] if good[i] is not None else -1)
        ax.annotate(f"peak p{par[peak]:.0f}\n{good[peak]:.3f} req/s",
                    xy=(par[peak], good[peak]), xytext=(10, -40),
                    textcoords="offset points",
                    arrowprops={"arrowstyle": "->", "color": "#555"})
        fig.legend(loc="lower center", ncol=2, frameon=False)
        save(fig, "c2_goodput")

    # C3 — TTFT percentiles. The mean hides the tail that the SLO is written against.
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, style in (("ttft_p50", "o-"), ("ttft_p95", "s-"), ("ttft_p99", "^-")):
        vals = [_num(r, key) for r in sweep]
        if any(v is not None for v in vals):
            ax.plot(par, vals, style, label=key.upper())
    ax.axhline(args.ttft_slo_ms, ls=":", color="#d62728",
               label=f"SLO {args.ttft_slo_ms:.0f} ms")
    ax.set(xlabel="parallelism", ylabel="TTFT (ms)", title="C3 TTFT percentiles vs parallelism")
    ax.grid(alpha=.3)
    ax.legend()
    save(fig, "c3_ttft_percentiles")

    # C4 — effect of input length at fixed parallelism. Prefill is roughly linear
    # in input length; decode is largely length-insensitive.
    groups: dict[float, list[dict]] = {}
    for r in rows:
        p = _num(r, "concurrency")
        if p is not None:
            groups.setdefault(p, []).append(r)
    cand = {p: sorted(rs, key=lambda r: _num(r, "input_tokens") or 0)
            for p, rs in groups.items()
            if len({_num(r, "input_tokens") for r in rs}) > 1}
    if cand:
        fig, ax = plt.subplots(figsize=(8, 5))
        for p, rs in sorted(cand.items()):
            ax.plot([_num(r, "input_tokens") for r in rs],
                    [_num(r, "ttft_p50") for r in rs], "o-", label=f"p{p:.0f}")
        ax.set(xlabel="input tokens (configured)", ylabel="TTFT P50 (ms)",
               title="C4 TTFT vs input length")
        ax.set_xscale("log", base=2)
        ax.grid(alpha=.3, which="both")
        ax.legend()
        save(fig, "c4_ttft_vs_length")

    for p in made:
        print(p)


if __name__ == "__main__":
    main()
