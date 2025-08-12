from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List

import lmfit
import numpy as np
import plotting.pyplotdefs as pd

def runmap[T, U](
    run: List[Dict[str, Any]],
    key: str | Callable[[Dict[str, Any]], T],
    f: Callable[[List[T]], U],
) -> U:
    if isinstance(key, str):
        return f([d[key] for d in run])
    elif isinstance(key, Callable):
        return f([key(d) for d in run])
    else:
        raise ValueError()

def nqmap[T, U](
    data_nq: List[List[Dict[str, Any]]],
    key: str | Callable[[Dict[str, Any]], T],
    f: Callable[[List[T]], U],
) -> U:
    return [runmap(run, key, f) for run in data_nq]

datafiles = [
    Path("outdata_20250806-0327.json"),
    Path("outdata_20250806-1544.json"),
    Path("outdata_20250807-0733.json"),
    Path("outdata_20250808-1610.json"),
    Path("outdata_20250811-1522.json"),
]

def allplot() -> None:
    datafile = datafiles[-1]
    with datafile.open("r") as infile:
        indata = json.load(infile)

    mc: int = indata["mc"]
    nqubits: List[int] = indata["nqubits"]
    peaking: List[float] = indata["peaking"]
    data: List[List[List[Dict[str, Any]]]] = indata["data"]

    P = pd.Plotter.new(
        nrows=3, sharex=True, figsize=[3.375, 4], as_plotarray=True)
    for (i, (nq, data_nq)) in enumerate(zip(nqubits, data)):
        gen_time = nqmap(data_nq, "gen_time", np.mean)
        gen_time_std = nqmap(data_nq, "gen_time", np.std)
        solve_time = nqmap(data_nq, "solve_time", np.mean)
        solve_time_std = nqmap(data_nq, "solve_time", np.std)

        target_prob = nqmap(data_nq, "target_prob", np.mean)
        peak_prob_est = nqmap(data_nq, "peak_prob_est", np.mean)
        peak_prob_est_std = nqmap(data_nq, "peak_prob_est", np.std)
        peak_prob = nqmap(data_nq, "peak_prob", np.mean)
        peak_prob_std = nqmap(data_nq, "peak_prob", np.std)

        corr = nqmap(data_nq, "correct", lambda x: 1 - np.mean(x))
        corr_std = [np.sqrt(p * (1 - p) / mc) for p in corr]
        (
            P
            [0]
            .loglog(peaking, len(peaking) * [None])
            .errorbar(
                peaking, gen_time, gen_time_std,
                marker=".", ls="-", c=f"C{i % 10}",
                label=f"{nq} qubits",
            )
            .errorbar(
                peaking, solve_time, solve_time_std,
                marker=".", ls=":", c=f"C{i % 10}",
            )
            [1]
            .loglog(peaking, len(peaking) * [None])
            .plot(
                peaking, target_prob,
                marker=".", ls="-", c=f"C{i % 10}",
            )
            .errorbar(
                peaking, peak_prob_est, peak_prob_est_std,
                marker=".", ls="--", c=f"C{i % 10}",
            )
            .errorbar(
                peaking, peak_prob, peak_prob_std,
                marker=".", ls=":", c=f"C{i % 10}",
            )
            [2]
            .loglog(peaking, len(peaking) * [None])
            .errorbar(
                peaking, corr, corr_std,
                marker=".", ls="-", c=f"C{i % 10}",
            )
        )
    (
        P
        [0]
        .ggrid()
        .set_ylabel("Gen/solve\ntimes [s]")
        [1]
        .ggrid()
        .set_ylabel("Peaking\nprob.")
        [2]
        .ggrid()
        .set_ylabel("$1 - \\mathregular{Solvability}$")
        .set_xlabel("Peaking threshold wrt uniform")
        .set_xlim(0.85 * min(peaking), max(peaking) / 0.85)
        .savefig(datafile.with_suffix(".png"))
        .close()
    )

def ifindr[T](items: Iterable[T], pred: Callable[[T], bool]) -> Optional[int]:
    for (i, x) in enumerate(items):
        if pred(x):
            return i
    return None

def threshold_plot() -> None:
    infiles = datafiles[-2:]
    num_qubits: List[int] = list()
    min_peak_thresh: List[Optional[float]] = list()
    for infile in infiles:
        with infile.open("r") as infile:
            indata = json.load(infile)
        nqubits: List[int] = indata["nqubits"]
        peaking: List[float] = indata["peaking"]
        data: List[List[List[Dict[str, Any]]]] = indata["data"]

        for (nq, data_nq) in zip(nqubits, data):
            corr = nqmap(data_nq, "correct", lambda x: 1 - np.mean(x))
            if (
                (i0 := ifindr(corr[::-1], lambda c: c > 1e-9)) is not None
                and i0 != 0
            ):
                p = peaking[len(corr) - 1 - i0]
                min_peak_thresh.append(p)
            else:
                min_peak_thresh.append(None)
        num_qubits += nqubits

    def model(
        params: lmfit.Parameters,
        x: np.ndarray[float, 1],
    ) -> np.ndarray[float, 1]:
        a = params["a"].value
        b = params["b"].value
        return a * x + b

    def residuals(
        params: lmfit.Parameters,
        x: np.ndarray[float, 1],
        y: np.ndarray[float, 1],
    ) -> np.ndarray:
        m = model(params, x)
        return (m - y) ** 2

    (num_qubits, min_peak_thresh) = zip(*[
        (x, y) for (x, y) in zip(num_qubits, min_peak_thresh)
        if y is not None and x != 20
    ])

    num_qubits = np.array(num_qubits)
    min_peak_thresh = np.array(min_peak_thresh)

    params = lmfit.Parameters()
    params.add("a", value=0.25, min=0.0)
    params.add("b", value=0.0)
    fit = lmfit.minimize(
        residuals,
        params,
        args=(num_qubits, np.log10(min_peak_thresh)),
    )

    num_qubits_plot = np.linspace(num_qubits.min(), num_qubits.max(), 1000)
    min_peak_thresh_plot = 10 ** model(fit.params, num_qubits_plot)
    print(fit.params["a"].value)
    print(fit.params["b"].value)
    valstr = (
f"""
$10^{{a N + b}}$
$a = {fit.params["a"].value:g}$
$b = {fit.params["b"].value:g}$
"""[1:-1]
    )

    (
        pd.Plotter()
        .semilogy(
            num_qubits, min_peak_thresh,
            marker=".", ls="", color="k",
        )
        .semilogy(
            num_qubits_plot, min_peak_thresh_plot,
            marker="", ls="-", color="k",
        )
        .text_ax(
            0.05, 0.95, valstr,
            fontsize="xx-small",
            ha="left",
            va="top",
        )
        .ggrid()
        .set_xlabel("Qubits")
        .set_ylabel("Min. peaking threshold wrt uniform")
        .savefig("min_threshold.png")
        .close()
    )


def main() -> None:
    # allplot()
    threshold_plot()

if __name__ == "__main__":
    main()

