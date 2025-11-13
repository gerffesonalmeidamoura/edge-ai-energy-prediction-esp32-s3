# gerar_mae_rmse.py
# Procura training_history.json e artifacts (metrics/predictions) em locais comuns.
# Plota MAE por época; RMSE por época se existir no history; senão, RMSE overall.

import argparse, json, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"]  = 42

NAMES_METRICS = [
    "metrics_from_artifacts.json", "metrics_test.json", "metrics.json"
]
NAMES_PRED = [
    "predictions_from_artifacts.csv", "predicoes_test.csv",
    "predicoes.csv", "resultados.csv"
]
NAMES_HISTORY = ["training_history.json"]

def moving_average(x, k):
    x = np.asarray(x, float)
    if k and k > 1 and len(x) >= k:
        return np.convolve(x, np.ones(k)/k, mode="same")
    return x

def first_key(d, names):
    for n in names:
        if n in d: return n
    return None

def extract_unit(ylab):
    m = re.search(r"\((.*?)\)", ylab)
    return m.group(1) if m else ""

def nice_limits(y1, y2, pad=0.05, cap_q=0.995):
    yy = np.r_[np.asarray(y1,float), np.asarray(y2,float)]
    lo = float(np.nanmin(yy))
    hi = float(np.nanquantile(yy, cap_q))
    if hi <= lo: hi = lo + 1.0
    rng = hi - lo
    return lo - pad*rng, hi + pad*rng

def plot_pair(y_tr, y_va, title, ylab, outdir: Path, fname, smooth_val=3, force_best_epoch=None):
    y_tr = np.asarray(y_tr, float)
    y_va = np.asarray(y_va, float)
    n = int(min(len(y_tr), len(y_va)))
    x = np.arange(1, n+1)

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    va_s = moving_average(y_va[:n], smooth_val)

    ax.plot(x, y_tr[:n], label=f"Train {ylab}", lw=1.8)
    ax.plot(x, va_s,       label=f"Val {ylab}",   lw=1.8)

    if force_best_epoch is not None and 1 <= force_best_epoch <= n:
        be = int(force_best_epoch)
    else:
        be = int(np.argmin(y_va[:n])) + 1
    bv = float(y_va[be-1])

    ylo, yhi = nice_limits(y_tr[:n], y_va[:n], pad=0.05, cap_q=0.995)
    ax.set_ylim(ylo, yhi)
    ax.axvline(be, ls="--", lw=1.1, color="gray")

    unit = extract_unit(ylab)
    text = f"best val = {bv:.3f}" + (f" {unit}" if unit else "") + f" at epoch {be}"

    y_train_be = float(y_tr[be-1])
    y_val_be   = float(y_va[be-1])
    y_val_s_be = float(va_s[be-1])
    base = max(y_train_be, y_val_be, y_val_s_be)
    yr = (yhi - ylo)
    yt = min(yhi - 0.05*yr, base + 0.12*yr)

    x_off = max(5, int(0.08*n))
    if be/n > 0.75:
        xt, ha = max(1, be - x_off), "right"
    else:
        xt, ha = min(n, be + x_off), "left"

    ax.annotate(text, xy=(be, bv), xytext=(xt, yt),
                ha=ha, va="bottom",
                arrowprops=dict(arrowstyle="->", lw=1.0, color="gray"),
                fontsize=9)

    step = max(1, int(np.ceil(n/6)))
    ax.set_xticks(np.arange(1, n+1, step))
    ax.set_xlim(1, n*1.02)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylab)
    ax.grid(True, ls=":", lw=0.6)
    leg = ax.legend(); leg.set_frame_on(False)
    fig.tight_layout()
    outdir.mkdir(exist_ok=True, parents=True)
    fig.savefig(outdir/f"{fname}.png", dpi=600, bbox_inches="tight")
    fig.savefig(outdir/f"{fname}.pdf",             bbox_inches="tight")
    plt.close(fig)

def find_file(candidates, roots):
    for root in roots:
        for name in candidates:
            p = root / name
            if p.exists():
                return p
    return None

def main():
    ap = argparse.ArgumentParser(description="Gerar MAE e RMSE (curva ou overall)")
    ap.add_argument("--hist", type=str, default=None, help="caminho para training_history.json")
    ap.add_argument("--artdir", type=str, default=None, help="pasta com artifacts (metrics/predictions)")
    ap.add_argument("--outdir", type=str, default="plots_history", help="pasta de saída")
    args = ap.parse_args()

    CWD = Path.cwd()
    roots_hist = [CWD, CWD.parent, CWD / "modelo_final_quantizado", CWD.parent / "modelo_final_quantizado"]
    roots_art  = [CWD, CWD / "artifacts_eval", CWD.parent / "artifacts_eval", CWD.parent]

    if args.hist:
        hist_path = Path(args.hist)
    else:
        hist_path = find_file(NAMES_HISTORY, roots_hist)

    if args.artdir:
        artdir = Path(args.artdir)
        roots_art = [artdir, *roots_art]
    outdir = Path(args.outdir)

    # ---- MAE / RMSE por época se houver history ----
    have_history = False
    if hist_path and hist_path.exists():
        _raw = json.loads(hist_path.read_text(encoding="utf-8"))
        hist = _raw.get("history", _raw)
        have_history = True

        mae_key      = first_key(hist, ["mae", "mean_absolute_error"])
        val_mae_key  = first_key(hist, ["val_mae", "val_mean_absolute_error"])
        rmse_key     = first_key(hist, ["rmse", "root_mean_squared_error"])
        val_rmse_key = first_key(hist, ["val_rmse", "val_root_mean_squared_error"])
        mse_key      = first_key(hist, ["mse", "mean_squared_error"])
        val_mse_key  = first_key(hist, ["val_mse", "val_mean_squared_error"])

        if mae_key and val_mae_key:
            plot_pair(hist[mae_key], hist[val_mae_key],
                      "Training & Validation MAE (kWh)", "MAE (kWh)",
                      outdir, "training_validation_MAE_kwh", smooth_val=3)
        else:
            print("Aviso: history não contém séries MAE/val_mae; pulando gráfico de MAE por época.")

        rmse_curved = False
        if rmse_key and val_rmse_key:
            plot_pair(hist[rmse_key], hist[val_rmse_key],
                      "Training & Validation RMSE (kWh)", "RMSE (kWh)",
                      outdir, "training_validation_RMSE_kwh", smooth_val=3)
            rmse_curved = True
        elif mse_key and val_mse_key:
            y_tr = np.sqrt(np.asarray(hist[mse_key], float))
            y_va = np.sqrt(np.asarray(hist[val_mse_key], float))
            plot_pair(y_tr, y_va,
                      "Training & Validation RMSE (kWh)", "RMSE (kWh)",
                      outdir, "training_validation_RMSE_kwh", smooth_val=3)
            rmse_curved = True
    else:
        print("Aviso: não encontrei training_history.json em locais padrão.")

    # ---- Se NÃO houver curva de RMSE, tenta RMSE overall pelos artifacts ----
    have_rmse_curve = have_history and (
        ("rmse" in (_raw.get("history", _raw))) or ("mse" in (_raw.get("history", _raw)))
    )

    if not have_rmse_curve:
        # procura metrics json
        met_path = find_file(NAMES_METRICS, roots_art)
        rmse_overall = None

        if met_path:
            met = json.loads(met_path.read_text(encoding="utf-8"))
            if "rmse" in met:
                rmse_overall = float(met["rmse"])

        if rmse_overall is None:
            # tenta predictions csv e calcula
            pred_path = find_file(NAMES_PRED, roots_art)
            if pred_path:
                df = pd.read_csv(pred_path)
                y  = df.filter(regex="ground|truth", axis=1).iloc[:,0].to_numpy(float)
                yhat = df.filter(regex="pred", axis=1).iloc[:,0].to_numpy(float)
                rmse_overall = float(np.sqrt(np.mean((y - yhat)**2)))

        if rmse_overall is not None and np.isfinite(rmse_overall):
            fig, ax = plt.subplots(figsize=(3.5, 2.2))
            ax.axis("off")
            ax.text(0.5, 0.60, "Validation RMSE (overall)", ha="center", va="center", fontsize=12)
            ax.text(0.5, 0.40, f"{rmse_overall:.3f} kWh", ha="center", va="center",
                    fontsize=18, fontweight="bold")
            fig.tight_layout()
            outdir.mkdir(exist_ok=True, parents=True)
            fig.savefig(outdir/"training_validation_RMSE_overall.png", dpi=600, bbox_inches="tight")
            fig.savefig(outdir/"training_validation_RMSE_overall.pdf",             bbox_inches="tight")
            plt.close(fig)
            print("Observação: sem RMSE/MSE no history; gerei RMSE overall a partir dos artifacts.")
        else:
            print("Aviso: sem rmse/mse no history E sem artifacts (metrics/predictions) para RMSE overall.")

if __name__ == "__main__":
    main()
