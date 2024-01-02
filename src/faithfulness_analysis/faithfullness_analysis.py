from __future__ import annotations

from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm, wilcoxon

from src.text_classification.external.statutils import stats_main


def plothist(a):
    n, bins, patches = plt.hist(a, 50, density=True)
    mu = np.mean(a)
    sigma = np.std(a)
    plt.plot(bins, norm.pdf(bins, mu, sigma))
    plt.show()


def plotbox(data1, data2, x_values: list, data1_lab: str, data2_lab: str, title: str, y_lab: str, x_lab: str,
            path: Path | None = None):
    # Create a DataFrame for easier plotting
    df1 = pd.DataFrame(data1, columns=[f'k={x}' for x in x_values])
    df1["Model"] = data1_lab
    df2 = pd.DataFrame(data2, columns=[f'k={x}' for x in x_values])
    df2["Model"] = data2_lab

    # Concatenate the two DataFrames
    df = pd.concat([df1, df2])

    # Melt the DataFrame to have thresholds and arrays in separate columns
    df_melt = df.melt(id_vars="Model", var_name=x_lab, value_name=y_lab)

    # Create the boxplot
    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1.09)
    sns.set_style("whitegrid")
    ax = sns.boxplot(x=x_lab, y=y_lab, hue="Model", data=df_melt, orient="v", palette="Set2", notch=True,
                     showmeans=True,
                     meanprops=dict(marker="o", markerfacecolor="red", markeredgecolor="black", markersize="11"),
                     medianprops=dict(color="red", label="_median_", linewidth=2),
                     flierprops=dict(alpha=0.3))

    # Add text labels to the top of each boxplot
    for i in range(len(x_values)):
        ax.text(i - 0.2, df_melt[df_melt["Model"] == data1_lab][y_lab].max() + 0.03, data1_lab, color="#5ab4ac",
                ha="center", weight="bold", rotation=0)
        ax.text(i + 0.2, df_melt[df_melt["Model"] == data2_lab][y_lab].max() + 0.04, data2_lab, color="#fc8d59",
                ha="center", weight="bold", rotation=0)

    plt.title(title)
    plt.tight_layout()

    if path:
        plt.savefig(path.with_suffix(".png"), dpi=400)

    plt.show()


def wilcoxon_rank_biserial_correlation(w_pos: float, n: float, precision: int) -> float:
    """
    Compute the matched rank biserial correlation using Kerby strategy, stating that the correlation is equal to the
    difference between the proportion of favorable and unfavorable evidence: r = f – u.

    @param w_pos: W+ statistic for population
    @param n: number of measurements
    @param precision: floating point precision
    @return: the rank-biserial correlation, simple difference formula
    """
    # Calculate the rank-biserial correlation as the effect size
    # rbc = 1 - (2 * au) / (n * (n + 1))
    rbc = (4 * w_pos / (n * (n + 1))) - 1

    return round(rbc, precision)


def main(precision: int = 4):
    ks = [1, 5, 10, 20, 50, 75]
    ks_to_use = [0, 1, 2, 3, 4, 5]
    print(f"Using k values={[ks[i] for i in ks_to_use]}")

    comp_lm = np.load(f"dumps/faithfulness/comp_lm_{SUFF}.npy").astype(np.float64)[:, ks_to_use]
    suff_lm = np.load(f"dumps/faithfulness/suff_lm_{SUFF}.npy").astype(np.float64)[:, ks_to_use]

    comp_xg = np.load(f"dumps/faithfulness/comp_xg_{SUFF}.npy").astype(np.float64)[:, ks_to_use]
    suff_xg = np.load(f"dumps/faithfulness/suff_xg_{SUFF}.npy").astype(np.float64)[:, ks_to_use]

    assert comp_xg.shape == suff_xg.shape == comp_lm.shape == suff_lm.shape, "Shapes are not the same"
    n_columns = int(comp_lm.shape[1])

    arrays = dict(comp_lm=comp_lm, suff_lm=suff_lm, comp_xg=comp_xg, suff_xg=suff_xg)

    global_metrics = {k: (
        float(np.round(v.mean(), decimals=precision)),
        float(np.round(np.sqrt(v.var(axis=0).mean()), decimals=precision)))
        for k, v in arrays.items()}
    by_k_metrics = {
        k: {
            ks[ks_to_use[i]]: (
                float(np.round(v[:, i].mean(), decimals=precision)), float(np.round(v[:, i].std(), decimals=precision)))
            for
            i in range(n_columns)
        } for k, v in arrays.items()
    }
    pprint(global_metrics)
    pprint(by_k_metrics)

    out_path = Path("plots")
    out_path.mkdir(exist_ok=True, parents=True)
    title = "Comparison of {} values for different percentage of features"
    plotbox(comp_xg, comp_lm, [ks[i] for i in ks_to_use], "XG", "LM", title.format("COMP"),
            x_lab="% of removed top-features (k)", y_lab="COMP",
            path=out_path / "comp_boxplot")
    plotbox(suff_xg, suff_lm, [ks[i] for i in ks_to_use], "XG", "LM", title.format("SUFF"),
            x_lab="% of retained top-features (k)", y_lab="SUFF",
            path=out_path / "suff_boxplot")

    result = dict()
    stat_method = wilcoxon
    for i in range(n_columns):
        u_comp, p_comp, *_ = stat_method(comp_xg[:, i], comp_lm[:, i], alternative="greater", nan_policy="raise")
        u_suff, p_suff, *_ = stat_method(suff_xg[:, i], suff_lm[:, i], alternative="less", nan_policy="raise")

        result[f"comp_{ks[ks_to_use[i]]}"] = (
            float(np.round(u_comp, decimals=precision)), float(np.round(p_comp, decimals=precision)),
            wilcoxon_rank_biserial_correlation(u_comp, n=len(comp_xg[:, i]), precision=precision))
        result[f"suff_{ks[ks_to_use[i]]}"] = (
            float(np.round(u_suff, decimals=precision)), float(np.round(p_suff, decimals=precision)),
            wilcoxon_rank_biserial_correlation(u_suff, n=len(suff_xg[:, i]), precision=precision))

    pprint(result)

    # p_values = [ # CMSB
    #     [1.0, 4.6889071121511585e-34, 0.0, 0.0, 0.0, 0.0],
    #     [1.0, 1.0, 1.0, 0.040625840378956904, 0.0, 1.9999999999999996e-64]
    # ]
    p_values = [  # IMDB
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.8979067535544744, 1.945139891451969e-21, 0.0, 0.0]
    ]

    experiments = ["COMP", "SUFF"]
    data = {k: v for k, v in zip(experiments, p_values)}

    stats_main(data)


if __name__ == "__main__":
    SUFF = "CMS"  # "IMDB"
    main(3)
