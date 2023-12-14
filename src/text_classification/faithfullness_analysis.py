from __future__ import annotations

from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, norm, mannwhitneyu, wilcoxon
import seaborn as sns

from src.text_classification.external.statutils import stats_main


def plothist(a):
    n, bins, patches = plt.hist(a, 50, density=True)
    mu = np.mean(a)
    sigma = np.std(a)
    plt.plot(bins, norm.pdf(bins, mu, sigma))
    plt.show()


def plotbox(data1, data2, x_values: list, data1_lab: str, data2_lab: str, title: str, y_lab: str, x_lab: str, path: Path | None = None):
    # Create a DataFrame for easier plotting
    df1 = pd.DataFrame(data1, columns=[f'q={x}' for x in x_values])
    df1["Model"] = data1_lab
    df2 = pd.DataFrame(data2, columns=[f'q={x}' for x in x_values])
    df2["Model"] = data2_lab

    # Concatenate the two DataFrames
    df = pd.concat([df1, df2])

    # Melt the DataFrame to have thresholds and arrays in separate columns
    df_melt = df.melt(id_vars="Model", var_name=x_lab, value_name=y_lab)

    # Create the boxplot
    plt.figure(figsize=(10, 8))
    sns.boxplot(x=x_lab, y=y_lab, hue="Model", data=df_melt, orient="v", palette="Set2",
                showmeans=True,
                meanprops=dict(marker="o", markerfacecolor="red", markeredgecolor="black", markersize="12"))

    # Plot means separately
    # means = df_melt.groupby(["Model", x_lab])[y_lab].mean().reset_index()
    # sns.pointplot(x=x_lab, y=y_lab, hue="Model", data=means, orient="v",
    #               markers="o", linestyles="", dodge=True, palette="Set1")

    plt.title(title)

    if path:
        plt.savefig(path.with_suffix(".png"), dpi=400)

    plt.show()


def rank_biserial_correlation(a: np.ndarray, b: np.ndarray, au: float) -> float:
    """
    Compute the rank biserial correlation between two set of measurements

    @param a: measurements for population a
    @param b: measurements for population b
    @param au: u-test statistic for population
    @return: the rank-biserial correlation


    # TODO: fix for Wilcoxon SRT
    """
    # Calculate the rank-biserial correlation as the effect size
    n1 = len(a)
    n2 = len(b)
    rbc = 1 - (2 * au) / (n1 * n2)

    return rbc


def main(precision: int = 4):
    ks = [1, 5, 10, 20, 50, 75]
    ks_to_use = [0, 1, 2, 3, 4, 5]
    print(f"Using k values={[ks[i] for i in ks_to_use]}")

    comp_lm = np.load("experiments/faithfulness/comp.npy").astype(np.float64)[:, ks_to_use]
    suff_lm = np.load("experiments/faithfulness/suff.npy").astype(np.float64)[:, ks_to_use]

    comp_xg = np.load("dumps/faithfulness/comp_xg.npy").astype(np.float64)[:, ks_to_use]
    suff_xg = np.load("dumps/faithfulness/suff_xg.npy").astype(np.float64)[:, ks_to_use]

    assert comp_xg.shape == suff_xg.shape == comp_lm.shape == suff_lm.shape, "Shapes are not the same"
    n_columns = int(comp_lm.shape[1])

    arrays = dict(comp_lm=comp_lm, suff_lm=suff_lm, comp_xg=comp_xg, suff_xg=suff_xg)

    global_metrics = {k: (float(np.round(v.mean(), decimals=precision)), float(np.round(np.sqrt(v.var(axis=0).mean()), decimals=precision))) for k, v in arrays.items()}
    by_k_metrics = {
        k: {
            ks[ks_to_use[i]]: (float(np.round(v[:, i].mean(), decimals=precision)), float(np.round(v[:, i].std(), decimals=precision))) for i in range(n_columns)
        } for k, v in arrays.items()
    }
    pprint(global_metrics)
    pprint(by_k_metrics)

    out_path = Path("plots")
    out_path.mkdir(exist_ok=True, parents=True)
    title = "Comparison of {} values for different percentage of features"
    plotbox(comp_xg, comp_lm, [ks[i] for i in ks_to_use], "XGBoost", "LM", title.format("COMP"), x_lab="% of removed top-features (q)", y_lab="COMP",
            path=out_path / "comp_boxplot")
    plotbox(suff_xg, suff_lm, [ks[i] for i in ks_to_use], "XGBoost", "LM", title.format("SUFF"), x_lab="% of retained top-features (q)", y_lab="SUFF",
            path=out_path / "suff_boxplot")

    result = dict()
    stat_method = wilcoxon
    for i in range(n_columns):
        # plothist(suff_xg[:, i])
        # plothist(suff_lm[:, i])
        u_comp, p_comp, *_ = stat_method(comp_xg[:, i], comp_lm[:, i], alternative="greater", nan_policy="raise")
        u_suff, p_suff, *_ = stat_method(suff_xg[:, i], suff_lm[:, i], alternative="less", nan_policy="raise")
        # comp_r = ttest_ind(comp_xg[:, i], comp_lm[:, i], equal_var=False)
        # suff_r = ttest_ind(suff_xg[:, i], suff_lm[:, i], equal_var=False)

        result[f"comp_{ks[ks_to_use[i]]}"] = (float(np.round(u_comp, decimals=precision)), float(np.round(p_comp, decimals=precision)),
                                              rank_biserial_correlation(comp_xg[:, i], comp_lm[:, i], u_comp))
        result[f"suff_{ks[ks_to_use[i]]}"] = (float(np.round(u_suff, decimals=precision)), float(np.round(p_suff, decimals=precision)),
                                              rank_biserial_correlation(suff_xg[:, i], suff_lm[:, i], u_suff))

    pprint(result)

    # p_values = [
    #     [1.0, 5.778849948981599e-47, 0.0, 0.0, 0.0],
    #     [2.816086414877753e-21, 4.682418035094356e-37, 5.3704015999999995e-53, 0.0, 0.0]
    # ]
    #
    # experiments = ["COMP", "SUFF"]
    # data = {k: v for k, v in zip(experiments, p_values)}
    #
    # stats_main(data)


if __name__ == "__main__":
    main(30)
