# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: covariability
#     language: python
#     name: covariability
# ---

import glob
from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Read output created by agreement_NN.py
DOM = 2
experiment = 2
agreement_files = np.array(
    sorted(
        glob.glob(
            f"../data/intermediate/agreement_results_ABI-IR_vs_ICON-DOM0{DOM}_exp{experiment}.pkl"
        )
    )
)
print(agreement_files)
dataframes = [None] * len(agreement_files)
for f, file in enumerate(agreement_files):
    dataframes[f] = pd.read_pickle(file)
df_all = pd.concat(dataframes)
df_all.head()

mask = (df_all.index >= "2020-01-10") & (df_all.index <= "2020-02-19")
df_all = df_all.loc[mask]
Sugar_df = df_all.Sugar.apply(pd.Series)
Gravel_df = df_all.Gravel.apply(pd.Series)
Flowers_df = df_all.Flowers.apply(pd.Series)
Fish_df = df_all.Fish.apply(pd.Series)
for df, p in zip(  # noqa: B905
    [Sugar_df, Gravel_df, Fish_df, Flowers_df], ["Sugar", "Gravel", "Fish", "Flowers"]
):
    rename_dict = {}
    for c in df.columns:
        rename_dict[c] = p + "_" + c
    df.rename(columns=rename_dict, inplace=True)

df_all = reduce(
    lambda left, right: pd.merge(
        left, right, left_index=True, right_index=True, how="outer"
    ),
    [Sugar_df, Gravel_df, Flowers_df, Fish_df],
)

df_daily_mean = df_all.resample("1D").mean()

df_no_high_clouds = pd.read_parquet("../data/result/no_high_clouds_DOM02.pq")

# +
high_clouds = False

fig, axs = plt.subplots(1, 4, sharey=True, sharex=True, figsize=(8, 4), dpi=150)
for p, pattern in enumerate(["Sugar", "Gravel", "Flowers", "Fish"]):
    if high_clouds:
        selection = df_daily_mean
    else:
        selection = df_daily_mean.loc[df_no_high_clouds.no_high_cloud]
    ranks_icon = selection[f"{pattern}_area_fraction_IIR"]
    ranks_abi = selection[f"{pattern}_area_fraction_AIR"]
    scatter = axs[p].scatter(
        pd.concat([ranks_icon, ranks_abi], axis=1).values.T[0],
        pd.concat([ranks_icon, ranks_abi], axis=1).values.T[1],
        label=pattern,
        color="grey",
        marker=".",
        clip_on=False,
        s=20,
    )
    axs[p].set_ylim(0, 1)
    axs[p].set_xlim(0, 1)
    axs[p].set_aspect(1)
    axs[p].text(0, 1, pattern)
    axs[p].set_xticks(np.arange(0, 1, 0.25), minor=True)
    axs[p].set_yticks(np.arange(0, 1, 0.25), minor=True)
    if p == 0:
        axs[p].set_ylabel(r"$A_{\mathrm{GOES16}}$")
    axs[p].set_xlabel(r"$A_{\mathrm{ICON-312m}}$")

sns.despine(offset=7)
plt.savefig("../figures/Pattern_area_fraction_scatter.pdf", bbox_inches="tight")
