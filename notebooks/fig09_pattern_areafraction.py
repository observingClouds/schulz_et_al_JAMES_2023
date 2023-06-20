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

# # Comparison of neural network classifications of ICON experiment 1 and 2

import logging
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.append("../src/helpers")
import agreement_helpers as h  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler("../logs/fig11.log"),
        logging.StreamHandler(),
    ],
)

# +
agreement_files = {
    "exp2_DOM1": "../data/intermediate/agreement_results_ABI-IR_vs_ICON-DOM01_exp2.pkl",
    "exp2_DOM2": "../data/intermediate/agreement_results_ABI-IR_vs_ICON-DOM02_exp2.pkl",
}


color_dict = {
    "Sugar": "#A1D791",
    "Fish": "#2281BB",
    "Gravel": "#3EAE47",
    "Flowers": "#93D2E2",
}
color_dict_dom = {
    "exp: 1, dom: 1": "deepskypblue",
    "exp: 2, dom: 1": "#109AFA",  # "royalblue",
    "exp: 1, dom: 2": "crimson",
    "exp: 2, dom: 2": "red",  # "crimson"
}
color_dict_dom = {
    "exp: 1, dom: 1": [0, 5, 1, 0],
    "exp: 2, dom: 1": [0, 5, 1, 0],
    "exp: 1, dom: 2": [0, 5, 1, 0],
    "exp: 2, dom: 2": [0, 5, 1, 0],
}


# -


def postprocess_agreement(arg_file_fmt, trange=("2020-01-11", "2020-02-19"), freq="1D"):
    df_all = h.read_pkls(arg_file_fmt)
    mask = (df_all.index >= trange[0]) & (df_all.index <= trange[1])
    df_all_masked = df_all.loc[mask]
    df_all = h.convert_df_dict_entries_to_columns(df_all_masked)
    df_mean = df_all.resample(freq).mean()

    df1 = df_mean.filter(
        items=[
            "Sugar_area_fraction_AIR",
            "Gravel_area_fraction_AIR",
            "Flowers_area_fraction_AIR",
            "Fish_area_fraction_AIR",
        ]
    ).rename(
        columns={
            "Sugar_area_fraction_AIR": "Sugar",
            "Gravel_area_fraction_AIR": "Gravel",
            "Flowers_area_fraction_AIR": "Flowers",
            "Fish_area_fraction_AIR": "Fish",
        }
    )

    df2 = df_mean.filter(
        items=[
            "Sugar_area_fraction_IIR",
            "Gravel_area_fraction_IIR",
            "Flowers_area_fraction_IIR",
            "Fish_area_fraction_IIR",
        ]
    ).rename(
        columns={
            "Sugar_area_fraction_IIR": "Sugar",
            "Gravel_area_fraction_IIR": "Gravel",
            "Flowers_area_fraction_IIR": "Flowers",
            "Fish_area_fraction_IIR": "Fish",
        }
    )
    return df1, df2


# Create an array with the colors you want to use
colors = [
    "#109AFA",
    "red",
    "lightskyblue",
    "tomato",
    "dodgerblue",
    "red",
]  # Set your custom color palette
customPalette = sns.set_palette(sns.color_palette(colors))

dfs = {}
for exp in [2]:
    for dom in [1, 2]:
        try:
            dfs[(exp, dom, "ABI")], dfs[(exp, dom, "ICON")] = postprocess_agreement(
                agreement_files[f"exp{exp}_DOM{dom}"]
            )
        except KeyError:
            print(f"No file format known for experiment {exp} and domain {dom}")
            continue


# +
translate_dict = {"exp1": "high CCN", "exp2": "control", "dom1": "624m", "dom2": "312m"}

# DATAFRAMES WITH TRIAL COLUMN ASSIGNED

diff = True

dfs_assgined = []
experiments = []
if diff:
    for DOM in [1, 2]:
        for exp in [2]:
            experiments.append(f"exp{exp}.dom{DOM}")
            df = (dfs[(exp, DOM, "ICON")] - dfs[(exp, DOM, "ABI")]).assign(
                expdom=f"ICON-{translate_dict['dom'+str(DOM)]}"
            )
            dfs_assgined.append(df)
else:
    for typ in ["ICON", "ABI"]:
        for exp in [1, 2]:
            for DOM in [1]:
                experiments.append(f"exp{exp}.dom{DOM}")
                df = dfs[(exp, DOM, typ)].assign(
                    expdom=(
                        f"exp: {translate_dict['exp'+str(exp)]}, dom: {DOM}, typ: {typ}"
                    )
                )
                dfs_assgined.append(df)

cdf = pd.concat(dfs_assgined)  # CONCATENATE
mdf = pd.melt(cdf, id_vars=["expdom"], var_name=["pattern"])  # MELT

logging.info(mdf.groupby(["expdom", "pattern"]).median())


ax = sns.boxplot(
    x="pattern", y="value", hue="expdom", data=mdf, palette=customPalette
)  # RUN PLOT
# ax = sns.boxplot(x="expdom", y="value", hue="pattern", data=mdf)  # RUN PLOT

if diff:
    plt.ylabel(r"difference to observations" + "\n" + r"too rare$\hspace{8}$too common")

sns.despine(bottom=True)
plt.legend(loc="upper right")  # ,bbox_to_anchor=(1,1))
plt.hlines(0, -0.8, 4, color="grey", linestyle=":")
plt.yticks([-1, 0, 1])
plt.tick_params(bottom=False)
plt.xlabel(None)
plt.savefig(
    f"../figures/Comparison_NN_mean_diff_to_obs_exp{'-'.join(experiments)}.pdf",
    bbox_inches="tight",
)
plt.show()

plt.clf()
plt.close()
# -
