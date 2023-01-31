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

import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pint_xarray
import seaborn as sns
import tqdm
import xarray as xr
from intake import open_catalog
from metpy.calc import potential_temperature, precipitable_water, wind_speed
from metpy.units import units

sys.path.append("../src/helpers")
import cluster_helpers as ch  # noqa: E402

# -
if __name__ == "__main__":  # noqa: C901
    conf_dict = {
        "SRM": {"label": "ICON-SRM", "color": "grey"},
        "DOM01": {"label": "ICON 624m", "color": "#109AFA"},
        "DOM02": {"label": "ICON 312m", "color": "red"},
        "DOM03": {"label": "ICON 156m", "color": "brown"},
        "obs": {"label": "GOES-16 ABI", "color": "black"},
    }

    client = ch.setup_cluster("local cluster", logging.ERROR)

    client

    cat = open_catalog(
        "https://raw.githubusercontent.com/eurec4a/eurec4a-intake/master/catalog.yml"
    )

    # +
    station = "BCO"
    ds_SRM = cat.simulations.ICON.SRM[f"meteogram_{station}"].to_dask()
    ds_LES_ctrl_DOM01 = cat.simulations.ICON.LES_CampaignDomain_control[
        f"meteogram_{station}_DOM01"
    ].to_dask()
    ds_LES_ctrl_DOM02 = cat.simulations.ICON.LES_CampaignDomain_control[
        f"meteogram_{station}_DOM02"
    ].to_dask()

    ds_SRM_synradar = cat.simulations.ICON.SRM[f"synthetic_radar_{station}"].to_dask()
    ds_LES_ctrl_synradar_DOM01 = (
        cat.simulations.ICON.LES_CampaignDomain_control.synthetic_radar_BCO_DOM01.to_dask()
    )
    ds_LES_ctrl_synradar_DOM02 = (
        cat.simulations.ICON.LES_CampaignDomain_control.synthetic_radar_BCO_DOM02.to_dask()
    )
    # -

    ds_SRM["height"] = ds_SRM.height.pint.quantify().pint.to("km").pint.dequantify()
    ds_SRM["height_2"] = ds_SRM.height_2.pint.quantify().pint.to("km").pint.dequantify()
    ds_LES_ctrl_DOM01["height"] = (
        ds_LES_ctrl_DOM01.height.pint.quantify().pint.to("km").pint.dequantify()
    )
    ds_LES_ctrl_DOM02["height"] = (
        ds_LES_ctrl_DOM02.height.pint.quantify().pint.to("km").pint.dequantify()
    )
    ds_LES_ctrl_DOM01["height_2"] = (
        ds_LES_ctrl_DOM01.height_2.pint.quantify().pint.to("km").pint.dequantify()
    )
    ds_LES_ctrl_DOM02["height_2"] = (
        ds_LES_ctrl_DOM02.height_2.pint.quantify().pint.to("km").pint.dequantify()
    )
    ds_SRM_synradar["height"] = (
        ds_SRM_synradar.height.pint.quantify().pint.to("km").pint.dequantify()
    )
    ds_LES_ctrl_synradar_DOM01["height"] = (
        ds_LES_ctrl_synradar_DOM01.height.pint.quantify().pint.to("km").pint.dequantify()
    )
    ds_LES_ctrl_synradar_DOM02["height"] = (
        ds_LES_ctrl_synradar_DOM02.height.pint.quantify().pint.to("km").pint.dequantify()
    )

    # ## Show differences depending on pattern influencing factors like u10m, t2m

    def calc_LTS(ds_snd, var_time="sounding"):
        LTS_obs = {}
        p700_idx = np.nanargmin(np.abs(ds_snd.p - 70000).values, axis=1)
        p1000_idx = np.nanargmin(np.abs(ds_snd.p - 100000).values, axis=1)

        for t, time in enumerate(ds_snd[var_time]):
            theta700 = ds_snd.sel({var_time: time}).theta[p700_idx[t]]
            theta1000 = ds_snd.sel({var_time: time}).theta[p1000_idx[t]]
            LTS_obs[t] = theta700 - theta1000
        return LTS_obs

    # ### Observations

    ds_snd = cat.radiosondes.bco.to_dask()
    ds_snd = ds_snd.load()

    L = calc_LTS(ds_snd)

    ds_snd["LTS"] = xr.concat(L.values(), dim="sounding")
    ds_snd.LTS.attrs["units"] = "K"

    daily_mean = (
        ds_snd.where(ds_snd.ascent_flag, drop=True)
        .groupby(ds_snd.where(ds_snd.ascent_flag, drop=True).launch_time.dt.date)
        .mean(keep_attrs=True)
    )  # do not use the wdir!!!

    ds_radar = cat.barbados.bco.radar_reflectivity.to_dask()
    ds_sfc_met = cat.barbados.bco.meteorology.to_dask()

    ds_sfc_met["T"] = ds_sfc_met.T.pint.quantify().pint.to("K").pint.dequantify()
    ds_sfc_met["T"].attrs["units"] = "K"  # instead of 'kelvin'

    ds_radar_10s = ds_radar.resample(time="10s").nearest()

    # + tags=[]
    ds_sfc_met = ds_sfc_met.load()
    # -

    common_times = sorted(
        set(ds_sfc_met.time.values).intersection(set(ds_radar_10s.time.values))
    )
    common_times_snd = sorted(
        set(
            ds_snd.where(ds_snd.ascent_flag, drop=True).launch_time.dt.date.values
        ).intersection(set(ds_radar_10s.time.dt.date.values))
    )

    # + tags=[]
    ds_radar_common = ds_radar_10s.sel(time=common_times)
    # -

    ds_radar_common["range"] = (
        ds_radar_common.range.pint.quantify().pint.to("km").pint.dequantify()
    )

    ds_sfc_met_common = ds_sfc_met.sel(time=common_times)

    ds_radar_common_sounding = ds_radar_common.sel(
        time=np.isin(ds_radar_common.time.dt.date, common_times_snd), drop=True
    )

    # ## OBS vs SIM

    logging.info(np.unique(ds_radar_common.time.dt.date))

    # ### Prepare input

    # #### PW

    # +
    pw = np.empty(len(ds_snd.launch_time))
    pw.fill(np.nan)
    for s, _ in enumerate(tqdm.tqdm(ds_snd.sounding)):
        # Only include soundings that went through the major part of the troposphere
        # Somehow the bottom and top arguments are not working here
        if (
            ds_snd.isel(sounding=s).p.min() <= 30000
            and ds_snd.isel(sounding=s).p.max() > 90000
        ):
            dp = ds_snd.isel(sounding=s).dp * units.K
            try:
                pw[s] = precipitable_water(
                    ds_snd.isel(sounding=s).p * units.Pa,
                    dp,
                ).magnitude  # in mm
            except ValueError:
                pass

    da_pw = xr.DataArray(pw, dims=["time"], coords={"time": ds_snd.launch_time.values})
    da_pw = da_pw.groupby(da_pw.time.dt.date).mean()
    da_pw.attrs["units"] = "mm"
    # -

    datasets = {1: ds_LES_ctrl_DOM01, 2: ds_LES_ctrl_DOM02}
    for dom in [1, 2]:
        fn = f"Meteogram_BCO_LTS_ICON-DOM0{dom}.nc"
        if os.path.exists(fn):
            ds_LTS = xr.open_dataset(fn)
        else:
            p700_idx = (
                np.abs((datasets[dom].P - 70000))
                .argmin(dim="height_2", skipna=True)
                .compute()
            )
            p1000_idx = (
                np.abs((datasets[dom].P - 100000))
                .argmin(dim="height_2", skipna=True)
                .compute()
            )

            times = datasets[dom].time
            LTS_daily = {}
            for time in tqdm.tqdm(times):
                T700 = (
                    datasets[dom].T.sel(time=time)[p700_idx.sel(time=time)]
                    * units.kelvin
                )
                T1000 = (
                    datasets[dom].T.sel(time=time)[p1000_idx.sel(time=time)]
                    * units.kelvin
                )
                theta700 = potential_temperature(
                    700.0 * units.mbar, T700
                ).metpy.dequantify()
                theta1000 = potential_temperature(
                    1000.0 * units.mbar, T1000
                ).metpy.dequantify()

                LTS_daily[time.values.astype("<M8[s]")] = float(theta700 - theta1000)
            df = pd.DataFrame.from_dict(LTS_daily, orient="index", columns=["LTS"])
            ds_LTS = xr.Dataset.from_dataframe(df)
            ds_LTS = ds_LTS.rename({"index": "time"})
            ds_LTS.LTS.attrs["units"] = "K"
            ds_LTS.to_netcdf(f"Meteogram_BCO_LTS_ICON-DOM0{dom}.nc")
        ds_LTS.LTS.attrs["units"] = "K"
        datasets[dom]["LTS"] = ds_LTS.LTS

    # ### Plot results

    # +
    var_sim = "Ze"
    var_obs = "Zf"
    Z_threshold = -50
    hgt_max = 4
    percentiles = [0.25, 0.50, 0.75]
    hue_vars = {
        "v": {"sim": "VEL", "obs": "VEL"},
        "T": {"sim": "T2M", "obs": "T"},
        "PW": {"sim": "TQV_DIA", "obs": "PW"},
        "LTS": {"sim": "LTS", "obs": "LTS"},
    }

    for hue_var, d in hue_vars.items():
        hue_var_sim = d["sim"]
        hue_var_obs = d["obs"]
        print(hue_var)
        if hue_var_sim == "VEL":
            u_sim = ds_LES_ctrl_DOM02["U10M"].values * units("m/s")
            v_sim = ds_LES_ctrl_DOM02["V10M"].values * units("m/s")
            d_sim_DOM02 = xr.DataArray(
                wind_speed(u_sim, v_sim).magnitude,
                dims=["time"],
                coords={"time": ds_LES_ctrl_DOM02.time},
            )
            q_sim_DOM02 = d_sim_DOM02.quantile(
                percentiles, method="nearest", skipna=True
            ).compute()

            u_sim = ds_LES_ctrl_DOM01["U10M"].values * units("m/s")
            v_sim = ds_LES_ctrl_DOM01["V10M"].values * units("m/s")
            d_sim_DOM01 = xr.DataArray(
                wind_speed(u_sim, v_sim).magnitude,
                dims=["time"],
                coords={"time": ds_LES_ctrl_DOM01.time},
            )
            q_sim_DOM01 = d_sim_DOM01.quantile(
                percentiles, method="nearest", skipna=True
            ).compute()
        elif hue_var_sim == "LTS":
            d_sim_DOM01 = (
                ds_LES_ctrl_DOM01[hue_var_sim]
                .groupby(ds_LES_ctrl_DOM01.time.dt.date)
                .mean()
            )
            d_sim_DOM02 = (
                ds_LES_ctrl_DOM02[hue_var_sim]
                .groupby(ds_LES_ctrl_DOM02.time.dt.date)
                .mean()
            )
            q_sim_DOM01 = (
                d_sim_DOM01.chunk(date=-1)
                .quantile(percentiles, method="nearest", skipna=True)
                .compute()
            )
            q_sim_DOM02 = (
                d_sim_DOM02.chunk(date=-1)
                .quantile(percentiles, method="nearest", skipna=True)
                .compute()
            )
        else:
            d_sim_DOM01 = ds_LES_ctrl_DOM01[hue_var_sim]
            q_sim_DOM01 = (
                d_sim_DOM01.chunk(time=-1)
                .quantile(percentiles, method="nearest", skipna=True)
                .compute()
            )
            d_sim_DOM02 = ds_LES_ctrl_DOM02[hue_var_sim]
            q_sim_DOM02 = (
                d_sim_DOM02.chunk(time=-1)
                .quantile(percentiles, method="nearest", skipna=True)
                .compute()
            )
        logging.info(f"Simulation quantiles: {q_sim_DOM01.values, q_sim_DOM02.values}")

        if hue_var_obs is not None:
            if hue_var_obs == "LTS":
                d_obs = daily_mean[hue_var_obs]
                q_obs = (
                    d_obs.chunk(date=-1)
                    .quantile(percentiles, method="nearest", skipna=True)
                    .compute()
                )
            elif hue_var_obs == "PW":
                d_obs = da_pw
                q_obs = (
                    d_obs.chunk(date=-1)
                    .quantile(percentiles, method="nearest", skipna=True)
                    .compute()
                )
            else:
                d_obs = ds_sfc_met_common[hue_var_obs]
                q_obs = (
                    d_obs.chunk(time=-1)
                    .quantile(percentiles, method="nearest", skipna=True)
                    .compute()
                )
            logging.info(f"Observation quantiles: {q_obs.values}")

        fig, axs = plt.subplots(
            2, 1, figsize=(2, 4.5), gridspec_kw={"height_ratios": [4, 1]}
        )
        plt.title(hue_var)
        # SIM
        if hue_var_sim == "LTS":
            (ds_LES_ctrl_synradar_DOM01[var_sim] > Z_threshold).groupby(
                ds_LES_ctrl_synradar_DOM01.time.dt.date
            ).mean().sel(
                height=slice(0, hgt_max),
                date=np.where(d_sim_DOM01 < q_sim_DOM01[0], True, False),
            ).mean(
                dim="date"
            ).plot(
                ax=axs.flatten()[0],
                y="height",
                color=conf_dict["DOM01"]["color"],
                linestyle=":",
            )
            (ds_LES_ctrl_synradar_DOM01[var_sim] > Z_threshold).groupby(
                ds_LES_ctrl_synradar_DOM01.time.dt.date
            ).mean().sel(
                height=slice(0, hgt_max),
                date=np.where(d_sim_DOM01 > q_sim_DOM01[2], True, False),
            ).mean(
                dim="date"
            ).plot(
                ax=axs.flatten()[0],
                y="height",
                color=conf_dict["DOM01"]["color"],
                linestyle="--",
            )
            (ds_LES_ctrl_synradar_DOM02[var_sim] > Z_threshold).groupby(
                ds_LES_ctrl_synradar_DOM02.time.dt.date
            ).mean().sel(
                height=slice(0, hgt_max),
                date=np.where(d_sim_DOM02 < q_sim_DOM02[0], True, False),
            ).mean(
                dim="date"
            ).plot(
                ax=axs.flatten()[0],
                y="height",
                color=conf_dict["DOM02"]["color"],
                linestyle=":",
            )
            (ds_LES_ctrl_synradar_DOM02[var_sim] > Z_threshold).groupby(
                ds_LES_ctrl_synradar_DOM02.time.dt.date
            ).mean().sel(
                height=slice(0, hgt_max),
                date=np.where(d_sim_DOM02 > q_sim_DOM02[2], True, False),
            ).mean(
                dim="date"
            ).plot(
                ax=axs.flatten()[0],
                y="height",
                color=conf_dict["DOM02"]["color"],
                linestyle="--",
            )
        else:
            try:
                (ds_LES_ctrl_synradar_DOM01[var_sim] > Z_threshold).sel(
                    height=slice(0, hgt_max),
                    time=np.where(d_sim_DOM01 < q_sim_DOM01[0], True, False)[1:],
                ).mean(dim="time").plot(
                    ax=axs.flatten()[0],
                    y="height",
                    color=conf_dict["DOM01"]["color"],
                    linestyle=":",
                )
                (ds_LES_ctrl_synradar_DOM01[var_sim] > Z_threshold).sel(
                    height=slice(0, hgt_max),
                    time=np.where(d_sim_DOM01 > q_sim_DOM01[2], True, False)[1:],
                ).mean(dim="time").plot(
                    ax=axs.flatten()[0],
                    y="height",
                    color=conf_dict["DOM01"]["color"],
                    linestyle="--",
                )
                (ds_LES_ctrl_synradar_DOM02[var_sim] > Z_threshold).sel(
                    height=slice(0, hgt_max),
                    time=np.where(d_sim_DOM02 < q_sim_DOM02[0], True, False)[1:],
                ).mean(dim="time").plot(
                    ax=axs.flatten()[0],
                    y="height",
                    color=conf_dict["DOM02"]["color"],
                    linestyle=":",
                )
                (ds_LES_ctrl_synradar_DOM02[var_sim] > Z_threshold).sel(
                    height=slice(0, hgt_max),
                    time=np.where(d_sim_DOM02 > q_sim_DOM02[2], True, False)[1:],
                ).mean(dim="time").plot(
                    ax=axs.flatten()[0],
                    y="height",
                    color=conf_dict["DOM02"]["color"],
                    linestyle="--",
                )
            except KeyError:
                pass
        # OBS
        factor = -1
        if hue_var_obs is not None:
            if hue_var_obs == "LTS":
                ds_radar_common_sel = (
                    ds_radar_common_sounding[var_obs] > Z_threshold
                ).sel(
                    range=slice(0, hgt_max),
                    time=slice(daily_mean.date.min(), daily_mean.date.max()),
                )
                (
                    ds_radar_common_sel.groupby(ds_radar_common_sel.time.dt.date)
                    .mean()
                    .sel(date=np.where(d_obs < q_obs[0], True, False))
                    .mean(dim="date")
                    * factor
                ).plot(
                    ax=axs.flatten()[0],
                    label="25%",
                    y="range",
                    color=conf_dict["obs"]["color"],
                    linestyle=":",
                )
                (
                    ds_radar_common_sel.groupby(ds_radar_common_sel.time.dt.date)
                    .mean()
                    .sel(date=np.where(d_obs > q_obs[2], True, False))
                    .mean(dim="date")
                    * factor
                ).plot(
                    ax=axs.flatten()[0],
                    label="75%",
                    y="range",
                    color=conf_dict["obs"]["color"],
                    linestyle="--",
                )
            elif hue_var_obs == "PW":
                ds_radar_common_sel = (ds_radar_common[var_obs] > Z_threshold).sel(
                    range=slice(0, hgt_max),
                    time=slice(da_pw.date.min(), da_pw.date.max()),
                )
                (
                    ds_radar_common_sel.groupby(ds_radar_common_sel.time.dt.date)
                    .mean()
                    .sel(date=np.where(d_obs < q_obs[0], True, False))
                    .mean(dim="date")
                    * factor
                ).plot(
                    ax=axs.flatten()[0],
                    label="25%",
                    y="range",
                    color=conf_dict["obs"]["color"],
                    linestyle=":",
                )
                (
                    ds_radar_common_sel.groupby(ds_radar_common_sel.time.dt.date)
                    .mean()
                    .sel(date=np.where(d_obs > q_obs[2], True, False))
                    .mean(dim="date")
                    * factor
                ).plot(
                    ax=axs.flatten()[0],
                    label="75%",
                    y="range",
                    color=conf_dict["obs"]["color"],
                    linestyle="--",
                )
            else:
                (
                    (ds_radar_common[var_obs] > Z_threshold)
                    .sel(range=slice(0, hgt_max))
                    .sel(time=np.where(d_obs < q_obs[0], True, False))
                    .mean(dim="time")
                    * factor
                ).plot(
                    ax=axs.flatten()[0],
                    label="25%",
                    y="range",
                    color=conf_dict["obs"]["color"],
                    linestyle=":",
                )
                (
                    (ds_radar_common[var_obs] > Z_threshold)
                    .sel(range=slice(0, hgt_max))
                    .sel(time=np.where(d_obs > q_obs[2], True, False))
                    .mean(dim="time")
                    * factor
                ).plot(
                    ax=axs.flatten()[0],
                    label="75%",
                    y="range",
                    color=conf_dict["obs"]["color"],
                    linestyle="--",
                )
        axs.flatten()[0].set_xlabel("echo fraction")
        axs.flatten()[0].set_ylabel("altitude / km")
        axs.flatten()[0].vlines(0, 0, hgt_max, color="grey", linewidth=0.1)
        axs.flatten()[0].set_xlim(-0.5, 0.5)
        axs.flatten()[0].set_xticklabels([0.5, 0, 0.5])
        axs.flatten()[0].set_ylim(0, hgt_max)
        axs.flatten()[0].set_yticks([0, 0.8, 2, 3], which="major")

        # Plot category data
        axs.flatten()[1].plot(
            q_sim_DOM01.values,
            [1, 1, 1],
            color=conf_dict["DOM01"]["color"],
            linestyle="-",
            marker=".",
        )
        axs.flatten()[1].plot(
            q_sim_DOM02.values,
            [0, 0, 0],
            color=conf_dict["DOM02"]["color"],
            linestyle="-",
            marker=".",
        )
        axs.flatten()[1].plot(
            q_obs.values,
            [2, 2, 2],
            color=conf_dict["obs"]["color"],
            linestyle="-",
            marker=".",
        )
        axs.flatten()[1].set_title(None)
        axs.flatten()[1].set_xlabel(f"{hue_var} / {d_obs.attrs['units']}")

        plt.tight_layout()
        sns.despine(offset=10)
        axs.flatten()[1].yaxis.set_visible(False)
        axs.flatten()[1].set_xticks(
            [
                np.round(
                    np.min([q_sim_DOM02.values, q_sim_DOM01.values, q_obs.values]), 2
                ),
                np.round(
                    np.max([q_sim_DOM02.values, q_sim_DOM01.values, q_obs.values]), 2
                ),
            ]
        )
        axs.flatten()[1].set_ylim(-0.2, 4)
        axs.flatten()[1].spines["left"].set_visible(False)
        axs.flatten()[0].legend(loc=(1.1, 0))
        plt.savefig(
            f"../figures/Cloudfraction_dependency_on_{hue_var}_comparison.pdf",
            bbox_inches="tight",
        )
        plt.show()
