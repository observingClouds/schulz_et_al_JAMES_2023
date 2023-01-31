#!/work/mh0010/m300408/envs/covariability/bin/python
# SBATCH --job-name=agreementNN
# SBATCH --partition=gpu
# SBATCH --exclusive
# SBATCH -t 8:00:00
# SBATCH --account=mh0010
# SBATCH -o ../../logs/agreementNN.%j
# SBATCH -e ../../logs/agreementNN.%j

import datetime as dt

import dask
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
import xarray as xr
import zarr

experiment = 2
label_map = {"Sugar": 0, "Fish": 3, "Flowers": 2, "Flower": 2, "Gravel": 1}
label_map_rv = {0: "Sugar", 1: "Gravel", 2: "Flowers", 3: "Fish"}
lat0, lat1, lon0, lon1 = [7.5, 17, -60.25, -45]
color_dict = {
    "Sugar": "#A1D791",
    "Fish": "#2281BB",
    "Gravel": "#3EAE47",
    "Flowers": "#93D2E2",
}


def iou_one_class_from_annos(arr1, arr2, return_iou=False):
    """Returns the IoU from lists of [x, y, w, h] annotations. Image size must
    be given because arrays are created internally. If return_iou is True, the
    actual IoU score is computed, otherwise i, u will be returned.

    >>> iou_one_class_from_annos([False,True],[False,False],return_iou=True)
    0.0
    >>> iou_one_class_from_annos([False,True],[True,False],return_iou=True)
    0.0
    >>> iou_one_class_from_annos([False,True],[True,True],return_iou=True)
    0.5
    >>> iou_one_class_from_annos([False,False],[False,False],return_iou=True)
    np.nan
    """
    i = intersect_from_arrs(arr1, arr2)
    u = union_from_arrs(arr1, arr2)
    if return_iou:
        if u > 0:
            return i / u
        # only exists in one of the inputs
        elif u == 0 and np.any([arr1, arr2]):
            return 0
        # no identifications in any input
        else:
            return np.nan
    else:
        return i, u


def wh2xy(x, y, w, h):
    """Converts [x, y, w, h] to [x1, y1, x2, y2], i.e. bottom left and top
    right coords.

    >>> helpers.wh2xy(10, 20, 30, 1480)
    (10, 20, 39, 1499)
    """
    return x, y, x + w - 1, y + h - 1


def intersect_from_arrs(arr1, arr2):
    """Applies bitwise_and followed by a sum.

    Note that the sum operation is expensive.
    """
    return np.count_nonzero(np.bitwise_and(arr1, arr2))


def union_from_arrs(arr1, arr2):
    """Applies bitwise_or followed by a sum.

    Note that the sum operation is expensive.
    """
    return np.count_nonzero(np.bitwise_or(arr1, arr2))


def get_date_from_filename(fn):
    fn_parts = fn.split("/")
    try:
        date = dt.datetime.strptime(fn_parts[-1].split("_")[4], "Day%Y%m%d")
    except ValueError:
        date = dt.datetime.strptime(fn_parts[-1].split("_")[-3], "TrueColor%Y%m%d")
    return date


def create_mask(boxes, labels, out, label_map=None):
    """Create or add mask to array."""
    if label_map is None:
        label_map = {"Sugar": 0, "Fish": 3, "Flower": 2, "Gravel": 1}
    xy_boxes = [wh2xy(*b) for b in boxes]

    for lb, lab in enumerate(labels):
        mask_layer = label_map[lab]
        x1, y1, x2, y2 = np.array(xy_boxes[lb]).astype(int)
        out[x1:x2, y1:y2, mask_layer] = True

    return out


def interSection(arr1, arr2):  # finding common elements
    values = list(filter(lambda x: x in arr1, arr2))
    return values


def merge_mask(mask):
    """Merge mask along time dimension (has to be first dimension)"""
    return np.any(~np.isnan(mask.astype("float")), axis=0)


def identify_where_class_missing(arr1, arr2):
    """Identify which input array does not contain any labels.

    >>> identify_where_class_missing([[True, False],[True, False]],
            [[False, False],[False, False]])
    2
    >>> identify_where_class_missing([False],[True])
    1
    >>> identify_where_class_missing([False],[False])
    3
    >>> identify_where_class_missing([True],[True])
    0
    """
    missing_arr1 = not np.any(arr1)
    missing_arr2 = not np.any(arr2)
    if not missing_arr1 and not missing_arr2:
        return 0
    elif missing_arr1 and not missing_arr2:
        return 1
    elif missing_arr2 and not missing_arr1:
        return 2
    elif missing_arr1 and missing_arr2:
        return 3


for DOM in [1, 2]:
    fn_ICON_IR = (
        "../../data/external/NN_classifications/"
        f"GOES16_CH13_classifications_exp{experiment}_RTTOVonICONdomain_DOM0{DOM}.zarr"
    )
    fn_ABI_IR = (
        "../../data/external/NN_classifications/"
        f"GOES16_CH13_classifications_ABIonICONdomain_DOM0{DOM}.zarr"
    )

    mask_ICON_IR_DOM01 = xr.open_zarr(fn_ICON_IR)
    mask_ABI_IR_DOM01 = xr.open_zarr(fn_ABI_IR)

    results = {}

    print("Find common times to all datasets")
    times_of_interest = sorted(np.unique(mask_ABI_IR_DOM01.time))
    times_of_interest_floor = sorted(
        np.unique(mask_ABI_IR_DOM01.time.dt.floor(freq="1T"))
    )

    sizes_calculated = False

    for i in tqdm.tqdm(range(len(times_of_interest))):
        mask_ABI_IR_DOM01_timesVIS = mask_ABI_IR_DOM01.sel(time=times_of_interest[i])
        try:
            mask_ICON_IR_DOM01_timesVIS = mask_ICON_IR_DOM01.sel(
                time=mask_ABI_IR_DOM01_timesVIS.time,
                method="nearest",
                tolerance=dt.timedelta(minutes=30),
            )
        except KeyError:
            continue

        mask_ABI_IR_DOM01_timesVIS = mask_ABI_IR_DOM01_timesVIS.sel(
            latitude=slice(lat1, lat0), longitude=slice(lon0, lon1)
        )
        mask_ICON_IR_DOM01_timesVIS = mask_ICON_IR_DOM01_timesVIS.sel(
            latitude=slice(lat1, lat0), longitude=slice(lon0, lon1)
        )

        if sizes_calculated is False:
            size_ICON_IR_DOM01 = len(mask_ICON_IR_DOM01_timesVIS.latitude) * len(
                mask_ICON_IR_DOM01_timesVIS.longitude
            )
            size_ABI_IR_DOM01 = len(mask_ABI_IR_DOM01_timesVIS.latitude) * len(
                mask_ABI_IR_DOM01_timesVIS.longitude
            )
            sizes_calculated = True

        # for d, date in enumerate(tqdm.tqdm(dates_in_all_datasets)):
        pattern_results = {}

        for pattern in ["Sugar", "Gravel", "Flowers", "Fish"]:
            arr_ABI_IR = mask_ABI_IR_DOM01_timesVIS.mask.sel(pattern=pattern)
            arr_ICON_IR = mask_ICON_IR_DOM01_timesVIS.mask.sel(pattern=pattern)
            pattern_results[pattern] = {}

            if "time" in arr_ICON_IR.dims:
                merged_mask_ICON_IR = merge_mask(arr_ICON_IR)
            else:
                merged_mask_ICON_IR = arr_ICON_IR
            if "time" in arr_ABI_IR.dims:
                merged_mask_ABI_IR = merge_mask(arr_ABI_IR)
            else:
                merged_mask_ABI_IR = arr_ABI_IR

            assert merged_mask_ICON_IR.shape == merged_mask_ABI_IR.shape
            assert len(merged_mask_ICON_IR.latitude) > 0
            assert len(merged_mask_ICON_IR.longitude) > 0

            pattern_results[pattern]["area_fraction_IIR"] = (
                np.count_nonzero(merged_mask_ICON_IR.fillna(False)) / size_ICON_IR_DOM01
            )
            pattern_results[pattern]["area_fraction_AIR"] = (
                np.count_nonzero(merged_mask_ABI_IR.fillna(False)) / size_ABI_IR_DOM01
            )
            iou_IIR_AIR = iou_one_class_from_annos(
                merged_mask_ICON_IR.fillna(False).astype(bool).load(),
                merged_mask_ABI_IR.fillna(False).astype(bool).load(),
                return_iou=True,
            )
            pattern_results[pattern]["iou_IIR_AIR"] = iou_IIR_AIR
            pattern_results[pattern]["missing_IIR_AIR"] = identify_where_class_missing(
                merged_mask_ICON_IR.fillna(False), merged_mask_ABI_IR.fillna(False)
            )

        results[times_of_interest[i]] = pattern_results

    df = pd.DataFrame.from_dict(results, orient="index")
    df.to_pickle(
        f"../../data/intermediate/agreement_results_ABI-IR_vs_ICON-DOM0{DOM}_exp{experiment}.pkl"
    )
