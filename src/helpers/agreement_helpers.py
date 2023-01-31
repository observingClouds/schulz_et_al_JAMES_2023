def read_pkls(pkl_fmt):
    """Resolve pickle filename format and load pickles to dataframe."""
    import glob

    import numpy as np
    import pandas as pd

    pkl_fns = np.array(sorted(glob.glob(pkl_fmt)))
    dataframes = [None] * len(pkl_fns)
    for f, file in enumerate(pkl_fns):
        dataframes[f] = pd.read_pickle(file)
    df_all = pd.concat(dataframes)
    return df_all


def convert_df_dict_entries_to_columns(df_all):
    from functools import reduce

    import pandas as pd

    Sugar_df = df_all.Sugar.apply(pd.Series)
    Gravel_df = df_all.Gravel.apply(pd.Series)
    Flowers_df = df_all.Flowers.apply(pd.Series)
    Fish_df = df_all.Fish.apply(pd.Series)

    for df, p in zip(  # noqa: B905
        [Sugar_df, Gravel_df, Fish_df, Flowers_df],
        ["Sugar", "Gravel", "Fish", "Flowers"],
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
    return df_all
