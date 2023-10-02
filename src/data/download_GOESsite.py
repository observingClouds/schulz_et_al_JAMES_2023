#!

"""Download GOES16 satellite data over the site."""
import datetime as dt
import os

import pandas as pd
from omegaconf import OmegaConf

cfg = OmegaConf.load("../../config/paths.cfg")
output_fmt = cfg.OBS.SATELLITES.GOES16.CH13.filename_fmt
dates = pd.date_range(dt.datetime(2020, 1, 9), dt.datetime(2020, 2, 1))
for date in dates:
    date_strr = date.strftime("%Y%m%d")
    command = (
        "GOES16_download -k 13 -r 7 24 -61 -44 -d {date} -t 1 10 -o '{out}'".format(
            date=date_strr, out=output_fmt
        )
    )
    os.system(f"{command}")
