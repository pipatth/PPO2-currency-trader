import pandas as pd
import v20
import yaml
import os
from datetime import datetime, timedelta

# DIR
CFG_DIR = os.getenv("HOME")

# config file
cfgStream = open(os.path.join(CFG_DIR, ".v20.conf"), "r")
cfg = yaml.load(cfgStream, Loader=yaml.FullLoader)
api = v20.Context(
    hostname=cfg["hostname"],
    port=cfg["port"],
    ssl=cfg["ssl"],
    datetime_format=cfg["datetime_format"],
    token=cfg["token"],
)

# get hourly candles from date, return dataframe of candles
def get_hourly_candle(dt, count, asset):
    # arguments for v20
    kwargs = {
        "accountID": cfg["active_account"],
        "instrument": asset,
        "granularity": "H1",
        "smooth": False,
        "price": "BAM",
        "count": count,
        "fromTime": dt,
        "dailyAlignment": 0,
        "alignmentTimezone": "UTC",
    }
    r = api.instrument.candles(**kwargs)
    ls = [c.dict() for c in r.get("candles", 200)]
    df = pd.io.json.json_normalize(ls)
    df["time"], df["sec"] = df["time"].str.split(".", 1).str
    df.index = df["time"]
    df = df.drop(["time", "sec"], axis=1)
    df = df.astype(float)
    return df


# save hourly candle data
def get_data(asset, dt, count, filename):
    # convert date to isoformat
    dt = datetime.strptime(dt, "%Y-%m-%d")
    dt = dt.isoformat("T") + "Z"

    # get data in chunk (max size 5000)
    results = []
    while count > 0:
        chunk_sz = min(count, 5000)
        d = get_hourly_candle(dt, chunk_sz, asset)
        d = d.reset_index()
        results.append(d)
        count -= len(d)
        dt = datetime.strptime(d["time"].max(), "%Y-%m-%dT%H:%M:%S")
        dt = dt + timedelta(hours=1)
        dt = dt.isoformat("T") + "Z"
        print("Loading {:} data from {:} for {:} bars".format(asset, dt, chunk_sz))

    # save to csv
    df = pd.concat(results, ignore_index=True)
    df = df.sort_values("time")
    df = df.reset_index(drop=True)
    df.to_csv(filename, sep="\t", index=False)
    print("Saved to " + filename)


# download training, testing data
get_data("USD_CAD", "2014-01-01", 30000, "USD_CAD_train.tsv")
get_data("USD_CAD", "2019-01-01", 2000, "USD_CAD_test.tsv")
