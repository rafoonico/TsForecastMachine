import polars as pl
import numpy as np


def add_time_features(df: pl.DataFrame, schema: dict):
    month = pl.col(schema["date"]).dt.month()
    return df.with_columns([
        (np.sin(2*np.pi*(month/12))).alias("month_sin"),
        (np.cos(2*np.pi*(month/12))).alias("month_cos")
    ])


def item_level_relatives(df: pl.DataFrame, schema: dict):
    mean_price = pl.mean(schema["price"]).over([schema["date"], schema["category"]])
    return df.with_columns((pl.col(schema["price"]) / (mean_price + 1e-6)).alias("price_rel"))
