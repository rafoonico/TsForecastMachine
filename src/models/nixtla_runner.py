import polars as pl
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS, TFT
from neuralforecast.losses.pytorch import SMAPE


def to_nf_df(df_pl: pl.DataFrame, schema: dict, target_col: str):
    df = df_pl.select([
        pl.col(schema["item"]).alias("unique_id"),
        pl.col(schema["date"]).alias("ds"),
        pl.col(target_col).alias("y")
    ]).sort(["unique_id", "ds"]).to_pandas()
    return df


def fit_predict_nhits(df_train, df_future, freq, input_size, h, max_steps=400):
    model = NHITS(h=h, input_size=input_size, max_steps=max_steps, loss=SMAPE(), scaler_type="standard")
    nf = NeuralForecast(models=[model], freq=freq)
    nf.fit(df_train)
    yhat = nf.predict(df_future)
    return yhat  # columns: unique_id, ds, NHITS


def normalize_to_shares(yhat_df: pd.DataFrame, items_map: pd.DataFrame):
    out = yhat_df.merge(items_map, on="unique_id", how="left")
    out["yhat_pos"] = out.iloc[:, 2].clip(lower=0)
    out["cat_sum"] = out.groupby(["category_id", "ds"])["yhat_pos"].transform("sum")
    out["share_hat"] = out["yhat_pos"] / out["cat_sum"].replace(0, 1e-9)
    return out
