import yaml
import polars as pl
import numpy as np

from src.data_prep.build_shares import load_raw, compute_shares
from src.data_prep.features import add_time_features, item_level_relatives
from src.models.nixtla_runner import to_nf_df, fit_predict_nhits, normalize_to_shares
from src.models.lstm_softmax import build_lstm_softmax
from src.evaluate.evaluate_shares import evaluate_distribution
from src.windows.rolling_origin import rolling_splits


def main():
    cfg = yaml.safe_load(open("config/config.yaml"))
    sch = cfg["schema"]
    freq = cfg["freq"]
    top_k = cfg["top_k"]
    tw = cfg["tw"]
    h = cfg["h"]
    eps = cfg["epsilon"]

    df = load_raw("data/raw/sales.csv", sch)
    df = item_level_relatives(df, sch)
    df = add_time_features(df, sch)
    df = compute_shares(df, sch, top_k, eps)

    # Example: build dataset for NeuralForecast
    df_nf = to_nf_df(df, sch, target_col=sch["qty"])

    # placeholder for train/val/test split and model training
    # df_train, df_future = ...
    # yhat = fit_predict_nhits(df_train=df_train, df_future=df_future, freq=freq,
    #                          input_size=12, h=h, max_steps=400)
    # shares_hat = normalize_to_shares(yhat, items_map)
    # out = evaluate_distribution(y_true, y_pred)
    # print(out)


if __name__ == "__main__":
    main()
