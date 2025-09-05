import polars as pl


def load_raw(path, schema):
    df = pl.read_parquet(path) if path.endswith(".parquet") else pl.read_csv(path)
    return df.with_columns(pl.col(schema["date"]).str.strptime(pl.Date, strict=False))


def compute_shares(df: pl.DataFrame, schema: dict, top_k: int, eps: float):
    # top-K items by total volume
    totals = (
        df.group_by(schema["item"]) 
          .agg(pl.col(schema["qty"]).sum().alias("qty_total"))
          .sort("qty_total", descending=True)
          .head(top_k)
    )
    top_items = totals[schema["item"]].to_list()
    df = df.filter(pl.col(schema["item"]).is_in(top_items))

    df = (
        df
        .with_columns(pl.all().exclude(schema["qty"]))
        .with_columns(pl.col(schema["qty"]).cast(pl.Float64))
        .with_columns(pl.col(schema["qty"]).fill_null(0.0))
        .with_columns(
            (
                pl.col(schema["qty"]) /
                pl.sum(schema["qty"]).over([schema["date"], schema["category"]])
            ).alias("share_raw")
        )
    )

    df = df.with_columns(
        pl.when(pl.col("share_raw") < eps).then(eps).otherwise(pl.col("share_raw")).alias("share_clip")
    )

    df = df.with_columns(
        (
            pl.col("share_clip") /
            pl.sum("share_clip").over([schema["date"], schema["category"]])
        ).alias("share")
    )

    return df.filter(pl.col("share").is_not_null())
