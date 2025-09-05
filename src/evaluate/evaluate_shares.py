from src.utils.metrics import smape, kl_div, bias


def evaluate_distribution(y_true, y_pred):
    """Evaluate share distributions."""
    out = {
        "sMAPE_%": smape(y_true, y_pred),
        "KL": kl_div(y_true, y_pred),
        "Bias_item_vector": bias(y_true, y_pred).tolist(),
    }
    return out
